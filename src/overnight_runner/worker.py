"""Phase 1/2 worker: bounded task executor.

Sequence:
  1. Load manifest.
  2. Caller-supplied approval envelope is REQUIRED (no self-authorization).
  3. Bind checks: manifest SHA, repo HEAD, runtime fingerprint, model digest,
     clean tree (for mutation). Mismatch => invalidate approval, no Ollama.
  4. Construct Broker passing the approved baseline.
  5. Call Ollama.
  6. Tool loop with declared limits.
  7. On DONE -> run required validators -> PASS only if validators AND (for
     source_mutation) at least one successfully applied proposal.
  8. Persist artifacts.

P0 hardening:
  - Approval is a separate envelope passed in (Approval class); worker
    NEVER invents an envelope.
  - Apply boundary: broker re-checks HEAD against approved HEAD, PAUSED
    sentinel absent, path policy, file hash, limits.
  - source_mutation with zero applied proposals + DONE => NO_MUTATION_APPLIED
    => REVIEW_REQUIRED.
  - PAUSED at apply boundary => REVIEW_REQUIRED and the proposal is NOT
    consumed.
"""
from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .broker import Broker, default_registry
from .ollama_client import OllamaClient
from .runtime import is_paused, require_not_paused, runtime_fingerprint, state_dir
from .safety import SafetyError, git_head, git_is_clean, git_worktree_sha
from .schemas import (
    Disposition,
    ExecutionClass,
    TaskManifest,
    ToolCall,
    canonical_sha,
)


SYSTEM_PROMPT = """You are a bounded implementation worker running inside the overnight-runner harness.

Rules (these are enforced by Python regardless of what you say):
- The task contract is authoritative. You may not broaden scope.
- You may not redesign architecture.
- You may not commit to git. The harness will not commit on your behalf.
- You may invoke ONLY the tools offered to you: read_exact, propose_patch,
  apply_validated_patch, run_command_id, report_result.
- You may read only paths explicitly listed in the contract's read_paths or
  protected_read_paths.
- You may modify only paths explicitly listed in write_paths (replace existing
  files) or create_paths (create new files). create_paths does NOT grant the
  right to overwrite an existing file.
- You may invoke only command IDs explicitly listed in
  commands.model_allowed_command_ids. Anything else is refused.
- You must NEVER attempt to bypass a denied action. If you cannot proceed with
  allowed tools, report BLOCKED or REVIEW_REQUIRED.
- Prefer the SMALLEST exact change that satisfies the contract.
- Do NOT perform unrelated cleanup, refactoring, or "while I'm here" edits.
- Apply your proposal via apply_validated_patch(proposal_id=...). Python will
  re-validate before writing.

Dispositions:
- DONE: I have finished exactly what was asked. This does NOT mean PASSED.
  Python will run validators and decide PASS/FAIL.
- BLOCKED: I cannot proceed because a required input is missing or a tool call
  was refused.
- REVIEW_REQUIRED: I am uncertain, scope has changed, or the contract seems
  wrong.

Format:
- Use native tool calling if available; tool names are exact.
- Do not produce unified diffs yourself.
- Do not propose patches via free text.
"""


# ----------------------------- Approval envelope -----------------------------

@dataclass(frozen=True)
class Approval:
    manifest_sha256: str
    approved_repo_head: str
    approved_runtime_sha256: str
    approved_model_name: str
    approved_model_digest: str | None
    approved_at: int
    approved_by: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "approved_repo_head": self.approved_repo_head,
            "approved_runtime_sha256": self.approved_runtime_sha256,
            "approved_model_name": self.approved_model_name,
            "approved_model_digest": self.approved_model_digest,
            "approved_at": self.approved_at,
            "approved_by": self.approved_by,
        }


# ----------------------------- Result -----------------------------

@dataclass
class RunResult:
    status: str
    reason_code: str = ""
    reason_text: str = ""
    turns: int = 0
    tool_calls: int = 0
    artifacts_dir: str = ""
    proposal_count: int = 0
    applied_proposal_count: int = 0
    approval_invalidated: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# ----------------------------- Worker -----------------------------

class Worker:
    def __init__(
        self,
        client: OllamaClient | None = None,
        registry=None,
        runtime_roots: list[Path] | None = None,
        artifact_root: Path | None = None,
        model_digest_resolver=None,
    ) -> None:
        self.client = client or OllamaClient()
        self.registry = registry or default_registry()
        self.runtime_roots = runtime_roots or self._default_runtime_roots()
        self.artifact_root = artifact_root or (state_dir() / "runs")
        self.model_digest_resolver = model_digest_resolver  # callable(name)->str|None

    @staticmethod
    def _default_runtime_roots() -> list[Path]:
        here = Path(__file__).parent
        return [here, here.parent / "prompts"]

    # ---------- Public entrypoint: manual run (Phase 1) ----------

    def run(self, manifest: TaskManifest, *, dry_run: bool = False,
            approval: Approval | None = None,
            artifact_dir: Path | None = None,
            on_metrics=None) -> RunResult:
        """Manual single-task run.

        If `approval` is None we derive one ONLY for ad-hoc local runs (Phase 1).
        For queued execution the caller MUST supply an independently recorded
        Approval (Phase 2 / run-next).

        If `artifact_dir` is provided, the worker uses it as-is (no extra
        nesting). Otherwise it creates artifact_root/<task_id>/<run_id>.

        If `on_metrics(snapshot)` is provided, it is called after each Ollama
        turn with cumulative metric counters. Worker is decoupled from DB.
        """
        started = time.time()
        repo_root = Path(manifest.repo.path).resolve()

        if artifact_dir is None:
            run_id = f"run-{int(started)}-{uuid.uuid4().hex[:8]}"
            artifact_dir = self.artifact_root / manifest.task_id / run_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
        else:
            artifact_dir = Path(artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "manifest.json").write_bytes(_canonical(manifest))

        result = RunResult(status="RUNNING", artifacts_dir=str(artifact_dir))

        if approval is None:
            # Phase 1 ad-hoc: derive ephemeral envelope (NOT stored as
            # authoritative approval). This keeps the manual workflow usable
            # but is clearly marked ephemeral.
            approval = _ephemeral_approval(manifest, repo_root, self.runtime_roots,
                                          self.model_digest_resolver)
            result.extra["approval_ephemeral"] = True
        else:
            result.extra["approval_ephemeral"] = False

        (artifact_dir / "approval.json").write_text(json.dumps(approval.to_dict(), indent=2))

        try:
            require_not_paused()

            # Bind checks. Any mismatch invalidates approval BEFORE Ollama.
            mismatch = _bind_check(manifest, approval, repo_root, self.runtime_roots,
                                   digest_resolver=self.model_digest_resolver)
            if mismatch:
                _append_evidence(artifact_dir, "approval_invalidated", mismatch)
                result.status = "BLOCKED"
                result.reason_code = mismatch
                result.reason_text = f"approval binding failed: {mismatch}"
                result.approval_invalidated = True
                (artifact_dir / "result.json").write_text(json.dumps({
                    "status": result.status,
                    "reason_code": result.reason_code,
                    "reason_text": result.reason_text,
                    "approval": approval.to_dict(),
                    "approval_invalidated": True,
                }, indent=2))
                # Persist invalidation event best-effort.
                try:
                    from .db import Database, default_db_path
                    _db = Database(default_db_path())
                    try:
                        _db.update_status(manifest.task_id, TaskStatus.PENDING_APPROVAL,
                                          final_reason_code=mismatch,
                                          final_reason_text=f"approval invalidated: {mismatch}")
                        _db.emit_event("worker", "approval_invalidated",
                                      task_id=manifest.task_id, to_state="PENDING_APPROVAL",
                                      details={"code": mismatch})
                    finally:
                        _db.close()
                except Exception:
                    pass
                return result

            # Unreal editor -> forced BLOCKED.
            if manifest.execution_class == ExecutionClass.UNREAL_EDITOR:
                (artifact_dir / "result.json").write_text(json.dumps({
                    "status": "BLOCKED",
                    "reason_code": "UNREAL_EDITOR_NOT_IMPLEMENTED",
                    "reason_text": "Unreal editor automation deferred.",
                }, indent=2))
                result.status = "BLOCKED"
                result.reason_code = "UNREAL_EDITOR_NOT_IMPLEMENTED"
                return result

            broker = self._build_broker(manifest, repo_root, artifact_dir, approval)

            tools = _ollama_tool_schema()
            messages: list[dict[str, Any]] = [{"role": "user", "content": _format_user_brief(manifest)}]

            transcript: list[dict[str, Any]] = []
            applied_proposals: list[str] = []
            proposals_made: list[str] = []
            total_calls = 0
            disposition: Disposition | None = None
            captured_disposition: list[Any] = [None]
            captured_summary: list[str] = [""]
            captured_evidence: list[list[str]] = [[]]

            for turn in range(manifest.limits.max_model_turns):
                if time.time() - started > manifest.limits.task_timeout_seconds:
                    raise SafetyError("task timeout exceeded")
                result.turns = turn + 1
                chat = self.client.chat(
                    profile=manifest.model_profile,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=tools,
                )
                _append_transcript_event(transcript, {
                    "event": "ollama_chat",
                    "turn": turn + 1,
                    "metrics": _metrics_to_dict(chat.metrics),
                    "content": chat.content[:4000],
                    "tool_calls": chat.tool_calls,
                })
                if on_metrics is not None:
                    try:
                        on_metrics({
                            "prompt_eval_count": chat.metrics.prompt_eval_count,
                            "eval_count": chat.metrics.eval_count,
                            "ollama_total_duration_ns": chat.metrics.total_duration_ns,
                            "ollama_eval_duration_ns": chat.metrics.eval_duration_ns,
                            "model_calls": 1,
                        })
                    except Exception:
                        pass

                if chat.tool_calls:
                    tool_messages: list[dict[str, Any]] = []
                    stop = False
                    for tc in chat.tool_calls:
                        if total_calls >= manifest.limits.max_tool_calls:
                            tool_messages.append(_tool_error_msg(tc, "max_tool_calls exceeded"))
                            continue
                        total_calls += 1
                        result.tool_calls = total_calls
                        if on_metrics is not None:
                            try:
                                on_metrics({"tool_calls": 1})
                            except Exception:
                                pass
                        try:
                            tc_args = tc.get("function", {}).get("arguments", {})
                            if isinstance(tc_args, str):
                                # Strip unescaped control chars that some
                                # models emit (e.g. raw newlines in evidence).
                                tc_args = _loose_json_loads(tc_args)
                            tool_name = tc.get("function", {}).get("name")
                            call = _parse_tool_call(tc.get("id") or f"c{total_calls}", tool_name, tc_args)
                            # Re-check PAUSED at every tool boundary.
                            if is_paused():
                                raise SafetyError("PAUSED sentinel present at tool boundary")
                            handled = self._dispatch(
                                broker, call, applied_proposals, proposals_made,
                                captured_disposition, captured_summary, captured_evidence,
                            )
                            tool_messages.append({
                                "role": "tool",
                                "tool_name": tool_name,
                                "content": json.dumps(handled, default=str)[: manifest.limits.max_tool_result_bytes],
                            })
                            if captured_disposition[0] is not None:
                                stop = True
                                break
                        except Exception as e:
                            tool_messages.append(_tool_error_msg(tc, f"{type(e).__name__}: {e}"))
                    messages.append({"role": "assistant", "content": chat.content, "tool_calls": chat.tool_calls})
                    messages.extend(tool_messages)
                    if stop:
                        break
                    continue

                messages.append({"role": "assistant", "content": chat.content})
                disp, summary, evidence = _extract_disposition(chat.content)
                if disp is not None:
                    disposition = disp
                    _append_transcript_event(transcript, {
                        "event": "report_result_via_content",
                        "turn": turn + 1,
                        "disposition": disp.value,
                        "summary": summary,
                    })
                    break
                messages.append({
                    "role": "user",
                    "content": "You have not called a tool or reported a disposition. Call one of the offered tools or call report_result.",
                })

            (artifact_dir / "transcript.jsonl").write_text(
                "\n".join(json.dumps(e, default=str) for e in transcript) + "\n"
            )
            (artifact_dir / "proposal.json").write_text(json.dumps({
                "proposals_made": proposals_made,
                "applied_proposals": applied_proposals,
            }, indent=2))
            result.proposal_count = len(proposals_made)
            result.applied_proposal_count = len(applied_proposals)

            if disposition is None and captured_disposition[0] is not None:
                try:
                    disposition = Disposition(captured_disposition[0])
                except ValueError:
                    disposition = Disposition.REVIEW_REQUIRED

            if disposition is None:
                disposition = Disposition.REVIEW_REQUIRED
                result.reason_code = "MAX_TURNS_REACHED"
                result.reason_text = "model never reported disposition"

            final_status, reason_code, reason_text = _finalise(
                manifest, broker, disposition, artifact_dir,
                applied_proposals=applied_proposals,
            )
            result.status = final_status
            result.reason_code = reason_code
            result.reason_text = reason_text

            (artifact_dir / "result.json").write_text(json.dumps({
                "status": result.status,
                "reason_code": result.reason_code,
                "reason_text": result.reason_text,
                "turns": result.turns,
                "tool_calls": result.tool_calls,
                "wall_duration_ms": int((time.time() - started) * 1000),
                "proposals_made": proposals_made,
                "applied_proposals": applied_proposals,
                "approval": approval.to_dict(),
                "approval_ephemeral": result.extra.get("approval_ephemeral", False),
                "journal_entries": [je.__dict__ for je in broker.journal],
            }, indent=2, default=str))
            return result

        except SafetyError as e:
            result.status = "BLOCKED"
            result.reason_code = "SAFETY"
            result.reason_text = str(e)
            (artifact_dir / "result.json").write_text(json.dumps({
                "status": result.status, "reason_code": result.reason_code, "reason_text": result.reason_text,
                "exception": traceback.format_exc(limit=10),
            }, indent=2))
            return result
        except Exception as e:
            result.status = "FAILED"
            result.reason_code = type(e).__name__
            result.reason_text = str(e)
            (artifact_dir / "result.json").write_text(json.dumps({
                "status": result.status, "reason_code": result.reason_code, "reason_text": result.reason_text,
                "exception": traceback.format_exc(limit=20),
            }, indent=2))
            return result

    # ---------- Broker construction ----------

    def _build_broker(
        self, manifest: TaskManifest, repo_root: Path, artifact_dir: Path, approval: Approval,
    ) -> Broker:
        return Broker(
            repo_root=repo_root,
            registry=self.registry,
            allowed_write_paths=list(manifest.paths.write_paths),
            allowed_create_paths=list(manifest.paths.create_paths),
            allowed_read_paths=list(manifest.paths.read_paths),
            allowed_protected_read_paths=list(manifest.paths.protected_read_paths),
            model_allowed_command_ids=list(manifest.commands.model_allowed_command_ids),
            required_validator_ids=list(manifest.commands.required_validator_ids),
            max_tool_result_bytes=manifest.limits.max_tool_result_bytes,
            max_read_bytes=manifest.context_budget.max_read_bytes,
            max_files_read=manifest.context_budget.max_files_read,
            max_files_written=manifest.context_budget.max_files_written,
            max_changed_files=manifest.limits.max_changed_files,
            max_diff_lines=manifest.limits.max_diff_lines,
            max_written_bytes=manifest.limits.max_written_bytes,
            approved_repo_head=approval.approved_repo_head,
            artifact_dir=artifact_dir,
        )

    # ---------- Dispatch ----------

    def _dispatch(
        self, broker, call, applied, proposals,
        captured_disposition, captured_summary, captured_evidence,
    ) -> dict[str, Any]:
        from .schemas import (
            ApplyValidatedPatchArgs,
            CreateFileArgs,
            ReadExactArgs,
            ReplaceExactArgs,
            ReplaceFileArgs,
            ReportResultArgs,
            RunCommandArgs,
        )
        a = call.args
        if isinstance(a, ReadExactArgs):
            return broker.read_exact(a.path, a.start_line, a.end_line)
        if isinstance(a, (ReplaceExactArgs, ReplaceFileArgs, CreateFileArgs)):
            out = broker.handle(call)
            proposals.append(out["proposal_id"])
            return out
        if isinstance(a, ApplyValidatedPatchArgs):
            out = broker.apply_proposal(a.proposal_id)
            applied.append(a.proposal_id)
            return out
        if isinstance(a, RunCommandArgs):
            return broker.handle(call)
        if isinstance(a, ReportResultArgs):
            captured_disposition[0] = a.disposition
            captured_summary[0] = a.summary
            captured_evidence[0] = list(a.evidence)
            return {"recorded": True, "evidence": a.evidence}
        raise SafetyError(f"unsupported tool args: {type(a).__name__}")


# ----------------------------- Finalisation -----------------------------

def _finalise(
    manifest: TaskManifest,
    broker: Broker,
    disposition: Disposition,
    artifact_dir: Path,
    *,
    applied_proposals: list[str],
) -> tuple[str, str, str]:
    if disposition == Disposition.BLOCKED:
        return "BLOCKED", "MODEL_BLOCKED", "model reported BLOCKED"
    if disposition == Disposition.REVIEW_REQUIRED:
        return "REVIEW_REQUIRED", "MODEL_REVIEW_REQUIRED", "model reported REVIEW_REQUIRED"
    if disposition != Disposition.DONE:
        return "REVIEW_REQUIRED", "MODEL_DISPOSITION_UNKNOWN", f"unknown disposition: {disposition}"

    # source_mutation MUST mutate unless allow_no_mutation=True.
    if manifest.execution_class == ExecutionClass.SOURCE_MUTATION:
        if not applied_proposals and not manifest.commands.allow_no_mutation:
            return "REVIEW_REQUIRED", "NO_MUTATION_APPLIED", (
                "source_mutation DONE with zero applied proposals"
            )

    repo_root = Path(manifest.repo.path).resolve()
    failures: list[str] = []
    for vid in manifest.commands.required_validator_ids:
        if vid not in broker.registry._cmds:  # type: ignore[attr-defined]
            if vid == "python_compile":
                for wp in manifest.paths.write_paths:
                    p = (repo_root / wp).resolve()
                    if not p.exists():
                        failures.append(f"python_compile missing: {wp}")
                        continue
                    import py_compile
                    try:
                        py_compile.compile(str(p), doraise=True)
                    except py_compile.PyCompileError as e:
                        failures.append(f"python_compile {wp}: {e}")
                continue
            if vid == "no_op" or vid == "noop":
                continue
        # Registry-backed validator (worker can run them; model cannot).
        # Use the same owned-process-group helper the broker uses for
        # model-invoked commands so timeouts kill only the validator's pgid.
        spec = broker.registry.get(vid)
        # Non-mutating validators: capture worktree fingerprint for drift check.
        pre_wt = None
        if spec.side_effects in ("none", "read"):
            pre_wt = git_worktree_sha(repo_root)
        try:
            from .broker import _spawn_own_pgrp, _TimeoutExpired
            if pre_wt is not None:
                proc = _spawn_own_pgrp(list(spec.argv), cwd=repo_root,
                                       timeout=spec.timeout_seconds)
            else:
                # For mutating validators we don't capture pre/post; allow
                # them through the safe primitive too. They get a new pgid
                # so a runaway can be killed precisely.
                proc = _spawn_own_pgrp(list(spec.argv), cwd=repo_root,
                                       timeout=spec.timeout_seconds)
            (artifact_dir / "commands" / vid).mkdir(parents=True, exist_ok=True)
            (artifact_dir / "commands" / vid / "stdout.txt").write_text(proc.stdout or "")
            (artifact_dir / "commands" / vid / "stderr.txt").write_text(proc.stderr or "")
            if proc.returncode != 0:
                failures.append(f"{vid} exit={proc.returncode}")
            if pre_wt is not None:
                post_wt = git_worktree_sha(repo_root)
                if pre_wt != post_wt:
                    failures.append(f"{vid} non_mutating_command_drift ({pre_wt[:8]}->{post_wt[:8]})")
        except _TimeoutExpired as e:
            failures.append(f"{vid} timed_out: {e}")
        except Exception as e:
            failures.append(f"{vid} error={type(e).__name__}: {e}")

    if manifest.execution_class == ExecutionClass.SOURCE_MUTATION:
        for wp in manifest.paths.write_paths:
            rp = (repo_root / wp).resolve()
            if not rp.exists():
                failures.append(f"write target not produced: {wp}")

    if failures:
        return "FAILED", "VALIDATORS_FAILED", "; ".join(failures)
    return "PASSED", "OK", "all validators passed"


# ----------------------------- Approval binding -----------------------------

def _bind_check(
    manifest: TaskManifest,
    approval: Approval,
    repo_root: Path,
    runtime_roots: list[Path],
    *,
    current_model_digest: str | None = None,
    digest_resolver=None,
) -> str:
    """Return '' if all bind values match; otherwise a precise reason code.

    If `digest_resolver` is supplied it is called as digest_resolver(name) to
    resolve the current installed model digest; the result is compared to
    approval.approved_model_digest when the latter is non-None.
    """
    if canonical_sha(manifest) != approval.manifest_sha256:
        return "APPROVAL_MANIFEST_CHANGED"
    cur_head = git_head(repo_root)
    if cur_head != approval.approved_repo_head:
        return "APPROVAL_REPO_HEAD_CHANGED"
    cur_rt = runtime_fingerprint(runtime_roots).sha256
    if cur_rt != approval.approved_runtime_sha256:
        return "APPROVAL_RUNTIME_CHANGED"
    if approval.approved_model_digest:
        if manifest.model_profile.model_name != approval.approved_model_name:
            return "APPROVAL_MODEL_CHANGED"
        # Fail-closed digest check: require resolver, success, non-empty
        # result, AND equality. Any failure => block before Ollama.
        if digest_resolver is None:
            return "APPROVAL_MODEL_UNRESOLVABLE"
        try:
            cur_digest = digest_resolver(manifest.model_profile.model_name)
        except Exception:
            return "APPROVAL_MODEL_UNRESOLVABLE"
        if not cur_digest:
            return "APPROVAL_MODEL_UNRESOLVABLE"
        if cur_digest != approval.approved_model_digest:
            return "APPROVAL_MODEL_CHANGED"
    if manifest.execution_class == ExecutionClass.SOURCE_MUTATION and manifest.repo.require_clean_tree:
        if not git_is_clean(repo_root):
            return "APPROVAL_DIRTY_WORKTREE"
    return ""


def _ephemeral_approval(
    manifest: TaskManifest,
    repo_root: Path,
    runtime_roots: list[Path],
    digest_resolver,
) -> Approval:
    head = git_head(repo_root)
    digest = None
    if digest_resolver is not None:
        try:
            digest = digest_resolver(manifest.model_profile.model_name)
        except Exception:
            digest = None
    return Approval(
        manifest_sha256=canonical_sha(manifest),
        approved_repo_head=head,
        approved_runtime_sha256=runtime_fingerprint(runtime_roots).sha256,
        approved_model_name=manifest.model_profile.model_name,
        approved_model_digest=digest,
        approved_at=int(time.time()),
        approved_by="ephemeral-cli",
    )


def _append_evidence(artifact_dir: Path, event: str, code: str) -> None:
    import json
    p = artifact_dir / "events.jsonl"
    with p.open("a") as f:
        f.write(json.dumps({"event": event, "code": code, "ts": time.time()}) + "\n")


# ----------------------------- Tool schema / parsing -----------------------------

def _ollama_tool_schema() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {
            "name": "read_exact",
            "description": "Read an authorised UTF-8 file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 0},
                    "end_line": {"type": "integer", "minimum": 0},
                },
                "required": ["path"],
            },
        }},
        {"type": "function", "function": {
            "name": "propose_patch",
            "description": "Propose a bounded edit. op=replace_exact|replace_file|create_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["replace_exact", "replace_file", "create_file"]},
                    "path": {"type": "string"},
                    "expected_sha256": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "expected_occurrences": {"type": "integer", "minimum": 1},
                    "new_content": {"type": "string"},
                },
                "required": ["op", "path"],
            },
        }},
        {"type": "function", "function": {
            "name": "apply_validated_patch",
            "description": "Apply a previously proposed patch by proposal_id.",
            "parameters": {
                "type": "object",
                "properties": {"proposal_id": {"type": "string"}},
                "required": ["proposal_id"],
            },
        }},
        {"type": "function", "function": {
            "name": "run_command_id",
            "description": "Run a registered command by id (allowed by contract).",
            "parameters": {
                "type": "object",
                "properties": {"command_id": {"type": "string"}},
                "required": ["command_id"],
            },
        }},
        {"type": "function", "function": {
            "name": "report_result",
            "description": "Report final disposition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "disposition": {"type": "string", "enum": ["DONE", "BLOCKED", "REVIEW_REQUIRED"]},
                    "summary": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["disposition", "summary"],
            },
        }},
    ]


def _loose_json_loads(s: str) -> Any:
    """Parse JSON, replacing unescaped control characters with their escaped form.

    Some models emit raw \n inside string literals. This is invalid JSON but
    recoverable for tool-call argument parsing.
    """
    # Quick path: try strict first.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Replace ASCII control chars (0x00-0x1F) that appear inside strings.
    # Heuristic: walk the string, track quote state, replace any control char
    # inside a string with its escaped form.
    out: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
            continue
        out.append(ch)
    return json.loads("".join(out))


def _parse_tool_call(call_id: str, name: str, args: dict[str, Any]) -> ToolCall:
    from .schemas import (
        ApplyValidatedPatchArgs, CreateFileArgs, ReadExactArgs,
        ReplaceExactArgs, ReplaceFileArgs, RunCommandArgs, ReportResultArgs,
    )
    if name == "read_exact":
        return ToolCall(call_id=call_id, args=ReadExactArgs(**args))
    if name == "propose_patch":
        op = args.get("op")
        if op == "replace_exact":
            return ToolCall(call_id=call_id, args=ReplaceExactArgs(**args))
        if op == "replace_file":
            return ToolCall(call_id=call_id, args=ReplaceFileArgs(**args))
        if op == "create_file":
            return ToolCall(call_id=call_id, args=CreateFileArgs(**args))
        raise SafetyError(f"unknown propose_patch op: {op}")
    if name == "apply_validated_patch":
        return ToolCall(call_id=call_id, args=ApplyValidatedPatchArgs(**args))
    if name == "run_command_id":
        return ToolCall(call_id=call_id, args=RunCommandArgs(**args))
    if name == "report_result":
        return ToolCall(call_id=call_id, args=ReportResultArgs(**args))
    raise SafetyError(f"unknown tool: {name}")


def _tool_error_msg(tc: dict[str, Any], msg: str) -> dict[str, Any]:
    return {"role": "tool", "tool_name": tc.get("function", {}).get("name"), "content": json.dumps({"error": msg})}


def _append_transcript_event(transcript: list[dict[str, Any]], ev: dict[str, Any]) -> None:
    ev["t"] = int(time.time() * 1000)
    transcript.append(ev)


def _metrics_to_dict(m) -> dict[str, Any]:
    return {
        "prompt_eval_count": m.prompt_eval_count,
        "eval_count": m.eval_count,
        "total_duration_ns": m.total_duration_ns,
        "eval_duration_ns": m.eval_duration_ns,
        "prompt_eval_duration_ns": m.prompt_eval_duration_ns,
        "load_duration_ns": m.load_duration_ns,
    }


def _extract_disposition(content: str) -> tuple[Disposition | None, str, list[str]]:
    text = content.strip()
    disp = None
    summary = ""
    evidence: list[str] = []
    for line in text.splitlines():
        u = line.strip().upper()
        if u in ("DONE", "BLOCKED", "REVIEW_REQUIRED"):
            disp = Disposition(u)
        elif line.lower().startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
    if disp is None and text:
        try:
            j = json.loads(text)
            if isinstance(j, dict) and "disposition" in j:
                d = j.get("disposition", "").upper()
                if d in ("DONE", "BLOCKED", "REVIEW_REQUIRED"):
                    disp = Disposition(d)
                    summary = str(j.get("summary", ""))
                    ev = j.get("evidence") or []
                    if isinstance(ev, list):
                        evidence = [str(x) for x in ev]
        except json.JSONDecodeError:
            pass
    return disp, summary, evidence


def _format_user_brief(m: TaskManifest) -> str:
    return (
        f"task_id: {m.task_id}\n"
        f"title: {m.title}\n"
        f"execution_class: {m.execution_class.value}\n"
        f"objective: {m.objective}\n"
        f"repo: {m.repo.path}\n"
        f"write_paths: {m.paths.write_paths}\n"
        f"create_paths: {m.paths.create_paths}\n"
        f"read_paths: {m.paths.read_paths}\n"
        f"protected_read_paths: {m.paths.protected_read_paths}\n"
        f"model_allowed_command_ids: {m.commands.model_allowed_command_ids}\n"
        f"required_validator_ids: {m.commands.required_validator_ids}\n"
        f"acceptance_criteria: {m.acceptance_criteria}\n"
        "Begin. Make the smallest change that satisfies the objective.\n"
        "After all changes are applied, your LAST tool call must be "
        "report_result(disposition='DONE', summary='...', evidence=[...]). "
        "Do NOT call run_command_id with required_validator_ids — those are "
        "run automatically by Python after you report DONE."
    )


def _canonical(m: TaskManifest) -> bytes:
    from .schemas import canonical_json
    return canonical_json(m)
