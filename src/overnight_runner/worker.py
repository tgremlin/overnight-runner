"""Phase 1 worker: single-task manually-invoked sequential executor.

Sequence:
  1. Load + validate manifest.
  2. Preflight (repo, fingerprint, paused, dry-run approval binding).
  3. Call Ollama.
  4. Loop: tool calls -> broker -> feed results back -> next turn.
  5. On DONE -> run required validators -> PASS/FAIL.
  6. Persist artifacts.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .broker import Broker, default_registry
from .ollama_client import OllamaClient
from .runtime import require_not_paused, runtime_fingerprint, state_dir
from .safety import (
    SafetyError,
    git_head,
    git_is_clean,
    git_worktree_sha,
    sha256_bytes,
)
from .schemas import (
    Disposition,
    ExecutionClass,
    ModelProfile,
    TaskManifest,
    ToolCall,
    canonical_json,
    canonical_sha,
)


SYSTEM_PROMPT = """You are a bounded implementation worker running inside the overnight-runner harness.

Rules (these are enforced by Python regardless of what you say):
- The task contract is authoritative. You may not broaden scope.
- You may not redesign architecture.
- You may not commit to git. The harness will not commit on your behalf.
- You may only invoke the tools offered to you: read_exact, propose_patch, run_command_id, report_result.
- You may read only paths listed in the contract's read_paths (or protected_read_paths).
- You may modify only paths listed in the contract's write_paths / create_paths.
- You must NEVER attempt to bypass a denied action. If you cannot proceed with allowed tools, report BLOCKED or REVIEW_REQUIRED.
- Prefer the SMALLEST exact change that satisfies the contract.
- Do NOT perform unrelated cleanup, refactoring, or "while I'm here" edits.

Dispositions:
- DONE: I have finished exactly what was asked. This does NOT mean PASSED. Python will run validators and decide PASS/FAIL.
- BLOCKED: I cannot proceed because a required input is missing or a tool call was refused.
- REVIEW_REQUIRED: I am uncertain, scope has changed, or the contract seems wrong.

Format:
- Tool calls must be issued through the model's native tool calling if available; otherwise include a JSON object with a single key "tool_call" whose value is the structured tool payload.
- Do not produce unified diffs yourself.
- Do not propose patches via free text.
"""


# ----------------------------- Result -----------------------------

@dataclass
class RunResult:
    status: str  # PASSED / FAILED / BLOCKED / REVIEW_REQUIRED / RUNNING
    reason_code: str = ""
    reason_text: str = ""
    turns: int = 0
    tool_calls: int = 0
    artifacts_dir: str = ""
    proposal_count: int = 0
    applied_proposal_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# ----------------------------- Worker -----------------------------

class Worker:
    def __init__(
        self,
        client: OllamaClient | None = None,
        registry=None,
        runtime_roots: list[Path] | None = None,
        artifact_root: Path | None = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.registry = registry or default_registry()
        self.runtime_roots = runtime_roots or self._default_runtime_roots()
        self.artifact_root = artifact_root or (state_dir() / "runs")

    @staticmethod
    def _default_runtime_roots() -> list[Path]:
        here = Path(__file__).parent
        return [here, here.parent / "prompts"]

    # ---------- Main entrypoint ----------

    def run(self, manifest: TaskManifest, *, dry_run: bool = False) -> RunResult:
        started = time.time()
        repo_root = Path(manifest.repo.path).resolve()

        # Artifact dir
        run_id = f"run-{int(started)}-{uuid.uuid4().hex[:8]}"
        artifact_dir = self.artifact_root / manifest.task_id / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Persist manifest immediately
        (artifact_dir / "manifest.json").write_bytes(canonical_json(manifest))

        result = RunResult(status="RUNNING", artifacts_dir=str(artifact_dir))

        try:
            require_not_paused()

            # Preflight
            self._preflight(manifest, repo_root, artifact_dir)

            # editor_automation tasks are blocked at this version.
            if manifest.execution_class == ExecutionClass.UNREAL_EDITOR:
                self._write_json(artifact_dir / "result.json", {
                    "status": "BLOCKED",
                    "reason_code": "UNREAL_EDITOR_NOT_IMPLEMENTED",
                    "reason_text": "Unreal editor automation is not implemented in this sprint.",
                })
                result.status = "BLOCKED"
                result.reason_code = "UNREAL_EDITOR_NOT_IMPLEMENTED"
                result.reason_text = "Unreal editor automation deferred."
                return result

            broker = Broker(
                repo_root=repo_root,
                registry=self.registry,
                allowed_write_paths=list(manifest.paths.write_paths),
                allowed_create_paths=list(manifest.paths.create_paths),
                allowed_read_paths=list(manifest.paths.read_paths) + list(manifest.paths.protected_read_paths),
                max_tool_result_bytes=manifest.limits.max_tool_result_bytes,
            )

            # Tools the model may invoke. We map directly to Ollama tools schema.
            tools = _ollama_tool_schema()

            messages: list[dict[str, Any]] = [
                {"role": "user", "content": _format_user_brief(manifest)},
            ]

            # Conversation log
            transcript: list[dict[str, Any]] = []
            applied_proposals: list[str] = []
            proposals_made: list[str] = []
            total_calls = 0
            disposition: Disposition | None = None
            # Captured from report_result tool call.
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

                # Process tool calls
                if chat.tool_calls:
                    tool_messages = []
                    stop = False
                    for tc in chat.tool_calls:
                        if total_calls >= manifest.limits.max_tool_calls:
                            tool_messages.append(_tool_error_msg(tc, "max_tool_calls exceeded"))
                            continue
                        total_calls += 1
                        result.tool_calls = total_calls
                        try:
                            tc_args = tc.get("function", {}).get("arguments", {})
                            if isinstance(tc_args, str):
                                tc_args = json.loads(tc_args)
                            tool_name = tc.get("function", {}).get("name")
                            call = _parse_tool_call(tc.get("id") or f"c{total_calls}", tool_name, tc_args)
                            handled = self._dispatch(
                                broker, call, applied_proposals, proposals_made,
                                captured_disposition, captured_summary, captured_evidence,
                            )
                            tool_messages.append({
                                "role": "tool",
                                "tool_name": tool_name,
                                "content": json.dumps(handled, default=str)[: manifest.limits.max_tool_result_bytes],
                            })
                            # If report_result was called, terminate the loop.
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

                # No tool call -> check if final assistant message is disposition
                messages.append({"role": "assistant", "content": chat.content})
                disp, summary, evidence = _extract_disposition(chat.content)
                if disp is not None:
                    disposition = disp
                    _append_transcript_event(transcript, {
                        "event": "report_result",
                        "turn": turn + 1,
                        "disposition": disp.value,
                        "summary": summary,
                        "evidence": evidence,
                    })
                    break

                # Otherwise nudge model to call a tool or report result.
                messages.append({
                    "role": "user",
                    "content": "You have not called a tool or reported a disposition. Either call one of the offered tools now or call report_result.",
                })

            # Save transcript + proposals
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

            # Run validators
            final_status, reason_code, reason_text = _apply_validators(manifest, broker, disposition, artifact_dir)

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
            }, indent=2))

            return result
        except SafetyError as e:
            result.status = "BLOCKED"
            result.reason_code = "SAFETY"
            result.reason_text = str(e)
            (artifact_dir / "result.json").write_text(json.dumps({
                "status": result.status,
                "reason_code": result.reason_code,
                "reason_text": result.reason_text,
                "exception": traceback.format_exc(limit=10),
            }, indent=2))
            return result
        except Exception as e:
            result.status = "FAILED"
            result.reason_code = type(e).__name__
            result.reason_text = str(e)
            (artifact_dir / "result.json").write_text(json.dumps({
                "status": result.status,
                "reason_code": result.reason_code,
                "reason_text": result.reason_text,
                "exception": traceback.format_exc(limit=20),
            }, indent=2))
            return result

    # ---------- Helpers ----------

    def _preflight(self, manifest: TaskManifest, repo_root: Path, artifact_dir: Path) -> None:
        if not repo_root.exists():
            raise SafetyError(f"repo path does not exist: {repo_root}")
        head = git_head(repo_root)
        if not head:
            raise SafetyError("repo has no commits")
        worktree_sha = git_worktree_sha(repo_root)
        # source_mutation requires clean tree.
        if manifest.execution_class == ExecutionClass.SOURCE_MUTATION and manifest.repo.require_clean_tree:
            if not git_is_clean(repo_root):
                raise SafetyError("repository working tree is not clean")

        # Approval envelope binding.
        approval = {
            "manifest_sha256": canonical_sha(manifest),
            "approved_repo_head": head,
            "approved_worktree_sha256": worktree_sha,
            "approved_runtime_sha256": runtime_fingerprint(self.runtime_roots).sha256,
            "approved_model_name": manifest.model_profile.model_name,
            "approved_at": int(time.time()),
        }
        # Approval is stored in the artifacts directory, NOT in the repo, to
        # avoid dirtying the worktree. The model never sees the on-disk shape;
        # the harness enforces binding on every read.
        approval_path = artifact_dir / "approval.json"
        if approval_path.exists():
            prior = json.loads(approval_path.read_text())
            for key in ("manifest_sha256", "approved_repo_head", "approved_runtime_sha256"):
                if prior.get(key) != approval[key]:
                    raise SafetyError(f"approval mismatch on {key}")
        approval_path.write_text(json.dumps(approval, indent=2))

    def _dispatch(
        self,
        broker: Broker,
        call: ToolCall,
        applied: list[str],
        proposals: list[str],
        captured_disposition: list[Any] | None = None,
        captured_summary: list[str] | None = None,
        captured_evidence: list[list[str]] | None = None,
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
            # Capture the disposition for final-result logic.
            if captured_disposition is not None:
                captured_disposition[0] = a.disposition
            if captured_summary is not None:
                captured_summary[0] = a.summary
            if captured_evidence is not None:
                captured_evidence[0] = list(a.evidence)
            return {"recorded": True, "evidence": a.evidence}
        raise SafetyError(f"unsupported tool args: {type(a).__name__}")

    def _write_json(self, p: Path, obj: dict[str, Any]) -> None:
        p.write_text(json.dumps(obj, indent=2, default=str))


# ----------------------------- Validators -----------------------------

def _apply_validators(manifest: TaskManifest, broker: Broker, disposition: Disposition, artifact_dir: Path) -> tuple[str, str, str]:
    """Run deterministic validators. Return (status, reason_code, reason_text)."""
    # The model disposition cannot PASS the task.
    if disposition == Disposition.BLOCKED:
        return "BLOCKED", "MODEL_BLOCKED", "model reported BLOCKED"
    if disposition == Disposition.REVIEW_REQUIRED:
        return "REVIEW_REQUIRED", "MODEL_REVIEW_REQUIRED", "model reported REVIEW_REQUIRED"
    if disposition != Disposition.DONE:
        return "REVIEW_REQUIRED", "MODEL_DISPOSITION_UNKNOWN", f"unknown disposition: {disposition}"

    # Done -> run required validators.
    repo_root = Path(manifest.repo.path).resolve()
    failures: list[str] = []
    for vid in manifest.commands.required_validator_ids:
        # Built-in validators are deterministic in-Python checks; they may
        # also be overridden via the registry. Registry entries win.
        builtin_handled = False
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
                builtin_handled = True
            elif vid == "no_op":
                builtin_handled = True
        if builtin_handled:
            continue
        # Generic registry-backed validator.
        spec = broker.registry.get(vid)
        try:
            import subprocess
            proc = subprocess.run(
                spec.argv,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=spec.timeout_seconds,
                shell=False,
            )
            (artifact_dir / "commands" / vid).mkdir(parents=True, exist_ok=True)
            (artifact_dir / "commands" / vid / "stdout.txt").write_text(proc.stdout or "")
            (artifact_dir / "commands" / vid / "stderr.txt").write_text(proc.stderr or "")
            if proc.returncode != 0:
                failures.append(f"{vid} exit={proc.returncode}")
        except Exception as e:
            failures.append(f"{vid} error={type(e).__name__}: {e}")

    # For source_mutation, ensure declared write_paths were actually written.
    if manifest.execution_class == ExecutionClass.SOURCE_MUTATION:
        for wp in manifest.paths.write_paths:
            rp = (repo_root / wp).resolve()
            if not rp.exists():
                failures.append(f"write target not produced: {wp}")

    if failures:
        return "FAILED", "VALIDATORS_FAILED", "; ".join(failures)
    return "PASSED", "OK", "all validators passed"


# ----------------------------- Helpers -----------------------------

def _ollama_tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_exact",
                "description": "Read an authorised UTF-8 file. Returns path, sha256, size_bytes, content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 0},
                        "end_line": {"type": "integer", "minimum": 0},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_validated_patch",
                "description": "Apply a previously proposed patch by proposal_id. Validates everything atomically.",
                "parameters": {
                    "type": "object",
                    "properties": {"proposal_id": {"type": "string"}},
                    "required": ["proposal_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
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
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command_id",
                "description": "Run a registered command by id. The model never supplies argv.",
                "parameters": {
                    "type": "object",
                    "properties": {"command_id": {"type": "string"}},
                    "required": ["command_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "report_result",
                "description": "Report final disposition (DONE|BLOCKED|REVIEW_REQUIRED) and summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "disposition": {"type": "string", "enum": ["DONE", "BLOCKED", "REVIEW_REQUIRED"]},
                        "summary": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["disposition", "summary"],
                },
            },
        },
    ]


def _parse_tool_call(call_id: str, name: str, args: dict[str, Any]) -> ToolCall:
    from .schemas import (
        ApplyValidatedPatchArgs,
        CreateFileArgs,
        ReplaceExactArgs,
        ReplaceFileArgs,
        RunCommandArgs,
        ReportResultArgs,
    )
    if name == "read_exact":
        from .schemas import ReadExactArgs
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
    return {
        "role": "tool",
        "tool_name": tc.get("function", {}).get("name"),
        "content": json.dumps({"error": msg}),
    }


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
    """Find the LAST disposition line in the assistant content."""
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
        # Look for {"disposition": "..."} JSON fallback.
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
        f"commands allowed: {m.commands.model_allowed_command_ids}\n"
        f"required validators: {m.commands.required_validator_ids}\n"
        f"acceptance_criteria: {m.acceptance_criteria}\n"
        "Begin. Make the smallest change that satisfies the objective. "
        "When done, call report_result(disposition='DONE', summary=...)."
    )
