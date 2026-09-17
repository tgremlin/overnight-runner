# overnight-runner — Sprint Report

Time-boxed ~30-minute implementation sprint. This document records what was built, what works, and what remains.

## Repository

- Local: `/mnt/storage/Repos/overnight-runner`
- GitHub: not created (this sprint prioritized working code over fighting auth). See "Assumptions / next steps" below.

## What is fully working

- **Phase 1 manual sequential runner**: `overnight-runner run <manifest.json> --repo <path>`
- **Strict Pydantic 2.13 manifest validation**: rejects unknown fields (`extra="forbid"`), wildcards, path traversal, unknown ops, etc. Deterministic canonical JSON + SHA-256 binding.
- **Tool broker**: `read_exact`, `propose_patch` (replace_exact/replace_file/create_file), `apply_validated_patch`, `run_command_id`, `report_result`. No arbitrary shell. Model never supplies argv.
- **Path sandbox**: repo containment, symlink-escape detection, `.git` always blocked, write/create/read authorisation, protected paths.
- **Atomic safe writes**: temp file in same directory, fsync, mode preservation, `os.replace`. Mutation journal per file (in artifacts).
- **Approval envelope**: binds manifest SHA, repo HEAD, runtime fingerprint. Stored in artifacts, not in repo (to avoid dirtying worktree).
- **Single-runner lock** via `flock` on `~/.local/state/overnight-runner/runner.lock`.
- **PAUSED sentinel** at `~/.local/state/overnight-runner/PAUSED`.
- **Runtime fingerprint** of schemas/broker/worker/ollama_client/config/prompts.
- **Ollama client**: direct `/api/chat`, `think=False`, bounded `num_ctx`/`num_predict`/`temperature`/`seed`, non-streaming, captures Ollama timing/token metrics.
- **Worker loop**: bounded turns, bounded tool calls, deadline enforcement, transcript JSONL.
- **Deterministic PASS only via required validators** (`python_compile`, `no_op`, plus any registered command). Model disposition `DONE` triggers validators; `BLOCKED`/`REVIEW_REQUIRED` short-circuit.
- **Unreal editor automation**: schema support added; tasks are forced `BLOCKED` with reason `UNREAL_EDITOR_NOT_IMPLEMENTED`.
- **Phase 2 SQLite**: WAL + `busy_timeout` + `synchronous=FULL`, `tasks`/`runs`/`events`, short `BEGIN IMMEDIATE` transactions, `import` / `approve` / `status` / `summary` / `run-next` CLI.
- **systemd templates**: service + timer with `Persistent=false`, linger notes.

## Real Ollama round-trip

Tested live with **gemma4:12b** at `http://127.0.0.1:11434` using `examples/source_mutation.json` against `/tmp/example-repo`. The model:

1. `read_exact hello.py`
2. `read_exact test_hello.py`
3. `propose_patch op=replace_exact path=hello.py` (twice — corrected sha on retry)
4. `apply_validated_patch proposal_id=…`
5. `propose_patch op=replace_exact path=test_hello.py`
6. `apply_validated_patch proposal_id=…`
7. `run_command_id python_compile` (model attempted, got refused — `python_compile` is a built-in deterministic validator, not a registered command)
8. Reported `REVIEW_REQUIRED` after seeing the refused command.

Result: both files were modified as expected, no git commit was made, full transcript + proposal + approval + result persisted under `~/.local/state/overnight-runner/runs/...`. Final status `REVIEW_REQUIRED` (correctly — because the model declined; the runner did NOT trust the model to mark itself PASSED).

## Safety cases tested (44 unit/integration tests)

- Wildcard path rejected (`*.py`)
- Path traversal rejected (`../escape.py`)
- Absolute path rejected
- Symlink escape detected
- `.git` writes denied
- Stale `expected_sha256` rejected
- Duplicate `old_text` rejected when `expected_occurrences=1`
- `replace_file` on missing file rejected
- `create_file` on existing file rejected
- Unknown `command_id` rejected
- `read_exact` on unauthorised path rejected
- Protected read without declaration rejected
- Dirty working-tree refuses `source_mutation` via worker preflight
- No git commit made by broker (HEAD unchanged before/after)
- Proposal is single-use (cannot apply twice)
- Apply re-checks file hash; external mutation between propose and apply is rejected
- `Unreal editor` task forced BLOCKED with explicit reason code

## Test counts

```
$ python3 -m pytest -q
...........................................   [100%]
43 passed
```

## Capability matrix

| Feature | Status |
|---|---|
| Manifest validation | PASS |
| Canonical SHA | PASS |
| Repo baseline checks | PASS |
| Path sandbox | PASS |
| Protected paths | PASS |
| Runner lock | PASS |
| PAUSED | PASS |
| `read_exact` | PASS |
| `propose_patch` | PASS |
| `apply_validated_patch` | PASS |
| `run_command_id` | PASS |
| `report_result` | PASS |
| Ollama client | PASS |
| Executor loop | PASS |
| Deterministic PASS (validator-driven) | PASS |
| Artifacts (manifest/approval/transcript/proposal/result/commands) | PASS |
| SQLite | PASS (Phase 2 partial) |
| Leases | NOT STARTED (Phase 2 schema present, lease logic deferred) |
| Recovery (crashed/stale) | NOT STARTED |
| systemd service | PASS (template) |
| systemd timer | PASS (template) |
| Tests | PASS (43) |
| GitHub remote | NOT STARTED (auth not exercised this sprint) |

## Assumptions made

1. **GitHub remote**: gh CLI auth is available (verified) but creating the repo + pushing in-sprint risked time. Local repo initialised; remote creation deferred to the next step (one command).
2. **Validator model**: built-in `python_compile` and `no_op` are short-circuited inside the worker; registry commands (e.g. `git_diff_check`, `pytest_runner_tests`) run via `subprocess.run` with `shell=False` and `start_new_session=True`.
3. **Approval storage**: written to artifacts dir, NOT the repo, to avoid dirtying the worktree (which would block future source_mutation runs).
4. **Apply model**: model invokes `apply_validated_patch(proposal_id=…)` explicitly; we do NOT auto-apply on `propose_patch`. This makes the apply step auditable in the transcript.
5. **No automatic retries**: source_mutation tasks are not auto-retried on crash. Per spec.
6. **Default model profile** uses `gemma4:12b`, `num_ctx=8192`, `num_predict=1024`, `temperature=0.0`.

## Highest-priority next task

Create the GitHub repository and push the initial commit. From `/mnt/storage/Repos/overnight-runner`:

```bash
gh repo create tgremlin/overnight-runner --public --source=. --remote=origin --push
```

Then continue Phase 2:
- lease + heartbeat + stale recovery
- morning `summary` that walks `runs/` artifacts
- `Retries: 1` policy wired in
- max-runtime + max-tasks + max-mutating-tasks guards
