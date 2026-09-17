# overnight-runner — Sprint Report

Time-boxed implementation sprint. This document records what was built, what works, and what remains.

## Repository

- Local: `/mnt/storage/Repos/overnight-runner`
- GitHub: https://github.com/tgremlin/overnight-runner

## Commits (since baseline ff59e01)

```
6be5795 P0 hardening: independent approval binding, exact read allowlist,
        write/create separation, per-contract command allowlist, declared
        limits, owned process-group timeout, mutation evidence artifacts,
        source_mutation-must-mutate, nonmutating drift detection, PAUSED at
        apply boundary, system prompt tool list; 23 new tests
24d0645 Phase 2 runner: durable runs rows, heartbeat thread with own DB
        connection, lease expiry, recovery_scan (read-only/mutation/unreal
        all REVIEW_REQUIRED, mutation never auto-retries), dependency
        enforcement, artifact_dir path passing, 7 new tests
```

## Test counts

```
$ python3 -m pytest -q
..........................................                              [ 98%]
.                                                                        [100%]
73 passed in 2.96s
```

## P0 hardening — what was fixed

1. **Independent approval binding.** Approval is now passed to `Worker.run()` as a parameter; the worker NEVER generates its own envelope for queued runs. Bind checks reject BEFORE Ollama is called with precise reason codes: `APPROVAL_MANIFEST_CHANGED`, `APPROVAL_REPO_HEAD_CHANGED`, `APPROVAL_RUNTIME_CHANGED`, `APPROVAL_DIRTY_WORKTREE`. On mismatch, the task is transitioned back to `PENDING_APPROVAL` with an `approval_invalidated` event and the worker never invokes Ollama.
2. **Apply-time baseline recheck.** `Broker.apply_proposal()` re-validates the approved `repo_head` (via the `approved_repo_head` carried on the broker), the path policy, the file hash, the PAUSED sentinel, and the limits. A proposal that would mutate under a drifted HEAD raises `SafetyError`.
3. **Exact read allowlist.** Empty `read_paths` means NO files readable (not "anything"). Protected reads require explicit declaration in `protected_read_paths` (or `read_paths`). Regression-tested.
4. **Write vs create separation.** `replace_exact` / `replace_file` require membership in `write_paths`; `create_file` requires membership in `create_paths`. `create_paths` does NOT grant the right to overwrite an existing file. Regression-tested.
5. **Per-contract command allowlist.** The broker checks `model_allowed_command_ids` AND rejects required-validator IDs from the model. Registry membership alone is insufficient.
6. **Owned process-group timeout.** `Broker._spawn_own_pgrp()` opens the child with `start_new_session=True`, captures its pgid via `os.getpgid`, and on timeout sends `SIGTERM` to that pgid only, then `SIGKILL` after a grace period. No broad process killing.
7. **Declared limits enforced.** `max_changed_files`, `max_diff_lines`, `max_written_bytes`, `max_files_read`, `max_read_bytes`, `max_tool_result_bytes`, `max_model_turns`, `max_tool_calls`, `task_timeout_seconds` are all enforced; tracked via a `Usage` set.
8. **`source_mutation` requires mutation.** DONE with zero applied proposals (unless `commands.allow_no_mutation=True`) is now `REVIEW_REQUIRED` with reason `NO_MUTATION_APPLIED`.
9. **Non-mutating command drift detection.** Commands classified `none`/`read` capture `git_worktree_sha` before/after. Any drift raises `SafetyError`.
10. **Mutation evidence artifacts.** Each mutation writes `before/<path>`, `proposed/<path>`, `after/<path>`, `preview.diff`, `actual.diff`, and appends to `mutation-journal.jsonl`. No automatic rollback; no automatic commit.
11. **PAUSED at apply boundary.** The worker checks `is_paused()` before every tool dispatch; if set, the apply is refused and the task ends `BLOCKED`.
12. **System prompt tool list updated.** The prompt now lists `apply_validated_patch` explicitly so the model knows to call it.

## Real Gemma vertical slice — `PASSED`

Live `gemma4:12b` (`4eb23ef187e2…`) run via direct `/api/chat`:

### Phase 1 (`overnight-runner run`)

```
Fixture: /tmp/gemma-fixture/config.py  (NAME="hello")
Manifest: examples/gemma_oneword.json  (source_mutation, 1 file, 1 validator)

Result:
  status: PASSED
  reason_code: OK
  turns: 6
  tool_calls: 6
  proposals_made: 1
  applied_proposals: 1

After run:
  config.py -> NAME="howdy"
  git log: only the init commit (NO commit by runner)
  git status: M config.py (uncommitted working-tree change)
```

### Phase 2 queued (`run-next`)

```
overnight-runner import examples/gemma_queue.json
overnight-runner approve gemma-queue-001
overnight-runner run-next

Result:
  task_id: gemma-queue-001
  status: PASSED
  reason_code: OK
  artifact_dir: ~/.local/state/overnight-runner/runs/gemma-queue-001/run-1789675831-345322/

DB after run:
  tasks row: status=PASSED, approved_repo_head=<bound>, final_reason_code=OK
  runs row:  status=PASSED, model_name=gemma4:12b, mutation_started=1, artifact_dir=...
  events:    task_imported, task_approved, run_started, run_finished

After run:
  config.py -> NAME="howdy"
  git log: only the init commit (NO commit by runner)
```

## Phase 2 recovery pieces completed

- `Database.insert_run`, `heartbeat`, `finish_run`, `mark_mutation_started`, `find_stale_runs`.
- `runner.execute_queued_task()` runs one queued task with:
  - durable `runs` row at start;
  - dependency enforcement before claiming (raises if any dep not `PASSED`);
  - heartbeat thread with its **own** Database connection (no sharing);
  - 30-second heartbeat; 120-second lease window;
  - `on_mutation` hook to set `mutation_started=1` on the runs row;
  - finished status + `post_worktree_sha256` at exit.
- `runner.recovery_scan()` finds stale `RUNNING` runs and applies the spec policy:
  - `source_mutation` → `REVIEW_REQUIRED` with `STALE_MUTATION_NEVER_RETRY` (never auto-retried);
  - `read_only` → `REVIEW_REQUIRED` with `STALE_READ_ONLY` (caller may retry with policy);
  - `unreal_editor` → `REVIEW_REQUIRED` with `STALE_UNREAL` (no editor launch);
  - emits `task_stale` event with reason.

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
| Deterministic PASS | PASS |
| Artifacts (manifest/approval/transcript/proposal/result/commands) | PASS |
| Mutation evidence (before/proposed/after/preview.diff/actual.diff/journal) | PASS |
| Approval independently recorded | PASS |
| Approval checked at execution | PASS |
| HEAD drift invalidates approval | PASS |
| Runtime drift invalidates approval | PASS |
| Apply-time baseline recheck | PASS |
| Exact read allowlist | PASS |
| Write/create separation | PASS |
| Per-contract command allowlist | PASS |
| Required validators reserved | PASS |
| Owned process-group timeout | PASS |
| Declared limits enforced | PASS |
| Mutation-required-for-mutation-task | PASS |
| Nonmutating command drift detection | PASS |
| SQLite run rows | PASS |
| Leases | PASS |
| Heartbeats | PASS |
| Stale recovery | PASS |
| Dependency enforcement | PASS (raise in `execute_queued_task`) |
| systemd service | PASS (template) |
| systemd timer | PASS (template) |
| Tests | PASS (73) |
| GitHub remote | PASS |

## Incomplete / deferred

- The `dependency enforcement` check in `execute_queued_task` raises `RuntimeError`; the CLI's `run-next` prints the error and returns non-zero but does not yet transition the task to a "waiting" state. A future change should mark the task `WAITING` (or leave it `APPROVED`) and emit an event.
- Nightly `max_runtime`/`max_tasks`/`max_mutation_tasks`/`retries` are not yet enforced by a single global guard. Each task is bounded individually (`Limits.task_timeout_seconds`). A scheduler-level guard belongs in a future change.
- `git_diff_check` was registered but is not used by the model in the verified Gemma vertical slice; it is only asserted by tests.
- The Ollama client does not yet compute `post_worktree_sha256` writes from the model — that is recorded by `execute_queued_task`.

## Assumptions

1. `pydantic>=2.6,<3` and `pytest>=8` installed via `pip --break-system-packages --user` (no system pip on the machine); pinned in `pyproject.toml`.
2. Built-in validators `python_compile` and `noop` are short-circuited by the worker; only registry commands actually spawn subprocesses.
3. The `OVERNIGHT_STATE_DIR` env var is read at call time so `monkeypatch` works in tests; default is `~/.local/state/overnight-runner`.
4. The single-runner lock uses `flock` on a state-dir file; systemd-timer-launched runs and ad-hoc manual runs are mutually exclusive.
5. `seed` in the model profile is optional; default `None`.
