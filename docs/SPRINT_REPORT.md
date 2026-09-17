# overnight-runner — Sprint Report

## Repository

- Local: `/mnt/storage/Repos/overnight-runner`
- GitHub: https://github.com/tgremlin/overnight-runner

## Commits (this sprint)

```
b465f21 P1 overnight pipeline: atomic claim+run row in one transaction, ...
       global runner lock before claim, manual run uses lock,
       dependency-aware claim (skip unmet deps, no WAITING state),
       orphan RUNNING task recovery, strict lease expiry,
       model digest binding at approve + execution, validator safe
       process groups, run metrics callback, read-only-first nightly
       with mutation-last rule, 6-task/1-mutation/8h limits, PAUSED
       stops new tasks, morning summary JSON+MD, systemd targets
       run-nightly, install script, 18 new tests
```

## Test counts

```
$ python3 -m pytest -q
..........................................                              [ 79%]
...................                                                      [100%]
91 passed
```

## P0/P1 fixes completed (this sprint)

1. **Atomic claim** — `Database.claim_next_approved` + `runner._atomic_claim`: a single `BEGIN IMMEDIATE` transaction picks one eligible APPROVED task, increments `attempt_no`, inserts the RUNNING `runs` row, transitions task `APPROVED -> RUNNING`, and writes `run_id` on the task. Invariant: `tasks.status='RUNNING' <=> exists RUNNING runs row`.
2. **Global runner lock before claim** — `cmd_run_next` and `cmd_run_nightly` acquire `runner_lock()` first; manual `cmd_run` is also under the lock so a human invocation cannot overlap a systemd run.
3. **Dependency-aware selection** — `json_each(dependencies_json)` is checked in SQL inside the claim; tasks with unmet deps are skipped (left APPROVED) and not chosen; `get_dep_blocked_approved` reports them in the summary.
4. **Strict lease expiry** — `find_stale_runs(now)` returns rows where `lease_expires_at < now`. No hidden 600s grace.
5. **Orphan RUNNING task recovery** — `find_orphan_running_tasks` finds `tasks.status='RUNNING'` with no active RUNNING runs row; recovered to `REVIEW_REQUIRED` with `ORPHANED_RUNNING_TASK`.
6. **Safe validator process groups** — Required-validator execution now goes through `Broker._spawn_own_pgrp` (SIGTERM-owned-PGID, then SIGKILL after a grace); non-mutating validators also do a `git_worktree_sha` before/after drift check.
7. **Model digest binding** — `cmd_approve` resolves the model digest via `OllamaClient.model_digest` (which uses `/api/tags`); refuses approval if unresolved; stores `approved_model_digest`; `_bind_check` re-resolves at execution and returns `APPROVAL_MODEL_CHANGED` on mismatch.
8. **Run metrics** — Worker accepts an `on_metrics(snapshot)` callback and emits per-turn and per-tool-call deltas. The runner accumulates into a `metrics_state` dict and persists the final totals on `finish_run`, independent of heartbeat timing.
9. **`run-nightly` CLI** — sequential; read-only loop until none/mutation-cap/budget/PAUSED; then at most one source_mutation; stops immediately after mutation.
10. **Nightly limits** — 8 h wall, 6 tasks, 1 mutation; enforced inside `run_nightly`.
11. **PAUSED** — checked at start, between tasks, and at tool/apply boundaries; PAUSED is never auto-deleted.
12. **Morning summary** — `~/.local/state/overnight-runner/sessions/<id>/summary.json` and `summary.md`. Includes session id, wall duration, stop reason, counts, per-task result rows, dep-blocked APPROVED list, and require-review list. Deterministic (no AI).
13. **systemd service** — `ExecStart=/usr/bin/env python3 -m overnight_runner.cli run-nightly`. Timer unchanged. `scripts/install-user-service.sh` provided; does NOT enable linger.

## Real Gemma nightly smoke test

```
overnight-runner import examples/read_inspect.json
overnight-runner import examples/nightly_mutation.json
overnight-runner approve nightly-read-inspect
overnight-runner approve nightly-mutation-001
overnight-runner run-nightly
```

Result:

```
session_id: nightly-1789676647-70d589
stop_reason: MUTATION_DONE_STOPPING
tasks_attempted: 2
tasks_passed: 2
read_only_attempted: 1
mutation_attempted: 1
wall_duration_seconds: 21
summary_dir: ~/.local/state/overnight-runner/sessions/nightly-1789676647-70d589
```

After run:

- `/tmp/nightly-fixture/config.py`: `NAME="howdy"` (changed).
- `git log`: only the init commit (NO commit by runner).
- `git status`: `M config.py` (uncommitted).
- `runs` table: 2 rows, both PASSED, both with attempt=1.
- Events: `run_started`, `run_finished` for each.

## Capability matrix

| Feature | Status |
|---|---|
| Atomic task+run claim | PASS |
| Runner lock before claim | PASS |
| Manual runner lock | PASS |
| Dependency-aware selection | PASS |
| Correct lease expiry | PASS |
| Orphan recovery | PASS |
| Safe validator process groups | PASS |
| Model digest binding | PASS |
| Run metrics | PASS |
| Attempt numbering | PASS |
| Read-only retry | NOT STARTED (conservative: no auto-retry in MVP) |
| 8h nightly limit | PASS |
| 6-task limit | PASS |
| 1-mutation limit | PASS |
| Mutation-last rule | PASS |
| PAUSED nightly behavior | PASS |
| Morning summary | PASS |
| run-nightly CLI | PASS |
| systemd run-nightly | PASS (template + install script) |
| Real Gemma nightly smoke | PASS |
| Tests | PASS (91) |
| GitHub push | PASS |

## Incomplete / deferred

- **Read-only auto-retry** is not implemented. Conservative policy: never auto-retry (matches spec "Safety > utilization"). Adding a single retry only for transient read_only failures is a future change.
- **Unreal editor** is forced BLOCKED by policy and never executed. Deferred until an explicit owner-slot reopening integration exists.
- **Scheduler loop** is not implemented. `run-nightly` runs ONE session and exits. A loop driver (cron-style) belongs in a future change; for now `systemd` calls `run-nightly` once per timer fire.
- **Model digest binding** for Phase 1 ad-hoc `run` is not enforced (still ephemeral). Phase 2 queued uses the bound digest.
- **Atomic claim** relies on `BEGIN IMMEDIATE`. If a second runner claims the same APPROVED task at the same instant, only one transaction commits; the loser sees `cur.rowcount != 1` and returns None. No additional distributed lock needed.
