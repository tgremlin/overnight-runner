# overnight-runner — Sprint Report

## PHASE 2 STATUS

**READY FOR CONTROLLED REAL-PROJECT TRIAL.**

Not production-proven. Not fully autonomous. The next step is empirical testing.

### Recommended trial sequence

1. **Trial 1** — Real repository, READ-ONLY contracts only.
2. **Trial 2** — Real repository, harmless test or documentation mutation.
3. **Trial 3** — Small mechanical source mutation.

Only after evidence from those should we change the architecture.

---

## Current state

- HEAD: see git log (pushed to `origin/main`).
- Tests: 110 passing (was 91 at the start of this hardening pass; +19).
- Real Gemma nightly smoke: **PASSED** (`run-nightly`, both tasks PASS, file changed, no commit).

## Hardening pass (this round)

P0:
1. **Model digest fail-closed.** `_bind_check()` requires resolver exists, succeeds, returns non-empty digest, AND matches. Failures return `APPROVAL_MODEL_UNRESOLVABLE` (no metadata) or `APPROVAL_MODEL_CHANGED` (digest mismatch) or `APPROVAL_MODEL_CHANGED` (model name mismatch). Ollama is never called on any failure (test asserts `chat` count == 0).
2. **Clean worktree includes untracked files.** `git_is_clean()` now uses `git status --porcelain --ignored` and ignores lines starting with `!!` (ignored). Non-ignored untracked files make the tree DIRTY; ignored files do not.
3. **Worktree drift sees untracked files.** `git_worktree_sha()` now hashes both tracked files (`git ls-files -z`) and non-ignored untracked files (`git ls-files --others --exclude-standard -z`). Drift detection catches untracked file creation, tracked file modification, and tracked file deletion.
4. **Monotonic clock for nightly deadlines.** `time.monotonic()` for `elapsed`, `remaining`, `shutdown-margin` checks. `time.time()` retained for audit/DB/summary timestamps.
5. **Task-timeout budget check.** Before claiming a candidate, the runner inspects the manifest's `limits.task_timeout_seconds` and skips it if it would not fit in the remaining nightly budget (with shutdown margin). An insufficient-budget task remains APPROVED for the next night.
6. **Single atomic-claim implementation.** The `runner._atomic_claim` duplicate has been deleted. There is exactly one implementation: `Database.claim_next_approved`. It runs in one `BEGIN IMMEDIATE` transaction that does eligibility + dependency + budget + attempt_no + insert RUNNING runs row + task APPROVED->RUNNING + `run_started` event.
7. **Transaction lost-race rollback.** `UPDATE ... WHERE status='APPROVED'` returning `rowcount != 1` now raises `ClaimConflict`. The transaction rolls back; caller may retry.

P1:
8. **Runtime fingerprint stability.** Excludes `__pycache__/`, `*.pyc`, `.git/`, `.tox/`, `.venv/`, `venv/`, `node_modules/`. Uses relative POSIX paths so fingerprint is independent of installation prefix. Bytecode regeneration does not change the fingerprint.
9. **Strict total-six-task.** `max_tasks=6` caps TOTAL attempts (read-only + mutation combined). Test: 6 RO + 1 mutation → only 6 run, mutation stays APPROVED. 5 RO + 1 mutation → 6 run (5 + 1).
10. **No automatic retries.** `DEFAULT_NIGHTLY['retries'] = 0`. Failed tasks are left for human review.
11. **Code dedup.** Removed legacy `execute_queued_task` shim, `_atomic_claim` duplicate, and `Broker.MutationJournalEntry` re-import in `worker.py`. `_execute_via_worker` retained as the single shared finalisation helper.

P1-2 documentation:
- `README.md` updated with Phase 2 status, operational modes (manual `run` vs durable `import -> approve -> run-nightly`), and explicit trial sequence.

## Capability matrix

| Feature | Status |
|---|---|
| Atomic task+run claim | PASS (single implementation in `Database`) |
| Runner lock before claim | PASS |
| Manual runner lock | PASS |
| Dependency-aware selection | PASS |
| Correct lease expiry | PASS (strict `< now`, no hidden grace) |
| Orphan recovery | PASS |
| Safe validator process groups | PASS |
| Model digest binding (fail-closed) | PASS |
| Untracked files count as dirty | PASS |
| Untracked files cause drift | PASS |
| Monotonic budget | PASS |
| Task-timeout budget pre-check | PASS |
| Total attempts <= max_tasks | PASS |
| Mutation-last rule | PASS |
| PAUSED nightly behavior | PASS |
| Morning summary | PASS |
| run-nightly CLI | PASS |
| systemd run-nightly | PASS |
| Real Gemma nightly smoke | PASS |
| Tests | PASS (110) |
| Runtime fingerprint stable | PASS |
| ClaimConflict on lost race | PASS |
| GitHub push | PASS |

## Incomplete / explicitly deferred

- Automatic read-only retry. Conservative policy: 0 retries. Reapproval/review required.
- Scheduler loop. `run-nightly` runs ONE session and exits; systemd calls it per timer fire.
- Unreal editor. Forced BLOCKED by policy.
- Phase 1 ad-hoc `run` still uses ephemeral approval (no model digest binding).

## End-of-sprint discipline

- No new product features added in this round.
- Tests +1 hardening count only (P0/P1 regressions).
- Single code path for atomic claim; less code overall (legacy shim removed).
- The sprint is FROZEN at Phase 2. The next change set must come from
  real-project trial evidence, not runner engineering.
