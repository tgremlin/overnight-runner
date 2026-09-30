# Repository map — `overnight-runner`

## What this repo is

The overnight Runner: the Python pipeline that admits a campaign, holds the grant and
the protected approval, runs the trusted validators itself, owns the evidence digest and
writes the durable state. Bounded local-model work; no autonomous agents.

## Canonical location

| Field | Value |
|---|---|
| Canonical path | `/mnt/storage/Repos/overnight-runner` (branch `main`) |
| **Authoritative remote** | **`github`** = `https://github.com/tgremlin/overnight-runner.git` |
| Aliases kept | `github-review` and `origin` — all three are the SAME GitHub URL |
| Frozen working clone | `~/work/overnight-runner-ae` on `feat/ov01l-admission-heartbeat` @ `dc9654cd1ba04c10a1dd479af5b113573fbcf781` — the **trusted, frozen** line; its tree hash `1ebf227b35388ce864fce0f53d235f01a62bc694` must not change |
| M5 candidate clone | `~/work/overnight-runner-m5-proposal` on `m5-proposal/ov2` — the **ACTIVATION CANDIDATE**; its branch is pushed, its working clone is left in place for the operator |

## Branch policy

- **`main` is now the frozen line** (`dc9654c`, fast-forwarded from `393e2f8` on
  2026-09-30 by `git merge --ff-only` after proving `393e2f8` was an ancestor).
- **`m5-proposal/ov2` is an ACTIVATION CANDIDATE and is NOT merged into `main`.** It is
  a pushed branch only. Merging it is an explicit operator command, never part of a
  cleanup campaign.
- Agent work happens in clones/worktrees under `~/work`; `main` moves by fast-forward.
- **"Published" means pushed to GitHub; "merged" means on `main`.** Reports must say
  which.

## What dirties a tree

| Source | Paths | Handling |
|---|---|---|
| Python bytecode | `__pycache__/`, `*.pyc` | ignored; run tests with `PYTHONDONTWRITEBYTECODE=1` |
| pytest | `.pytest_cache/` | ignored |
| Runner state | `~/.trio/runner-state` (real state dir) | **never** the working tree; tools must point `OVERNIGHT_STATE_DIR` at a temp dir — an unset or REAL value is a typed `STATE_DIR_UNSET_OR_REAL` refusal |

## Tests

```sh
cd <clone> && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest
```

Green at the frozen line: **361 passed, 34 subtests passed, 0 failed**. Do NOT add `-q`
(the repo's `addopts` already sets it; a second `-q` suppresses the summary line).

## Hygiene

The cross-repo read-only check lives in the Forge repo:
`/mnt/ue/Projects/trio-game-forge/scripts/repo-hygiene.sh`.
