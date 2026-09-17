# overnight-runner

A local, bounded work pipeline for tiny approved task contracts executed by small local Ollama models.

The local model is **not** an autonomous engineer. It is a bounded executor inside a deterministic Python harness.

Primary design principle: **make the model the least authoritative component in the system.**

## What this is

- A safety-first executor for small coding models (e.g. `gemma4:12b`).
- Deterministic Python owns planning, validation, writes, process control, state transitions.
- The model only proposes bounded edits via brokered tools.
- Phase 1 = manual sequential single-task runner.
- Phase 2 = SQLite queue + leases + recovery + systemd.
- Phase 3 = limited read-only concurrency (deferred).

## Install (dev)

```bash
cd overnight-runner
python3 -m pip install --user -e .
```

## CLI

```bash
overnight-runner doctor
overnight-runner validate path/to/manifest.json
overnight-runner run path/to/manifest.json --repo path/to/test-repo
overnight-runner import path/to/manifest.json        # Phase 2
overnight-runner approve <task-id>                    # Phase 2
overnight-runner run-next                             # Phase 2
overnight-runner status                               # Phase 2
overnight-runner summary                              # Phase 2
```

## Runtime layout

| Path | Purpose |
|---|---|
| `~/.local/lib/overnight-runner/` | Installed runtime |
| `~/.local/state/overnight-runner/state.db` | SQLite queue (Phase 2) |
| `~/.local/state/overnight-runner/PAUSED` | Sentinel pause switch |
| `~/.local/state/overnight-runner/runner.lock` | Single-runner lock |
| `~/.local/state/overnight-runner/runs/<task-id>/<run-id>/` | Artifacts |

## Status

This is a time-boxed sprint. See `docs/SPRINT_REPORT.md` for what works and what remains.

See `examples/` for example manifests.
