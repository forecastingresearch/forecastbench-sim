Read the repository CLAUDE.md and this folder's README.md. Use uv run.

## Fleet state and keys

`fleet/paths.sh` (sourced by every fleet script) requires `FBSIM_RUN`, sources the run's `run.env` and puts everything
outside the checkout, through the repository's `data/` and `credentials/` symlinks (`scripts/setup_worktree.sh` creates
them in every worktree):

| path | contents |
|---|---|
| `credentials/.env` | `RUNPOD_API_KEY` (read by `rp.sh` when the variable is unset), `OPENROUTER_API_KEY` (the `run/` elicitation scripts) |
| `credentials/freeciv/runpod_key{,.pub}` | ssh key for every pod; `rp.sh create` passes the public key, override with `FBSIM_SSH_KEY` |
| `credentials/freeciv/archive_key{,.pub}` | ssh key the workers use to rsync to the archive pod |
| `data/freeciv/fbsim_v3/runs/<run>/run.env` | the run's settings (`fleet/1_new_run.sh` writes it; also shipped to every pod) |
| `data/freeciv/fbsim_v3/runs/<run>/fleet_state/` | `pods.tsv`, `archive_pods.txt`, `addr_cache.tsv`, `queue/`, `anchors/`, `anchors_raw/`, `bundle/`, logs |
| `data/freeciv/fbsim_v3/volume.txt` | the RunPod network volume shared by every run's archive pod |

Each run has its own pod ledger, and pods are named `fbsim3-<run>-...`, so fleets of different runs do not see each
other's pods. All worktrees share the run folders. The RunPod account (pod and spend limits) is still shared.

On a machine without the keys yet, from the repository root:

```bash
C=credentials/freeciv; mkdir -p "$C" && chmod 700 "$C"
ssh-keygen -t ed25519 -N '' -C fbsim3-runpod -f "$C/runpod_key"
ssh-keygen -t ed25519 -N '' -C fbsim3-archive -f "$C/archive_key"
```

Never copy keys or run state into the repository.
