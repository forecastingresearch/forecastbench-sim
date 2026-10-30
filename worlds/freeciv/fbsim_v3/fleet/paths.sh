# paths.sh — sourced by the fleet scripts once HERE is set: the run (FBSIM_RUN), its configuration and state, and the
# keys (all outside the checkout, through the repository's data/ and credentials/ symlinks; see scripts/setup_worktree.sh).
REPO="$(cd "$HERE/../../../.." && pwd)"
RUNS="$REPO/data/freeciv/fbsim_v3/runs"            # one folder per run; 1_new_run.sh creates them
[ -n "${FBSIM_RUN:-}" ] || { echo "set FBSIM_RUN to a run in $RUNS (create one with fleet/1_new_run.sh)" >&2; exit 2; }
RUNDIR="$RUNS/$FBSIM_RUN"
[ -f "$RUNDIR/run.env" ] || { echo "no $RUNDIR/run.env (create the run with fleet/1_new_run.sh $FBSIM_RUN ...)" >&2; exit 2; }
set -a; . "$RUNDIR/run.env"; set +a                # FBSIM_N_PLAYERS, FBSIM_BARBARIANS, FBSIM_FORK_TURN, FBSIM_HORIZONS, FBSIM_RNG, ...
export FBSIM_RUN FBSIM_V3_CORPUS="$RUNDIR"         # the run folder is also its corpus (raw/, consol_v4/, bank_v1/, sets/, ...)
STATE="$RUNDIR/fleet_state"                        # pods.tsv ledger, queues, anchors, bringup/launch logs
VOL="/vol/fbsim_v3/$FBSIM_RUN"                     # this run's folder on the archive volume: $VOL/<pod name>/
POD_PREFIX="fbsim3-$FBSIM_RUN-"                    # rp.sh create prefixes every pod name with it
END_TURN="${FBSIM_HORIZONS##*,}"
HSAVE_TURNS="$FBSIM_FORK_TURN $((FBSIM_FORK_TURN + 1)) ${FBSIM_HORIZONS//,/ } $((END_TURN + 1))"   # fork saves pulled
CRED="$REPO/credentials"                          # .env (RUNPOD_API_KEY), freeciv/runpod_key, freeciv/archive_key
SSHKEY="${FBSIM_SSH_KEY:-$CRED/freeciv/runpod_key}"
ARCHKEY="$CRED/freeciv/archive_key"
SSHO=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR)
# podsh PORT root@IP [CMD...] — ssh to a pod; safe when the checkout path has spaces. For rsync -e, quote the key
# inside the string instead ("ssh -i '$SSHKEY' ..."): rsync splits that string honoring quotes.
podsh() { local port="$1"; shift; ssh -i "$SSHKEY" -p "$port" "${SSHO[@]}" "$@"; }
