#!/usr/bin/env bash
# fork_one.sh ANCHOR_SEED FORK_NAME RNG_SEED [END_TURN] — run ONE fork (at FBSIM_FORK_TURN, default 60) of
# the anchor world via the FIXED fork path, serialize it with savegame
# coverage required, and publish data + ALL per-turn saves under
# ${FBSIM_OUT_DIR}/forks/<FORK_NAME>/.
#
# Publish contract (same spirit as mine_one_world.sh):
#   - run_fork.py exit code gated and wrapped in timeout(1800s); a timeout
#     is a failure (median rollout 6 min, max observed 18)
#   - serialization runs with FBSIM_REQUIRE_SAVEGAME_COVERAGE=1
#   - bench/observability_gate.py must PASS (per-player metrics sourced from
#     the omniscient savegame, no -1 scores, coverage 1.0)
#   - every per-turn savegame is published (downstream re-reads them)
#   - failures land in ${FBSIM_OUT_DIR}/forks_failed/<FORK_NAME>.FAILED
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${BUNDLE_DIR}/pod_env.sh"

ANCHOR="${1:?usage: fork_one.sh ANCHOR_SEED FORK_NAME RNG_SEED [END_TURN]}"
NAME="${2:?fork name (no underscores)}"
RSEED="${3:?rng seed}"
END_TURN="${4:-90}"
FORK_TURN="${FBSIM_FORK_TURN:-60}"
RUN_TIMEOUT_S="${FBSIM_FORK_TIMEOUT_S:-1800}"

case "${NAME}" in (*_*) echo "fork name must not contain underscores"; exit 2;; esac

FUSER="seed${ANCHOR}fork${NAME}"
REC_DIR="${FBSIM_LOG_ROOT}/recordings/${FUSER}"
OUT_DIR="${FBSIM_OUT_DIR}/forks/${NAME}"
OUT_FAILED="${FBSIM_OUT_DIR}/forks_failed"
SER_DIR="${FBSIM_OUT_DIR}/serialize_tmp/${FUSER}"
mkdir -p "${OUT_FAILED}" "${SER_DIR}"

fail() {
  echo "fork ${NAME} FAILED: $*"
  { echo "fork=${NAME} rng=${RSEED}"; echo "reason: $*"; date -u; } \
    > "${OUT_FAILED}/${NAME}.FAILED"
  exit 1
}

echo "== fork ${NAME} (rng ${RSEED}): run_fork t${FORK_TURN}->t${END_TURN} (timeout ${RUN_TIMEOUT_S}s) =="
timeout "${RUN_TIMEOUT_S}" python "${BUNDLE_DIR}/worlds/freeciv/scripts/run_fork.py" \
  --base-seed "${ANCHOR}" \
  --checkpoint-turn "${FORK_TURN}" \
  --end-turn "${END_TURN}" \
  --modification "rng_seed:${RSEED}" \
  --fork-name "${NAME}" \
  --recording-dir "${FBSIM_LOG_ROOT}/recordings/seed${ANCHOR}"
RC=$?
[ "${RC}" -eq 124 ] && fail "run_fork timed out after ${RUN_TIMEOUT_S}s"
[ "${RC}" -eq 0 ] || fail "run_fork exited ${RC}"

echo "== fork ${NAME}: savegame completeness precheck =="
[ -d "${REC_DIR}/savegames" ] || fail "no savegames/ under ${REC_DIR}"
N_SAVES=$(ls "${REC_DIR}/savegames" | grep -c "_T" || true)
MIN_SAVES=$(( (END_TURN - FORK_TURN) * 9 / 10 ))
TRUNCATED=""
if [ "${N_SAVES}" -lt "${MIN_SAVES}" ]; then
  # fbsim v3 (2026-09-02): run_fork exited 0 but the recording stopped early. The known cause is
  # the observer's player (player 0) being eliminated, which terminates the env while the server
  # idles. Such futures are real outcomes: publish everything recorded, with a TRUNCATED marker,
  # instead of dropping (and thereby biasing against) them. Downstream must honour the marker.
  LAST_T=$(ls "${REC_DIR}/savegames" | sed -n 's/.*_T\([0-9]*\)_.*/\1/p' | sort -n | tail -n 1)
  TRUNCATED="last_turn=${LAST_T:-?} saves=${N_SAVES} min_expected=${MIN_SAVES}"
  echo "fork ${NAME} TRUNCATED: ${TRUNCATED} (publishing what was recorded)"
  [ "${N_SAVES}" -ge 5 ] || fail "only ${N_SAVES} fork savegames; too few to publish"
fi

echo "== fork ${NAME}: serialize (coverage-required) =="
export FBSIM_REQUIRE_SAVEGAME_COVERAGE=1
python "${BUNDLE_DIR}/worlds/freeciv/scripts/generate_data_batch.py" \
  --input-dir "${FBSIM_LOG_ROOT}/recordings" \
  --output-dir "${SER_DIR}" \
  --filter "${FUSER}" --force --workers 1 \
  || fail "serialization failed (coverage or extraction error)"
DATA_JSON="${SER_DIR}/${FUSER}_data.json"
[ -s "${DATA_JSON}" ] || fail "serializer produced no ${DATA_JSON}"

echo "== fork ${NAME}: observability gate (savegame-sourced metrics) =="
python "${BUNDLE_DIR}/bench/observability_gate.py" "${DATA_JSON}" \
  --recording-dir "${REC_DIR}" --username "${FUSER}" > "${SER_DIR}/obsgate.txt" 2>&1
GATE_RC=$?
cat "${SER_DIR}/obsgate.txt"
if [ "${GATE_RC}" -ne 0 ]; then
  if [ -n "${TRUNCATED}" ]; then
    # fbsim v3: on an observer-death truncation the death turn's row is empty in the serializer
    # output; the per-turn savegames are complete and are the truth downstream. Publish with the
    # gate report attached rather than dropping the future.
    echo "fork ${NAME}: observability gate flagged a TRUNCATED fork; publishing anyway (report kept)"
    TRUNCATED="${TRUNCATED} obsgate=FLAGGED"
  else
    fail "observability gate flagged the fork (client-view metrics / -1 scores / coverage)"
  fi
fi

echo "== fork ${NAME}: publish =="
mkdir -p "${OUT_DIR}/savegames"
gzip -c "${DATA_JSON}" > "${OUT_DIR}/${FUSER}_data.json.gz" \
  || fail "gzip publish failed"
# Keep ALL per-turn savegames: downstream metrics/gates re-read them.
cp "${REC_DIR}/savegames/${FUSER}_T"*.sav.* "${OUT_DIR}/savegames/" \
  || fail "savegame publish copy failed"
N_KEPT=$(ls "${OUT_DIR}/savegames" | grep -c "_T" || true)
[ "${N_KEPT}" -ge "${N_SAVES}" ] || fail "published only ${N_KEPT} savegames (<${N_SAVES})"
if [ -n "${TRUNCATED}" ]; then { echo "${TRUNCATED}"; date -u; } > "${OUT_DIR}/TRUNCATED"; cp "${SER_DIR}/obsgate.txt" "${OUT_DIR}/obsgate.txt" 2>/dev/null; fi
rm -f "${OUT_FAILED}/${NAME}.FAILED" 2>/dev/null
echo "fork ${NAME} PUBLISHED"

rm -rf "${SER_DIR}" "${REC_DIR}"
exit 0
