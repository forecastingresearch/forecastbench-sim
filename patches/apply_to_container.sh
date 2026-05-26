#!/usr/bin/env bash
# Apply the Monte Carlo reseed patch inside the freeciv-web docker container.
#
# What this does (Freeciv-side, not civbench):
#   1. Patches server/savegame/savegame3.c::sg_load_random to consult
#      /tmp/freeciv_reseed and call fc_srand() if set.
#   2. Rebuilds the freeciv server binary inside the container.
#   3. Commits the modified container back to freeciv/freeciv-web.
#
# Idempotent: re-running first restores the .bak.preMC backup so the
# patch is reapplied cleanly.
set -euo pipefail

CONTAINER=${CONTAINER:-freeciv-web}
SRC=/docker/freeciv/freeciv/server/savegame/savegame3.c
BAK=${SRC}.bak.preMC

echo "[1/4] Ensuring backup exists and restoring clean source..."
docker exec "$CONTAINER" bash -c "
  if [ ! -f $BAK ]; then cp $SRC $BAK; fi
  cp $BAK $SRC
"

echo "[2/4] Inserting reseed hook into sg_load_random (only the first occurrence — inside the random.saved=TRUE branch)..."
# We embed a small python script in the container to do the textual insert
# safely. We target the FIRST occurrence of the magic line — that's the one
# inside sg_load_random's `random.saved == TRUE` branch.
docker exec "$CONTAINER" python3 - <<'PY'
src = "/docker/freeciv/freeciv/server/savegame/savegame3.c"
with open(src) as f:
    content = f.read()

marker = "    fc_rand_set_state(loading->rstate);\n"
hook = """    fc_rand_set_state(loading->rstate);

    /* CIVBENCH MC reseed hook: if /tmp/freeciv_reseed exists, treat its
     * contents as an unsigned integer and call fc_srand() to overwrite
     * the restored state. Consumed (deleted) on use. */
    {
      FILE *mc_fp = fopen("/tmp/freeciv_reseed", "r");
      if (mc_fp != NULL) {
        unsigned long mc_seed = 0;
        if (fscanf(mc_fp, "%lu", &mc_seed) == 1) {
          log_normal("CIVBENCH: reseeding RNG via fc_srand(%lu)", mc_seed);
          fc_srand((RANDOM_TYPE) mc_seed);
          loading->rstate = fc_rand_state();
        }
        fclose(mc_fp);
        (void) unlink("/tmp/freeciv_reseed");
      }
    }
"""
# Replace only the first occurrence (the one in sg_load_random's saved=TRUE branch).
i = content.find(marker)
if i < 0:
    raise SystemExit("ERROR: marker line not found")
content = content[:i] + hook + content[i + len(marker):]
with open(src, "w") as f:
    f.write(content)
print("Patched savegame3.c")
PY

echo "[3/4] Rebuilding freeciv server (this can take a couple of minutes)..."
docker exec "$CONTAINER" bash -c "
  make -j2 -C /docker/freeciv/freeciv/build/server 2>&1 | tail -40
"

echo "[4/4] Verifying patch landed in built binary by searching for log string..."
docker exec "$CONTAINER" bash -c "
  BIN=/docker/freeciv/freeciv/build/server/freeciv-server
  if strings \"\$BIN\" | grep -q 'CIVBENCH: reseeding RNG'; then
    echo \"OK: reseed log string present in \$BIN\"
  else
    echo \"WARNING: reseed log string not found in \$BIN.\"
    exit 1
  fi
"

echo "Done. Patch installed in container. (Did not commit image — see README in patches/.)"
