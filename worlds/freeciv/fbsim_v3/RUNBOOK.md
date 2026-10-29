# FreeCiv v3 runbook: one eval run, end to end

A run is done with nine numbered scripts in `worlds/freeciv/fbsim_v3/fleet/`, one per step. Code paths are relative
to `worlds/freeciv/fbsim_v3/` unless they start with `data/` or `credentials/`, the repository-root symlinks to the
local data and credentials folders (`scripts/setup_worktree.sh`).

## 1. Runs

A **run** is one configuration: player count, barbarians, fork (snapshot) turn, horizons, rng range, anchor seeds,
natural conditionals, and the pod machines and network volume it uses. Each run has one folder, which holds everything
about it:

```
data/freeciv/fbsim_v3/
  volume.txt                     default network volume for new runs (ID, data center)
  runs/<run>/                    also reachable as fleet/runs/<run>/
    run.env                      the configuration and creation commit (1_new_run.sh writes it; shipped to every pod)
    shell.sh                     source it for manual commands: exports FBSIM_RUN, sets R, S, K, V3, REPO
    RUN.md                       optional hand-written notes (keep rule, costs, problems)
    fleet_state/                 pods.tsv ledger, archive_pods.txt, queue/, anchors_raw/, anchors/, ANCHORS.txt,
                                 anchor_qc.txt, bundle/, bringup and launch logs
    raw/                         pulled from /vol/fbsim_v3/<run>/ on the archive volume
    archive_counts.txt  corpus_qc.txt  consol_v4/  consol_v4_SUMMARY.json
    bank_v1/
    sets/draw_v1/                question sets with criteria and continuous_norm_constants.json
    reports/seed<N>/turn_<FFF>_report.txt
    run/models_v2.csv            model panel (+ models_v2.source.txt)
    evals/<label>/               results.jsonl, calls.jsonl, spend_session.json, PREFLIGHT_v2.md (+ natcond/)
    scores/<label>/              score_items.csv, score_summary.csv, SCORES.md
```

Within a run, pods are named `fbsim3-<run>-<name>`, workers archive to `/vol/fbsim_v3/<run>/<pod name>/`, and every
script reads the fork turn, horizons, player count, barbarians and natcond flag from `run.env`. Runs never see each
other's pods, queues or archive folders.

Existing runs, all with 1 player and the fork at turn 60:
- `p1_t60_pilot` (October 27, 2026: anchors 9101–9104 × 100 replays). It predates run folders, so its archive folders
  are `/vol/fbsim_v3/fbsim3-{anchors,w01}/`, not under `/vol/fbsim_v3/p1_t60_pilot/`.
- `p1_pilot2` (October 28, 2026: anchors 9201–9203 × 10 replays), a smoke test of steps 1–6.
- `p1_t60` (October 29–30, 2026: anchors 9301–9308 × 1,000 replays), steps 1–9; its `RUN.md` has the details.

## 2. Keys

| what | where |
|---|---|
| `RUNPOD_API_KEY`, `OPENROUTER_API_KEY` | `credentials/.env` (or exported) |
| pod ssh key | `credentials/freeciv/runpod_key{,.pub}`; override with `FBSIM_SSH_KEY` |
| archive ssh key | `credentials/freeciv/archive_key{,.pub}` |

## 3. Steps

Run every step from `fleet/`:

```bash
cd worlds/freeciv/fbsim_v3/fleet          # from the repository root
```

Every script takes the run name as its first argument and selects the run itself, so nothing needs to be exported. It
prints what it is doing with timestamps. The long steps (2, 4, 5) wait and report progress. All steps can be stopped
with Ctrl-C and rerun: they reuse the pods, bringups, queue and finished work they find. Before deleting a pod or
sending paid API calls, a script asks y/N.

### Step 1: create the run

```bash
./1_new_run.sh p2_t60 --players 2 --barbarians DISABLED --seeds 9201-9240
./1_new_run.sh p5_t120 --fork-turn 120 --seeds 9301-9330          # horizons default to 150,180,210,240,270
```

| option | default |
|---|---|
| `--seeds LO-HI` | required: the anchor seeds this run may use; must not overlap another run's |
| `--players`, `--barbarians` | 5, server default |
| `--fork-turn`, `--horizons` | 60, fork + 30, +60 … +150 |
| `--replays N` or `--rng LO-HI` | 1,000 replays per anchor (rng 5001-6000); `--replays N` means rng 5001-(5000+N). Later steps take the count from the corpus; the bank and natural-conditional thresholds are a tenth of it |
| `--natcond` | 1 with two or more players, else 0 |
| `--flavor`, `--vcpu`, `--cloud` | `cpu3c`, 32, `SECURE` (anchor and worker pods) |
| `--arch-flavor`, `--arch-vcpu` | `cpu3c`, 8 (archive pod) |
| `--volume`, `--volume-dc` | from `data/freeciv/fbsim_v3/volume.txt` |
| `--force` | create the run even if the checkout is dirty |

It refuses a checkout with uncommitted or untracked changes unless you pass `--force`. Pods get the working tree, and
`run.env` records the commit, so a clean tree makes that record true. With `--force`, `run.env` notes the changes.
Leave the tree unchanged until the last pod is brought up (step 4).

Writes `runs/<run>/run.env`, `shell.sh` and an empty `fleet_state/`.

### Step 2: play the anchors

```bash
./2_anchors.sh p2_t60                 # every seed in FBSIM_SEEDS; or name some: ./2_anchors.sh p2_t60 9201 9202
```

1. Creates (or reuses) the pod `fbsim3-<run>-anchors` and brings it up. The bundle is built from your working tree;
   `run.env` goes to `/workspace/fleet/run.env`.
2. Plays the anchors, as many at once as the pod has civservers (the bringup log reports the count). Each game runs to
   fork turn + 1. Prints `done / running / failed / waiting` every minute and starts the next seeds as slots free up.
3. When none is left, pulls `/workspace/out/anchors/` to `fleet_state/anchors_raw/`. Each `seed<N>/` there has the data
   file, sgtables, every save, `settings.txt`, the ruleset, recording states and gate reports.
4. Runs the anchor QC (per-civ table at the fork turn, health flags) and saves it in `fleet_state/anchor_qc.txt`.

Failed anchors are listed with where their logs are. The anchor pod stays up, so you can play more seeds by rerunning
with them. Then choose which anchors to keep, and write the rule into `RUN.md` before looking at any replays.

### Step 3: keep anchors

```bash
./3_keep_anchors.sh p2_t60 9201 9202 9204 …
```

Checks each seed: a pulled data file, a fork-turn save, and a `settings.txt` that matches `run.env`. It then puts the
fork-turn saves in `fleet_state/anchors/`, which every later step uses as the anchor list, and removes seeds not named.
The choice is logged in `fleet_state/ANCHORS.txt`. Finally it asks to delete the anchor pod; step 4 refuses to start
while that pod is live. Refused once the replay queue exists.

### Step 4: start the fleet

```bash
./4_start_fleet.sh p2_t60 25 [--deadman 48] [--max-retries 5]
```

1. Archive pod `fbsim3-<run>-arch` on the run's network volume. Prepares it (archive key, `/vol/fbsim_v3/<run>/`) and
   copies `anchors_raw/` there, where `finalize.sh` later finds the ruleset.
2. Workers `fbsim3-<run>-w01` … `wNN`. Each deletes itself after `--deadman` hours whatever its state (default 48), so
   choose that above the expected run time.
3. Deals the queue: kept anchors × `FBSIM_RNG`, in proportion to vCPUs, into `fleet_state/queue/<pod>/todo/`.
4. Brings the workers up in parallel (logs: `fleet_state/bringup_<pod>.log`).
5. Launches the lanes (3/4 of the vCPUs per worker). Each worker syncs `/workspace/out/` to its archive folder every
   5 minutes. Prints the fleet status at the end. A launch that ends without lanes prints an `ERROR:` line, keeps its
   log as `fleet_state/failed_launch_<pod>_<time>.log` and is retried, up to `--max-retries` times per worker
   (default 5); past that the script stops. Raise `--max-retries` to retry again, since the earlier failures still count.

### Step 5: wait, finish, delete

```bash
./5_finish.sh p2_t60 [--interval 300]
```

1. Prints the fleet status every interval until every worker's queue is empty.
2. Lists the tasks that failed twice. If there are any, it asks whether to rerun them on the drained workers first
   (90-minute timeout each), then waits for the reruns and lists what still failed. A missing task means that anchor
   ends with fewer replays.
3. Finishes each worker (exact savegame tables, final sync) and deletes it once the archive holds all its forks and
   tables.

### Step 6: pull the corpus

```bash
./6_pull_corpus.sh p2_t60
```

Runs `finalize.sh`:
1. per-anchor counts on the archive, into `archive_counts.txt`;
2. pulls `/vol/fbsim_v3/<run>/` into `raw/`: anchors, queues, logs, data files, tables, `TRUNCATED` markers, and fork
   saves at the fork turn, fork + 1, each horizon and end + 1;
3. corpus QC into `corpus_qc.txt`;
4. consolidation into `consol_v4/` and `consol_v4_SUMMARY.json`.

Then it prints replays per anchor and asks to delete the archive pod. The volume keeps its copy.

### Step 7: question sets and reports

```bash
./7_make_sets.sh p2_t60 --models-csv PATH/TO/models_v2.csv
```

Runs `bank_v1.py`, `draw_v1.py`, `criteria_v1.py`, `norm_constants_v1.py` and `regen_reports.py` on the run, and copies
the model panel to `run/models_v2.csv`, recording where it came from in `run/models_v2.source.txt`. Refuses to redraw
once `evals/` has results. Read every report in `reports/` before eliciting.

Two draw rules differ from the paper's for a run, so that the mid-range bank spreads evenly over q. The bank's cells
(q band × horizon) take one question each in turn instead of filling one after another. With one player, a question
subject may recur once per q band, because each game has only one civilization to ask about. On `p1_t60` this took
the bank from 276 questions, mostly at low q, to 676.

### Step 8: elicit

```bash
./8_elicit.sh p2_t60 main [--models "Name A,Name B"] [other elicit_v2.py options]
```

1. A dry run prints the call counts and writes the example prompts to `evals/<label>/PREFLIGHT_v2.md`.
2. Asks before sending anything to OpenRouter.
3. Then elicits the bank, tails, mirrors and continuous sets into `evals/<label>/`. With natcond, it also elicits the
   unbatched natural conditionals into `evals/<label>/natcond/`.

`--models` defaults to every model in the panel. Rerunning with the same label resumes: answered calls aren't sent again.

### Step 9: score

```bash
./9_score.sh p2_t60 main
```

Scores every `results.jsonl` under `evals/<label>/` into `scores/<label>/` and prints the headline tables. An answer
that could not be parsed is left out of the scores (the paper scored a binary one as 0.5); the parse column gives the
share answered. Before scoring, check `evals/<label>/calls.jsonl` for calls that failed outright, such as a dropped
stream with no text: rerunning step 8 with the same label resends only those.

## 4. Manual commands

For anything the steps don't cover, source the run's shell file from `fleet/`. It exports `FBSIM_RUN` and sets `R`
(the run folder), `S` (`$R/fleet_state`), `K` (ssh keys) and `V3`:

```bash
source runs/p2_t60/shell.sh
./status.sh                 # one line per live pod and totals
./rebalance.sh              # even out uneven queues
./requeue_slice.sh POD      # deal a dead worker's remaining tasks to the others
./switch_anchor.sh OLD NEW  # replace an anchor mid-run
./rp.sh list | ./rp.sh ssh POD [CMD] | ./rp.sh delete POD
```

The Python scripts accept `--run <run>` (or use `FBSIM_RUN`) and default every folder to the run's. Without a run they
keep the paper's settings and rules (5 players, turn 60, horizons 90..210, the paper's bank draw, unparsed binary
answers scored as 0.5) and take explicit folders. Their outputs then match the code before run folders byte for byte
(checked October 28 and 30, 2026).

## 5. Before the first T120 run

- **Not yet run on real T120 games.** The T120 code path has only been checked on the pilot corpus shifted
  by 60 turns: sets, criteria and prompts match the T60 ones with every turn moved by 60. Run a small smoke first, for
  example a run with 1 anchor and `--rng 5001-5020`: anchors to 121, forks 120 → 270, saves at 120/121/150/…/270/271.
- **Barbarian line.** For default barbarians at a fork after turn 60, the reports no longer say "none exist", because
  raids begin around turn 60. Read the T120 reports' header closely.
- **Time per replay.** Measure it in that smoke. Each replay still plays 150 turns, but later turns are likely slower.
- **Step scripts on RunPod.** All nine steps ran for the T60 one-player run `p1_t60` (8 anchors × 1,000 replays,
  16 workers, October 29–30, 2026), after a smoke test of steps 1–6 with `p1_pilot2`. Step 7 needs enough replays per
  anchor: with 10, no question can reach the tails (q ≤ 5%) or mirrors sets, and `draw_v1.py` fails.
