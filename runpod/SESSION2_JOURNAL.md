# Session 2 Run Journal

**Started**: 2026-06-13 PT  
**Goal**: run E1-E5 from `runpod/EXPERIMENT_PLAN.md`, with parallel pods where supply allows, while keeping implementation compromises small and documented.

## 2026-06-13 Preflight

- Read `runpod/README.md`, `runpod/SESSION_REPORT.md`, and `runpod/WRITEUP.md`.
- Confirmed `.env` has both required keys (`RUNPOD_API_KEY`, `HF_ACCESS_TOKEN`) without printing their values.
- Local disk has about 72 GiB free; enough for result rsync and adapter pulls.
- Local data present:
  - `data/training/train.jsonl`: 641 rows
  - `data/training/val.jsonl`: 57 rows
  - `data/forecastbench/eval_post2025.csv`: 401 examples plus header
- Found current repo already has uncommitted runpod script changes from Session 1; patched in place without reverting them.
- Script status before patching:
  - `04_eval.py` already had `--chat-template` and `llm.chat`.
  - `02_sft.py` still used raw `text` records and had no Brier-on-val callback.
  - `03_rl.py` still required `--adapter`, used raw prompts, and used `vllm_gpu_memory_utilization=0.35`.

## Implementation Notes

- Patched `02_sft.py` to train on `messages` records so TRL applies Qwen3's chat template.
- Added `BrierOnValCallback` to `02_sft.py`, writing `brier_curve.jsonl` once per epoch.
- Patched `03_rl.py` to allow RL from base when `--adapter` is omitted.
- Patched `03_rl.py` to wrap RL prompts with the chat template at dataset-build time.
- Patched `03_rl.py` vLLM colocate memory utilization from `0.35` to `0.25`.

## E4 Split

- Created `data/training_E4/train.jsonl` and `data/training_E4/val.jsonl`.
- Held out `wonder_completed`, `territory_comparative`, and `tech_comparative`.
- Resulting split: 506 train rows across 7 templates, 192 held-out rows across 3 templates.

## Provisioning

- GraphQL calls returned 403 with the local RunPod key, but REST API calls succeeded. Compromise: use the current RunPod REST API for pod lifecycle operations instead of the GraphQL snippets in the plan. This is operational only and does not affect the experiment measurements.
- Provisioned parallel layout:
  - E1: `ypcieyl1am1zgo`, secure `NVIDIA A100-SXM4-80GB`, `$1.49/hr`
  - E2/E3: `eo4f0kxunsu3xa`, secure `NVIDIA H200`, `$4.39/hr`
  - E4: `pn0g5gm98bk13n`, secure `NVIDIA H200`, `$4.39/hr`
  - E5: `uacz82qsf7wdqd`, secure `NVIDIA H200`, `$4.39/hr`
- Parallel burn rate while all four are running: about `$14.66/hr`.

## Setup Debugging

- Initial setup failed on all four pods after installing `vllm==0.11.2` because the plan's dependency pin `transformers>=4.51,<4.55` conflicts with `trl==0.21.0`, which now requires `transformers>=4.55`.
- Compromise: relax the setup pin to `transformers>=4.55,<4.58`. This preserves Qwen3 support and TRL 0.21 behavior while satisfying the current resolver; it should not change the experimental comparison except through a modern compatible tokenizer/model implementation.
- vLLM's current dependency set installed `torch==2.9.0+cu128`, so the planned torch 2.8 FlashAttention wheel no longer matched. Compromise: skip FA2 rather than risk a source compile; scripts fall back to SDPA. This affects wall time, not the target/reward/eval definitions.
- REST-created pods did not expose `HF_TOKEN` into SSH sessions. Compromise: pass the token from local `.env` over the SSH stdin stream for the finalize step without printing it.
- First E2/E4 launch failed immediately because current `trl==0.21.0` has `SFTConfig(max_length=...)` instead of the plan/script's `max_seq_length`. Patched `02_sft.py` to use `max_length`; this is an API compatibility fix, preserving the intended 8192-token cap.

## Launch Status at 2026-06-13 22:09 UTC

- Rsynced patched repo, data, E4 split, and run scripts to all four pods.
- E2/E3:
  - First launch failed at `SFTConfig(max_seq_length=...)`.
  - Relaunched clean after patching `max_length`.
  - Confirmed stepping in E2 SFT: step 5 by 22:09 UTC, loss around 0.33-0.35.
- E4:
  - First launch failed at the same `SFTConfig` incompatibility.
  - Relaunched clean after patching `max_length`.
  - Confirmed stepping in template-holdout SFT: step 7 by 22:09 UTC, loss around 0.31-0.38.
- E5:
  - Confirmed vLLM initialized and GRPO is stepping.
  - Early speed is about 50-52 sec/step, slower than the original 3 h wall estimate for 300 steps. This likely pushes E5 closer to 4.2-4.5 h before post-RL eval.
- E1:
  - Initial `hf_transfer` download on A100 stalled at a 12 GiB cache with an unchanged incomplete shard.
  - Killed the stuck downloader, removed stale locks/incomplete files, and restarted standard HF download without `hf_transfer`.
- Cost estimate at this point: about 16-17 minutes of four-pod parallel runtime at `$14.66/hr`, roughly `$4.0` burn so far.

## Launch Status at 2026-06-13 22:15 UTC

- E1 cache issue resolved by manually finalizing a full-size completed `model-00003` blob and symlink after repeated HF downloader stalls on the A100. Verified `AutoConfig` and `AutoTokenizer` load from local cache.
- E1 launched; vLLM is loading Qwen3-8B for `chat_t0`.
- E2 SFT is stepping: step 18 at about 24 sec/step.
- E4 SFT is stepping: step 19 at about 24 sec/step.
- E5 GRPO is stepping: step 9/300 at about 50-52 sec/step.
- Expected wall update:
  - E2 SFT: about 1.8 h before Brier-selected E3 starts; E3 likely slower than planned if GRPO remains near E5 speed.
  - E4 SFT: about 1.5 h before held-out checkpoint evals.
  - E5: about 4.3 h for RL plus post-RL eval.

## Checkpoint at 2026-06-13 22:27 UTC

- E1 completed and results were pulled to `tmp/session2/results/E1/`.
- Terminated E1 pod `ypcieyl1am1zgo`; remaining burn rate is three H200 pods at about `$13.17/hr`.
- E1 results:
  - Chat template, `temperature=0.0`: ForecastBench parsed `247/401` (`61.6%`), Brier `0.1524`, constant baseline `0.1411`; Freeciv-val parsed `5/57` (`8.8%`), Brier `0.4245`, constant baseline `0.1116`.
  - Chat template, `temperature=0.6`: ForecastBench parsed `248/401` (`61.8%`), Brier `0.1414`, constant baseline `0.1325`; Freeciv-val parsed `8/57` (`14.0%`), Brier `0.2184`, constant baseline `0.1580`.
- Initial interpretation: H1 is not supported on parse-rate. The chat template did not restore ForecastBench parsing to the expected `>=85%`; t=0.6 Brier improved versus the session-1 no-template `0.173`, but it remains worse than the parsed-set constant baseline.
- E2 SFT: step `50`, epoch `0.730`, loss `0.0464`, no epoch-end Brier line yet.
- E4 SFT: step `51`, epoch `0.919`, loss `0.0461`, no epoch-end Brier line yet.
- E5 RL-from-base: still stepping; logs are dominated by TRL completion tables, with no final trainer state or eval yet.

## Checkpoint at 2026-06-13 22:40 UTC

- E4 crossed epoch 1; the Brier callback completed mechanically and training resumed.
- E4 callback result was too sparse to use as a checkpoint selector: `n_parsed=1/30`, Brier `0.7225`. This is a significant measurement issue, not a training crash. The likely cause is that vanilla `model.generate` in the callback is much less reliable at producing the exact parseable tail than the vLLM eval path, especially at long context and early SFT checkpoints.
- E2 reached epoch 1 and is inside its first callback. Based on the E4 sparse parse result, I no longer trust callback Brier alone for E3 checkpoint selection.
- Compromise: leave the SFT runs in place, keep callback output as a diagnostic, but change the E2-to-E3 handoff to evaluate every E2 checkpoint with `runpod/04_eval.py` using the actual chat-template/max-token eval path before selecting the E3 initializer. This adds H200 time but measures the thing E3 depends on more directly.
- Because the E2 pod was already running the old `run_E2E3.sh`, I installed a handoff guard (`/workspace/scripts/e2_handoff_guard.sh`, PID `2243`) that waits for the current SFT process to exit, stops the old shell if needed, then runs `/workspace/scripts/run_E2_post_sft_E3.sh`.
- E5 RL-from-base: about step `39/300`, GPU at 100%, roughly `51 sec/step`. Remaining RL time is still about `3.6-3.8 h` plus eval.

## Checkpoint at 2026-06-13 22:46 UTC

- E2 epoch-1 callback finished with `n_parsed=0/30` and null Brier; SFT resumed and reached step `81`, epoch `1.175`.
- This strengthens the callback diagnosis: the running callback's `max_new_tokens=256` is too short under chat-template/thinking-mode generation, so it can fail to reach the final `PROBABILITY:` line even when the checkpoint may be usable under the planned 1024-token eval.
- Patched local `runpod/02_sft.py` so future/restarted callbacks use `max_new_tokens=1024`, pass an attention mask, and write `brier_callback_raw.jsonl` tails for debugging. This does **not** affect the already-running E2/E4 processes, so their `brier_curve.jsonl` files remain diagnostic only.
- Cost estimate: E1 ran from roughly `21:53-22:27 UTC` at `$1.49/hr`; the three H200 pods have been running since roughly `21:53 UTC` at `$13.17/hr` combined. Rough spend by this checkpoint is about `$12-13`, excluding small rounding/boot variance.

## Checkpoint at 2026-06-13 22:55 UTC

- E2 SFT healthy: step `104/276`, epoch `1.511`, GPU at 100%. Callback curve still has only the epoch-1 null parse line.
- E2 handoff guard still idle and watching the SFT PID; no E3 process has started.
- E4 SFT healthy: step `102/224`, epoch `1.829`, GPU at 100%. No post-SFT checkpoint evals yet.
- E5 RL-from-base healthy: step `57/300`, GPU at 100%, current speed around `50-52 sec/step`.
- Rough spend by this checkpoint is about `$14-15`. Remaining expected wall time: E4 roughly `1-1.5 h` including callbacks/evals; E5 roughly `3.5 h` plus eval; E2/E3 likely longest because E2 still has ~2.5 epochs plus checkpoint evals before E3 starts.

## Checkpoint at 2026-06-13 23:05 UTC

- E2 SFT healthy: step `131/276`, epoch `1.905`, GPU at 100%; guard still idle and watching.
- E4 SFT healthy: completed epoch 2 callback, still sparse (`n_parsed=1/30`, Brier `0.7225`), then resumed. This further confirms the callback is not a reliable selector in the live run.
- E5 RL-from-base healthy: step `70/300`, GPU at 100%, no final adapter/eval yet.
- No pods ready to terminate.

## Eval Parser Patch at 2026-06-13 23:12 UTC

- User pointed out the main parse lever: disable Qwen3 thinking mode for eval, and optionally use vLLM guided decoding to force a parseable `PROBABILITY:` suffix.
- Patched local and remote `runpod/04_eval.py` on E2/E3, E4, and E5 pods:
  - `--chat-template` now passes `chat_template_kwargs={"enable_thinking": False}` by default.
  - Added `--enable-thinking` escape hatch if we need the old behavior.
  - Added vLLM `GuidedDecodingParams(regex=...)` by default to force a parseable final probability line.
  - Added `--no-guided-probability` escape hatch.
  - Added automatic fallback to unguided decoding if vLLM rejects the guided regex.
- Patched local and remote `runpod/02_sft.py` callback for future/restarted runs to use `enable_thinking=False`, `max_new_tokens=1024`, an attention mask, and raw-tail debug logging. The currently running E2/E4 SFT processes keep their old in-memory callback code, so their callback curves remain diagnostic only.
- Scope note: I did **not** change `03_rl.py` prompt construction mid-flight. E5 is already running with the original RL prompt format; changing only the future E3 RL prompt format would confound the E3-vs-E5 ablation. The eval/selection path is now no-thinking/guided; the RL training prompt path remains as launched.
- Verified remote syntax on all three pods and tokenizer-only `enable_thinking=False` behavior on the E2 pod.

## Checkpoint at 2026-06-13 23:17 UTC

- E2 completed epoch 2 under the old in-memory callback code; callback again parsed `0/30`, then eval/checkpointing ran. This does not affect the post-SFT checkpoint-selection evals, which will use the patched `04_eval.py`.
- E2 SFT status: step `138/276`, epoch `2.0`, no E3 yet, guard still waiting for SFT PID exit.
- E4 SFT status: step `132/224`, epoch `2.36`.
- E5 RL-from-base status: step `79/300`, still around `52 sec/step`.
- No pods ready to terminate.

## Checkpoint at 2026-06-13 23:31 UTC

- E2 SFT healthy: step `178/276`, epoch `2.584`; guard still idle.
- E4 SFT reached step `168/224`, epoch `3.0`, and is in the old callback/eval/checkpointing pause. The eventual post-SFT checkpoint eval loop will use patched `04_eval.py`.
- E5 RL-from-base healthy: step `98/300`, still about `52 sec/step`.
- Rough spend is now about `$22-24` including E1 and current H200 runtime. No pod is ready to terminate.

## Checkpoint at 2026-06-13 23:46 UTC

- E2 SFT reached step `207/276`, epoch `3.0`, and is in the old callback/eval/checkpointing pause. Guard still idle, no E3 yet.
- E4 SFT is in final epoch: step `193/224`, epoch `3.45`. This is the next likely pod to finish useful training; after SFT, it should evaluate checkpoints with the patched no-thinking/guided parser.
- E5 RL-from-base reached step `116/300`; still no final adapter/eval.
- No pods ready to terminate yet.

## Checkpoint at 2026-06-13 23:57 UTC

- E4 SFT is nearly done: step `221/224`, epoch `3.955`; post-SFT checkpoint evals should start soon and will be the first test of the no-thinking/guided eval patch.
- E2 SFT healthy: step `229/276`, epoch `3.321`; old callback parse remains `0/30` through epoch 3, as expected.
- E5 RL-from-base healthy: step `128/300`.
- No pods ready to terminate yet.

## E4 Eval Recovery at 2026-06-14 00:13 UTC

- E4 SFT completed all 4 epochs. The original post-SFT eval loop crashed before writing eval files because a held-out Freeciv prompt was `10,915` chat-templated tokens while `04_eval.py` hard-coded `max_model_len=10240`.
- Patched `04_eval.py` to expose `--max-model-len` and default to `16384`; rsynced to all active pods so E2/E3 and E5 final evals avoid the same failure.
- Reran only E4 checkpoint evals, no SFT redo.
- Patched no-thinking/guided eval worked: parse counts were `192/192`, `190/192`, `192/192`, `192/192` for checkpoints `56`, `112`, `168`, `224`.
- E4 held-out-template Brier by checkpoint:
  - `checkpoint-56`: Brier `0.1925`, constant `0.1510`
  - `checkpoint-112`: Brier `0.2192`, constant `0.1510`
  - `checkpoint-168`: Brier `0.2186`, constant `0.1496`
  - `checkpoint-224`: Brier `0.2362`, constant `0.1510`
- Initial H3 read: no evidence of template-level generalization; every checkpoint is worse than the held-out parsed-set constant baseline, with the best checkpoint at epoch 1.
- E4 pod is now free. I am using it for an additional base no-thinking/guided eval (`runs/E1_nothink_guided`) before terminating because this directly measures the parse fix suggested after E1.

## E4 Closed at 2026-06-14 00:30 UTC

- Additional base no-thinking/guided eval completed on the freed E4 pod:
  - `t=0.0`: ForecastBench parsed `401/401`, Brier `0.1658`, constant `0.1551`; Freeciv-val parsed `57/57`, Brier `0.1842`, constant `0.1502`.
  - `t=0.6`: ForecastBench parsed `401/401`, Brier `0.1661`, constant `0.1551`; Freeciv-val parsed `57/57`, Brier `0.1821`, constant `0.1502`.
- Interpretation: disabling thinking + guided probability fixed parse coverage completely, but did not make the base model beat constant-baseline calibration.
- Pulled E4 artifacts locally under `tmp/session2/results/E4/`: final adapter, all four checkpoint adapters (`56/112/168/224`), eval JSON/logs, SFT metrics, and E1 no-thinking/guided evals.
- Terminated E4 pod `pn0g5gm98bk13n`. Remaining active pods: E2/E3 and E5. Burn rate is now about `$8.78/hr`.

## Checkpoint at 2026-06-14 00:35 UTC

- E2 SFT finished all 4 epochs. The handoff guard stopped the old script path and ran the patched post-SFT checkpoint evals.
- E2 checkpoint evals used no-thinking/guided parsing and all parsed `57/57` Freeciv-val examples:
  - `checkpoint-69`: Brier `0.1946`, constant `0.1502`
  - `checkpoint-138`: Brier `0.2001`, constant `0.1502`
  - `checkpoint-207`: Brier `0.1818`, constant `0.1502`
  - `checkpoint-276`: Brier `0.2044`, constant `0.1502`
- The handoff selected `runs/E2/checkpoint-207` for E3 because it had the lowest Freeciv-val Brier among parsed checkpoints.
- Important interpretation: the parse-fixed/chat-template SFT run did **not** reproduce Session 1's Freeciv-val win. Under the no-thinking/guided eval, every E2 checkpoint is worse than constant baseline. This makes E3 a robustness check of the planned RL fix, not an RL-on-a-strong-SFT-initializer test.
- E3 RL has started from `checkpoint-207`; early speed is about `60 sec/step`, slower than the original plan.
- E5 RL-from-base is still running, around step `178/300`.

## Checkpoint at 2026-06-14 00:50 UTC

- Active pods confirmed: only E2/E3 (`eo4f0kxunsu3xa`) and E5 (`uacz82qsf7wdqd`). Current burn rate: about `$8.78/hr`.
- E3 RL-from-SFT is healthy but slow: step `14/300`, about `55-58 sec/step`, GPU at 100%.
- E5 RL-from-base is healthy: step `191/300`, about `53 sec/step`. It should hit checkpoint 200 soon and finish RL in roughly `1.6 h` plus final eval.
- Rough spend by this checkpoint: about `$38-40` order-of-magnitude including the now-terminated E1/E4 pods and current E2/E5 runtime. The run is likely to exceed the original `$45` forecast if E3 continues at ~1 minute/step, because E3 alone still has several hours left.

## Checkpoint at 2026-06-14 01:06 UTC

- E3 RL-from-SFT: step `30/300`, still healthy at about `59 sec/step`. No checkpoint yet.
- E5 RL-from-base: step `209/300`, checkpoint-200 exists.
- E5 checkpoint-200 trainer metrics:
  - Reward entries: `40` (logged every 5 steps).
  - Average reward over first four logged entries: about `-0.130`.
  - Average reward over last four logged entries through step 200: about `-0.118`.
  - Read as mild reward improvement but noisy; final eval is still the decision point.

## Checkpoint at 2026-06-14 01:26 UTC

- E3 RL-from-SFT: step `51/300`, around `60 sec/step`, no checkpoint yet.
- E5 RL-from-base: step `233/300`, around `52 sec/step`, no final adapter/eval yet.
- No pods ready to stop.

## Checkpoint at 2026-06-14 01:56 UTC

- E3 RL-from-SFT: step `82/300`, about `60 sec/step`, checkpoint 100 not reached yet.
- E5 RL-from-base: step `268/300`, about `53 sec/step`, roughly 30 minutes of RL remaining plus final eval.
- No pods ready to stop.

## Checkpoint at 2026-06-14 02:16 UTC

- E3 RL-from-SFT: step `103/300`; checkpoint-100 exists.
- E3 checkpoint-100 trainer metrics: early rewards around `-0.15`; latest logged rewards fluctuate around `-0.11` to `-0.14`. Mild improvement, still noisy.
- E5 RL-from-base: step `291/300`; final RL should finish soon, then patched final eval runs.
- No pod ready to stop yet.

## E5 Closed at 2026-06-14 02:36 UTC

- E5 RL-from-base completed all 300 steps and final patched eval.
- Final E5 eval:
  - ForecastBench: parsed `401/401`, Brier `0.1677`, constant `0.1551`, base rate `0.1920`.
  - Freeciv-val: parsed `57/57`, Brier `0.1711`, constant `0.1502`, base rate `0.5614`.
- Interpretation: RL-from-base achieved full parse coverage under no-thinking/guided eval, but did not beat constant-baseline calibration on either suite. Relative to the base no-thinking/guided eval (`FB 0.1658/0.1661`, Freeciv `0.1842/0.1821`), E5 slightly worsened FB and slightly improved Freeciv, but neither result crosses the constant baseline.
- Pulled E5 artifacts locally under `tmp/session2/results/E5/`: final adapter, checkpoint-200 and checkpoint-300 adapters/trainer states, final eval JSON, and logs.
- Terminated E5 pod `uacz82qsf7wdqd`. Only E2/E3 remains active. Burn rate is now about `$4.39/hr`.

## Checkpoint at 2026-06-14 03:07 UTC

- E2/E3 is the only active pod.
- E3 RL-from-SFT: step `167/300`, about `60 sec/step`, no checkpoint-200 yet.
- Expected remaining time: roughly `2.2 h` for RL plus final eval and artifact pullback.

## Checkpoint at 2026-06-14 03:47 UTC

- E3 RL-from-SFT: step `208/300`; checkpoint-200 exists.
- E3 checkpoint-200 reward read: reward improved mildly by checkpoint-100, but by checkpoint-200 latest logged rewards are mostly around `-0.16`, with one `-0.126` outlier. This is not a clear monotonic learning curve.
- Expected remaining time: roughly `90 min` for RL plus final eval/pullback.

## Checkpoint at 2026-06-14 04:32 UTC

- E3 RL-from-SFT: step `255/300`; roughly `45 min` of RL remaining plus final eval.
- No final adapter/eval yet; pod still useful and remains active.

## E3 / Session Closed at 2026-06-14 06:14 UTC

- E3 RL-from-SFT completed all 300 steps and final patched no-thinking/guided eval.
- Final E3 eval:
  - ForecastBench: parsed `401/401`, Brier `0.1928`, constant `0.1551`, base rate `0.1920`.
  - Freeciv-val: parsed `57/57`, Brier `0.1891`, constant `0.1502`, base rate `0.5614`.
- E3 reward did not improve materially: first four logged rewards averaged about `-0.1496`; last four averaged about `-0.1520`.
- Pulled E2/E3 artifacts locally under `tmp/session2/results/E2E3/`, including all E2 checkpoint adapters (`69/138/207/276`), E3 checkpoint adapters (`200/300`), the final E3 adapter, eval JSONs/logs, trainer states, and handoff logs. The first broad rsync timed out after most files; targeted recovery pulled the final E3 adapter before termination.
- Terminated E2/E3 pod `eo4f0kxunsu3xa`. RunPod REST API then reported `0` active pods.
- Wrote Session 2 plots under `runpod/plots/v2/`:
  - `01_session2_headline.png`
  - `02_sft_brier_curves.png`
  - `03_e4_per_template.png`
  - `04_rl_rewards.png`
- Approximate actual spend from journaled runtimes:
  - E1 A100: `~0.57 h * $1.49/hr = $0.85`
  - E4 H200: `~2.62 h * $4.39/hr = $11.5`
  - E5 H200: `~4.72 h * $4.39/hr = $20.7`
  - E2/E3 H200: `~8.3 h * $4.39/hr = $36.4`
  - Total: `~$69-70`, versus the original `$45` forecast.
- Main reason for cost overrun: GRPO wall time was close to `~1 min/step` rather than the planned `~3 h` per 300-step run, E2/E3 stayed up for the dependent handoff and artifact pullback, and E4 was reused for the no-thinking/guided base diagnostic before termination.
