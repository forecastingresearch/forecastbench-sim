# fbsim v3 rerun — run log (2026-09-02, hand-driven RunPod fleet)

Scripts: `tmp/fbsim_v3_run/` (rp.sh, bringup.sh, launch_workers.sh, status.sh, rebalance.sh, requeue_slice.sh,
switch_anchor.sh, finish_all.sh, pull2.sh, sg_tables.py, consolidate_v4.py, sweep_v4.py, regen_reports.py,
anchor_qc.py). On-pod: `on_pod/` (worker.sh lanes over a file-rename queue, anchor_one.sh, sync_loop.sh,
finish_pod.sh). Ledger: `state/pods.tsv`. Bundle used on pods: `tmp/pilot_v2/fleet_prep/pod_bundle/`.

## Timeline (UTC)
- 01:03 first pods (helper misread HTTP codes; 7 pods created at once, all kept). 01:10 anchors 7001-7024 played to T61 on w01 (~1 min each), all healthy by savegame truth.
- 01:11 smoke: 2 forks of old seed1000 anchor on w02: 497 s, 153 saves, observability gate PASS.
- 01:16 scavenged 23 more pods (6x cpu5c-32, 17x cpu3c-32). 01:21 production launched: 8 anchors x 1000 rng 5001-6000 dealt by vCPU over 27 workers; 600 lanes.
- 01:2x pod hqbyzv01525qdr deleted by mistake before work; its 307 tasks redistributed.
- 01:45 archive volume hkwejil48w expanded 100 -> 250 GB (projection ~87 GB corpus + 12 GB v2).
- 02:0x observer-death truncations found (all anchor 7016). Server-wait patch tested: server idles after player 0 dies (no fix). fork_one.sh patched: truncated forks publish with TRUNCATED marker instead of failing.
- 02:16 anchor 7016 (3.1% truncations) replaced by 7011 (0 so far); 492 remaining 7016 tasks dropped, 438 finished 7016 forks kept on archive as extra.
- 02:3x consolidate_v4.py + sweep_v4.py validated on 20 real forks (0 cross-check mismatches). Turn-60 reports for the 8 anchors regenerated with exact series and exact turn-1..60 events (tmp/fbsim_v3_corpus/reports/).

## Final anchors
7001 7003 7005 7008 7010 7011 7014 7022 (T60 saves in state/anchors/, full anchor data + 62 saves in state/anchors_raw/).

## Patches made today
- bundle fork_one.sh: TRUNCATED publish path (state/fork_one.sh.orig = before)
- bundle fork_manager.py: FBSIM_WAIT_SERVER_END (env-gated, inert; on w01 only)
- main repo txt_report.py: savegame-truth scores/government series, rank from scores, map_interval<=0 disables maps (state/txt_report.py.before_v3 = before)
- 02:5x anchor 7014 fork rng5290: observer died at turn 178 -> published TRUNCATED (1 of ~700 so far); 6 more truncations are leftover 7016 forks.
- 03:2x-04:1x production tail; finish_pod (sg_tables + final sync) per drained pod; 25 workers verified against the archive and deleted (finish_all.sh).
- 03:5x five deterministic server hangs identified (7001 rng5014/5841, 7005 rng5439/5639, 7022 rng5823): autosaves stop at a fixed turn with all civs alive; same turn on rerun. Substituted with seeds 6001+ on w01 (state/SUBSTITUTIONS.md). Archive volume at 92 GB.
- 04:2x RunPod REST API timed out for ~10 min; rp.sh got an address cache (state/addr_cache.tsv).
- 04:3x all substitutes done; w01 finishing; final chain (state/final_chain.sh -> finalize.sh) launched: pull2, corpus_qc, consolidate_v4, sweep_v4, mine_v3 into tmp/fbsim_v3_corpus/.
- 2026-09-03T08:06:05Z archive pods deleted; volume hkwejil48w (250 GB, 104 GB used: fbsim_v3 92 GB + fbsim-v2 12 GB) retained as the only cloud resource.
- QC round 1 (owner): S4 redefined as arrivals at a DIFFERENT non-anarchy government (62% of previously counted "changes" were restorations of the same government after anarchy); W1 reworded "Will X complete the W by T?" (uniqueness stated in criteria only); "territory (tiles)" -> "territory size"; S2 war-end dropped (duplicate of NW2 cease-fire-by for pairs at war at t60). Re-generation + re-draw deferred until QC flags are collected.
- QC round 2 (owner): S4 counts every transition incl. anarchy ("Anarchy counts as a form of government, for this question."); gov_at reworded "Will X's government be G at turn T?"; pool deduped by question text; natcond admissibility 0.02 < p(Y|X) < 0.98; civ names with "and", capitalised state names, 99% drawdown wording kept as is. Re-draw v1.4.
- QC round 3 (owner): timing block (P7/P8/P9, censored "never") REMOVED from the continuous set — no special-case scoring; replaced by 60 value-at-T questions (score, population, cities, territory; 3 per metric per horizon), plain CRPS. "(razed)" moved to criteria. Continuous supply filter: IQR>0 and median>0. Re-draw v1.5.
- 2026-09-03 draw v1.5 final: criteria attached to every item (criteria_v1.py, family+params keyed); continuous items carry their 1,000 truth values; natcond cells carry the turn-2 preamble. Docs: draw_v1/README.md (schema + scoring), CHANNEL_READINESS.md (decisions section). Not yet done: elicitation harness for the new schema (turn-1, turn-2 + controls), CRPS scorer over `values`, dry run, paid derisk, conditional-continuous pilot (60), family_specs.json rev 27.
- Criteria review (Opus subagent, 63-item sample): (1) tech-discovered event/state ambiguity -> clause "does not know it at turn 60"; (2) losses: each loss event counts separately (mirrors captures); (3) diplomatic-state 'reach' on pairs already in the state at t60 -> those instances excluded at generation (2 of 75 drawn items) + clause. Also: eliminated-civ values clause on value families; Techs anchor on tech-count families. Controls assigned independently (300 no-news, 300 single-prompt). Re-draw v1.6.
- 2026-09-04 v1.7: techs = real technologies (A_NONE excluded everywhere); eliminated-civ admissibility rule; report cuts + diplomatic_change events; NW2 questions state the 61-T window; tech-known clause reverted per owner. Reports regenerated.
- 2026-09-04: BORDERS section + barbarians header line added to reports (regen_reports.py); all 8 regenerated.
