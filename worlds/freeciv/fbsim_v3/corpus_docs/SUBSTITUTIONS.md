# Seed substitutions — fbsim v3 corpus (2026-09-02 rerun)

Each anchor has 1,000 continuations, RNG seeds 5001–6000. Five seeds hung deterministically: the freeciv server
stopped advancing/autosaving at a fixed turn with every civ alive (dozens of cities each), the client kept
waiting until the 30-minute timeout, and a second run of the same seed (90-minute timeout, idle pod) hung at the
same turn. These are technical failures, not game outcomes, and carry no information about the future.
As in the 2026-08-24 campaign (SUBSTITUTIONS.md there), each hung seed is replaced by a fresh seed from 6001
upward on the same anchor. Substitution is at the seed level, never conditioned on an outcome.

| anchor | hung seed | last server autosave turn | substitute seed |
|---|---|---|---|
| 7001 | 5014 | 193 | 6001 |
| 7001 | 5841 | timed out twice on the fleet (turn not recorded) | 6002 |
| 7005 | 5439 | 123 | 6001 |
| 7005 | 5639 | 188 | 6002 |
| 7022 | 5823 | 206 | 6001 |

Rate: 5 / 8,000 = 0.06 % (previous campaign: 5 / 8,000).

Also recorded, NOT substituted (they are real futures, published with a TRUNCATED marker and all saves to the
last recorded turn): observer-civ deaths — 7001 rng5482 (turn 117), 7014 rng5290 (turn 178).
Anchor 7016 (3.1 % observer deaths) was replaced wholesale by anchor 7011 before its run completed; its 507
finished forks remain on the archive volume under a7016/ as extra material.

## Observer-death replacements (user decision, 2026-09-03 05:15Z)
At the owner's direction the two observer-death forks are ALSO replaced so that all 8x1000 continuations run to
turn 210: 7001 rng5482 (observer dead at 117) -> rng6003; 7014 rng5290 (observer dead at 178) -> rng6001.
Note: unlike the hang substitutions, this is conditioned on an outcome (the observed civ being eliminated);
at 1/1000 per anchor the effect on any frequency is <= 0.001. The two originals stay on the volume under
/vol/fbsim_v3/_excluded/ and locally under tmp/fbsim_v3_corpus/raw_excluded/.
