#!/usr/bin/env python
"""release_appendix_2026-09-20.py --paper-root DIR [--check]

Appendix F (release) and the reproducibility statement, 20 September 2026: the yellow items that facts in hand can
close are written in green, and the decisions that remain (public host, data license, maintenance policy) stay yellow
but shorter.  Facts used: the project repository is public under GPL-3.0 (its name is withheld in the anonymous
submission and given in a LaTeX comment); the FreeCiv material of the paper is on a branch of it and the Micropolis world
on a fork; Micropolis's per-question results and usage records are in the paper repository under data/micropolis/;
CivRealm's sources carry GPL-3.0 headers; the FreeCiv archive is a 250 GB network volume at about $17.50 per month;
the raw FreeCiv outputs are now the grouped prompts of 20 September plus the natural-conditional records.  Also the
reasoning-effort note of Appendix D, which promised a repeat "in the second run": the check was not repeated.
Each replacement must match exactly once.
"""
import argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--paper-root", required=True); ap.add_argument("--check", action="store_true")
a = ap.parse_args(); P = Path(a.paper_root).resolve()

R = [
    ("appendix/F_release.tex",
     r"The three worlds, the scoring code and the released files will live in one repository. \yellow{Its name is to be confirmed (Jaeho); the code is not yet public.}",
     r"The three worlds, the scoring code and the released files live in one public repository under GPL-3.0, named in the camera-ready version; the FreeCiv material of this paper is on a branch of it and the Micropolis world on a fork, both to be merged for the release."
     "\n% Repository: github.com/forecastingresearch/forecastbench-sim (public, GPL-3.0; main holds the shared package, worlds/pandemic and the earlier worlds/freeciv code);\n% FreeCiv v3: branch freeciv-v3, worlds/freeciv/fbsim_v3; Micropolis: fdrocha/forecastbench-sim, branch micropolis, worlds/micropolis."),
    ("appendix/F_release.tex",
     r"\yellow{Both file kinds are described in the world code; the files themselves are with Fabio.}",
     r"The per-question results (\texttt{data/micropolis/binary\_forecasts.csv} and \texttt{continuous\_forecasts.csv}), the usage record of every call (\texttt{model\_usage.csv}) and the model scores are in the paper repository; the ground-truth files per snapshot are described in the world code \yellow{(files: Fabio)}."),
    ("appendix/F_release.tex",
     r"FreeCiv: one JSON-lines file per model with 2,023 rows, one per item and prompt kind (turn 1, turn 2 and the no-news control). Each row holds the response text and the value read from it. It also records the provider that served the call, the prompt, completion, stored-prefix and reasoning token counts, the cost, the finish reason, the reasoning parameter sent and the reasoning text where the provider returned it. A run log per model accompanies the rows.",
     r"FreeCiv: for each model, the 44 grouped prompts of 20 September with their responses, one JSON-lines record per prompt with the value read for each of its questions, and the 854 natural-conditional records (turn 1, turn 2 and the no-news control), one per prompt, from 9 September or from the 20 September rerun for the two re-pinned models. Each record holds the response text and the values read from it, the provider that served the call, the prompt, completion, stored-prefix and reasoning token counts, the cost, the finish reason, the reasoning parameter sent and the reasoning text where the provider returned it. The one-question run of 9 September is released in full beside them (\cref{app:ablation-grouping}), and a run log per model accompanies the records."),
    ("appendix/F_release.tex",
     r"\yellow{To confirm per simulator (Jaeho, with Fabio and Nick). The Freeciv server is distributed under GPL-2.0 \citep{freeciv2026}. MicropolisCore is under GPL-3.0, with the trademark terms of the Micropolis Public Name License \citep{micropolis2008}. Starsim is under the MIT license \citep{starsim2026software}. The CivRealm license is to be checked \citep{qi2024civrealm}. The ForecastBench-Sim repository carries a GPL-3.0 license file, so the world packages and the forks carry the same license. The question sets, world reports, scores and human-pilot data need a data license (to decide, Jaeho). Redistribution of raw model outputs is subject to each provider's terms (to check, Jaeho).}",
     r"\green{The Freeciv server is distributed under GPL-2.0 \citep{freeciv2026} and CivRealm under GPL-3.0 \citep{qi2024civrealm}. MicropolisCore is under GPL-3.0, with the trademark terms of the Micropolis Public Name License \citep{micropolis2008}. Starsim is under the MIT license \citep{starsim2026software}. The ForecastBench-Sim repository carries a GPL-3.0 license file, so the world packages and the forks carry the same license.} \yellow{The question sets, world reports, scores and human-pilot data need a data license (to decide, Jaeho), and redistribution of raw model outputs is subject to each provider's terms (to check, Jaeho).}"),
    ("appendix/F_release.tex",
     r"\yellow{To decide (Jaeho). Who hosts the archive and the code, and at what cost; the FreeCiv volume costs about \$17.50 per month at present. How often the sets are refreshed,",
     r"\green{The code and the paper's data files are in the repository. The FreeCiv archive is a 250~GB network volume at a cloud provider, billed at about \$17.50 per month.} \yellow{To decide (Jaeho): the public host of the archive. How often the sets are refreshed,"),
    ("appendix/F_release.tex",
     r"Needs (Jaeho): the v3 archive (92 GB) moved to a public host, with the per-world truth files, the corpus (question sets, reports, raw outputs, scores) and the run scripts. The batched rerun at 50 questions per prompt as a new draw version. Repository names. The data license. The maintenance decisions above.",
     r"Needs (Jaeho): the v3 archive (92 GB, on the 250 GB volume) moved to a public host, with the per-world truth files, the corpus (question sets, reports, the raw outputs of both elicitations, scores) and the run scripts. The data license. The maintenance decisions above. Done on 20 September: the repository is public under GPL-3.0, and the grouped run of that day is the released elicitation of draw v1.8, with the one-question run of 9 September beside it."),
    ("sections/11_reproducibility.tex",
     r"\Cref{app:release} names the repository \yellow{(name to confirm, Jaeho)} and license \yellow{(to decide, Jaeho)} and gives the scripts that score the outputs and produce every figure.",
     r"\Cref{app:release} describes the repository and the licenses \yellow{(data license to decide, Jaeho)} and gives the scripts that score the outputs and produce every figure."),
    ("appendix/D_ablations.tex",
     r"At these levels the effort setting does not produce the compression. \yellow{Provisional, one question per prompt; we will repeat the check in the second run.}",
     r"At these levels the effort setting does not produce the compression. The check was made on the run of 9 September and was not repeated on the grouped prompts."),
]
texts = {}
for f, old, new in R:
    s = texts.get(f) or (P / f).read_text()
    if s.count(old) != 1: print(f"ABORT: {f}: found {s.count(old)} times: {old[:80]!r}"); sys.exit(1)
    texts[f] = s.replace(old, new)
for f, s in texts.items():
    print(("would write" if a.check else "wrote"), f)
    if not a.check: (P / f).write_text(s)
