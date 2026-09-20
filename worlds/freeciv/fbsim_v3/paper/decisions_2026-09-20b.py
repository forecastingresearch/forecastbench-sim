#!/usr/bin/env python
"""decisions_2026-09-20b.py --paper-root DIR [--check]

Jaeho's decisions of 20 September 2026 (evening) written into the paper:
  Appendix E stays; its caveat says the pilot predates the benchmark's current version but shows how people do on this
  shape of text.  Data license CC BY 4.0 (Appendix F, Section 11).  The 92 GB of saved games is not hosted publicly:
  it stays on the volume, available on request, and the release carries the derived files.  Maintenance paragraph
  reduced to what is decided.  The reason for 50 rather than 100 questions per prompt (a longer list would test
  long-context handling as much as forecasting, and load the score on general capability) in Appendix D.
Each anchor must match exactly once.
"""
import argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--paper-root", required=True); ap.add_argument("--check", action="store_true")
a = ap.parse_args(); P = Path(a.paper_root).resolve()
R = [
    ("appendix/E_human_pilot.tex",
     r"\yellow{The two pilot games come from an earlier version of the pipeline and are not among the eight v3 anchors of \cref{sec:freeciv}, so the pilot scores are not comparable with the model scores in this paper. A larger study on the v3 question sets, scored against the replay truth, is future work (Jaeho).}",
     r"\green{The two pilot games come from an earlier version of the benchmark and are not among the eight anchors of \cref{sec:freeciv}, so the pilot scores are not comparable with the model scores in this paper. The pilot still shows how well people forecast from reports and questions of this shape, which is the context in which the model scores should be read; a larger study on the released question sets, scored against the replay truth, is future work.}"),
    ("appendix/F_release.tex",
     r"\yellow{The question sets, world reports, scores and human-pilot data need a data license (to decide, Jaeho), and redistribution of raw model outputs is subject to each provider's terms (to check, Jaeho).}",
     r"\green{The question sets, world reports, scores and human-pilot data are released under the Creative Commons Attribution 4.0 license (CC BY 4.0). The raw model outputs are released as received, and each provider's terms govern their further use.}"),
    ("appendix/F_release.tex",
     r"The archive of saved games and game data for all 8,000 replays is 92 GB \yellow{(on a cloud volume; public host to decide, Jaeho)}. The per-world truth files are 1.1 GB, and the corpus needed for scoring is 311 MB compressed.",
     r"The archive of saved games and game data for all 8,000 replays is 92 GB; it stays on the cloud volume where it was written and is available on request. The release carries what is derived from it: the per-world truth files (1.1 GB) and the corpus needed for scoring (311 MB compressed), which suffice to score the released outputs, to score a new model and to draw new questions from the same replays."),
    ("appendix/F_release.tex",
     r"\yellow{To decide (Jaeho): the public host of the archive. How often the sets are refreshed, and whether a refreshed set replaces or joins the released one. How a new model enters the model list, given that every model needs an ECI value. How set versions and score tables are named. A point of contact for errors in the released files. Whether the authors will run submitted models or only publish the materials.}",
     r"\green{The released files are versioned by draw (this paper's sets are draw v1.8) and by run date. A new draw follows \cref{app:release-refresh} and joins the released sets rather than replacing them, so that scores on an older draw stay comparable. Errors in the released files are reported through the repository's issue tracker.}"),
    ("appendix/F_release.tex",
     r"Needs (Jaeho): the v3 archive (92 GB, on the 250 GB volume) moved to a public host, with the per-world truth files, the corpus (question sets, reports, the raw outputs of both elicitations, scores) and the run scripts. The data license. The maintenance decisions above. Done on 20 September: the repository is public under GPL-3.0, and the grouped run of that day is the released elicitation of draw v1.8, with the one-question run of 9 September beside it.",
     r"Needs (Jaeho): the per-world truth files, the corpus (question sets, reports, the raw outputs of both elicitations, scores) and the run scripts placed in the repository's release; the saved games stay on the volume, available on request. Done on 20 September: the repository is public under GPL-3.0, the data license is CC BY 4.0, and the grouped run of that day is the released elicitation of draw v1.8, with the one-question run of 9 September beside it."),
    ("sections/11_reproducibility.tex",
     r"\Cref{app:release} describes the repository and the licenses \yellow{(data license to decide, Jaeho)} and gives",
     r"\Cref{app:release} describes the repository and the licenses and gives"),
    ("appendix/D_ablations.tex",
     r"which is why the reported run keeps the Micropolis limit of 20.",
     r"which is why the reported run keeps the Micropolis limit of 20. We kept 50 rather than 100 questions per prompt, although the Micropolis ablation found no difference beyond four, because a longer list tests how a model handles a long context as much as how it forecasts, and a score that rose with general capability for that reason alone would confound the comparison with ECI."),
    ("appendix/D_ablations.tex",
     r"\item One sentence on why 50 rather than 100, given no measured difference beyond four (Fabio, Jaeho).",
     r"\item One sentence on why 50 rather than 100, given no measured difference beyond four (Fabio; Jaeho's reason is stated in \cref{app:ablation-grouping})."),
]
texts = {}
for f, old, new in R:
    s = texts.get(f) or (P / f).read_text()
    if s.count(old) != 1: print(f"ABORT: {f}: found {s.count(old)}: {old[:70]!r}"); sys.exit(1)
    texts[f] = s.replace(old, new)
for f, s in texts.items():
    print("would write" if a.check else "wrote", f)
    if not a.check: (P / f).write_text(s)
