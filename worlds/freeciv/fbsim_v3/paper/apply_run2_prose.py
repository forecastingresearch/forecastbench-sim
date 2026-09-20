#!/usr/bin/env python
"""apply_run2_prose.py --paper-root DIR [--numbers run2_numbers.json] [--check]

The FreeCiv-owned prose edits that run 2 requires: every statement that FreeCiv's numbers are provisional or that
its questions went one per prompt, replaced by the run-2 protocol; the "(provisional)" qualifiers in headings and the
protocol table; the Appendix A run paragraph and the fulfilled items of its to-do box.  Only green or dark-yellow text
in FreeCiv-owned passages is touched; red (Nick), blue (Fabio) and black text is never changed.  Each replacement must
match exactly once, or the script stops before writing anything.  --check reports without writing.

Numbers that only the finished run can give ({calls}, {cost}, ...) are read from the JSON written by
run2_numbers.py; without it the placeholders stay and the script refuses to write.
"""
import argparse, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--paper-root", required=True)
ap.add_argument("--numbers", default="")
ap.add_argument("--check", action="store_true")
a = ap.parse_args()
P = Path(a.paper_root).resolve()
N = json.load(open(a.numbers)) if a.numbers else {}

PROV_YELLOW = "\\yellow{Provisional: one question per prompt; batched rerun pending.} "
PROV_GREEN_A = "\\green{Provisional: one question per prompt; a second run at 50 questions per prompt is pending. "
PROV_GREEN_B = "\\green{Provisional: one question per prompt. "

RUN2_SETUP = ("The questions of one game share its report. We asked the bank, tail, mirror and continuous questions in prompts of "
              "at most 50 binary or 20 continuous questions, the limits of the Micropolis run, in horizon order for binary questions "
              "and by quantity for continuous ones. Each natural-conditional question had its own prompt, because turn 2 of a cell "
              "continues that exchange: it reveals one sentence and asks the question again. A no-news control on 100 cells continues "
              "the exchange with no new information (\\cref{app:worlds}).")

# (file, old, new); `new` may hold {placeholders} filled from the numbers file
R = [
    # Section 2: protocol sentence
    ("sections/03_benchmark_design.tex",
     "\\yellow{FreeCiv is still to be rerun at 50 per prompt; its numbers here come from a one-question-per-prompt run and are provisional.}",
     "\\green{FreeCiv uses the same limits for its unconditional sets, 50 binary and 20 continuous questions per prompt. Its natural conditionals keep one question per prompt, since the second turn continues the conversation about that question.}"),
    # Section 4: setup and the provisional sentence
    ("sections/05_freeciv.tex",
     "Each question went in its own prompt with no persona; turn 2 of a cell continues the exchange with one revealed sentence and asks again, and a no-news control on 100 cells continues it with no new information (\\cref{app:worlds}). \\yellow{Every FreeCiv result below is provisional until the rerun at 50 questions per prompt.}",
     RUN2_SETUP),
    ("sections/05_freeciv.tex", "\\caption{\\green{Provisional: one question per prompt. FreeCiv scores against", "\\caption{\\green{FreeCiv scores against"),
    ("sections/05_freeciv.tex", "% \\caption{\\green{Provisional: one question per prompt; a second run at 50 questions per prompt is pending. Reliability", "% \\caption{\\green{Reliability"),
    # Appendix A: elicitation paragraph and the run record
    ("appendix/A_worlds_and_protocol.tex",
     "\\textbf{Elicitation.}\nWe send every prompt through OpenRouter with the report before the question.",
     "\\textbf{Elicitation.}\nWe send every prompt through OpenRouter with the report before the questions."),
    ("appendix/A_worlds_and_protocol.tex",
     "A turn-1 prompt is 7,565 to 8,544 tokens. There is no system message and no persona. The prompt says a proper scoring rule will evaluate the answer, names none, gives no example, and asks for the probability inside a tag, or the five percentiles in a labelled block. Every request sets an output limit of 16,384 tokens (24,000 for DeepSeek V4 Flash, 6,000 for Kimi K2) and provider-default sampling.",
     "A prompt with several questions has {prompt_tokens_lo} to {prompt_tokens_hi} tokens; a natural-conditional prompt has 7,565 to 8,544. There is no system message and no persona. The prompt says a proper scoring rule will evaluate the answers and names none. It asks for one line per question in a delimited block, a probability or the five percentiles; a natural-conditional prompt asks for the probability inside a tag. Every request sets an output limit of 32,768 tokens and provider-default sampling."),
    ("appendix/A_worlds_and_protocol.tex",
     "We pin proprietary models to their first-party host and open-weight models to one host each (\\cref{tab:hosting}).",
     "We pin every model to the host that Micropolis used (\\cref{tab:hosting}), so the two worlds queried the same endpoints."),
    ("appendix/A_worlds_and_protocol.tex",
     "The scored rows cost \\$393.83 across the 24 models. A check at medium effort on 7 models and 200 bank items cost a further \\$34 (\\cref{app:ablation-effort}). All FreeCiv results are provisional pending the rerun at 50 questions per prompt.",
     "The scored rows cost \\$393.83 across the 24 models. A check at medium effort on 7 models and 200 bank items cost a further \\$34 (\\cref{app:ablation-effort}). That run asked one question per prompt.\n\nThe run of 20 September 2026 is the one the paper reports. It asked the bank, tail, mirror and continuous questions in groups, with the limits of the Micropolis run: {n_bin_prompts} binary prompts of {bin_lo} to {bin_hi} questions and {n_cont_prompts} continuous prompts of {cont_lo} to {cont_hi}, one game per prompt and {n_prompts} prompts per model. We sent each prompt once, and an answer we could not read stays missing. Every model answered at least {parse_min} percent of its questions, and {parse_mean} percent on average. {parse_note} The natural-conditional arm kept the protocol of 9 September, one question per prompt. We reuse its forecasts for the 22 models whose host did not change. DeepSeek V3 and DeepSeek V4 Flash moved to the hosts that Micropolis used and answered them again. DeepSeek V4 Flash ran at effort low, as in the other two worlds; the setting has no effect on it, and it returned about {dsv4_reasoning} reasoning tokens per grouped call. The grouped calls cost \\${cost_batched} and the two natural-conditional reruns \\${cost_natcond}."),
    ("appendix/A_worlds_and_protocol.tex",
     "Needs (Jaeho): (1) the Freeciv server version and ruleset of the container; (2) one-sentence definitions of the five natural-conditional groups A, B, C1, C2 and D; (3) the decision on the DeepSeek V4 Flash regime (effort low, which the model ignores, or reasoning disabled), stated here and told to Fabio and Nick; (4) after the rerun at 50 questions per prompt, the calls per model, the cost, the parse rates and the turn-2 design when 50 questions share one prompt.",
     "Needs (Jaeho): the Freeciv server version and ruleset of the container. Done on 20 September 2026: the group definitions (Question sets, above), the DeepSeek V4 Flash regime (effort low, as in Micropolis and StarSim) and the record of the grouped run."),
    ("appendix/A_worlds_and_protocol.tex", "About: the items above marked in dark yellow, and the version of this subsection for the batched rerun.", "About: the one item above still marked in dark yellow."),
    # protocol table
    ("appendix/A_worlds_and_protocol.tex", " & \\green{Micropolis} & \\green{FreeCiv (provisional)} & \\green{StarSim} \\\\", " & \\green{Micropolis} & \\green{FreeCiv} & \\green{StarSim} \\\\"),
    ("appendix/A_worlds_and_protocol.tex", "\\green{Questions per prompt} & \\blue{\\MPDPerPromptBinary\\ binary, \\MPDPerPromptContinuous\\ continuous} & \\green{1} \\yellow{(50 in the rerun)} & \\green{1} \\\\",
     "\\green{Questions per prompt} & \\blue{\\MPDPerPromptBinary\\ binary, \\MPDPerPromptContinuous\\ continuous} & \\green{50 binary, 20 continuous; 1 for natural conditionals} & \\green{1} \\\\"),
    # Appendix C
    ("appendix/C_full_results.tex", "\\subsection{FreeCiv (provisional)}", "\\subsection{FreeCiv}"),
    ("appendix/C_full_results.tex",
     "\\yellow{Provisional: every FreeCiv number in this appendix comes from the run with one question per prompt of 9 September 2026 and will be replaced by the rerun at 50 questions per prompt (Jaeho).}\n",
     "\\green{Every FreeCiv number in this appendix comes from the run of 20 September 2026 (\\cref{app:worlds}).}\n"),
    ("appendix/C_full_results.tex", "\\caption{\\yellow{Provisional: one question per prompt; batched rerun pending.} \\green{FreeCiv scores per model", "\\caption{\\green{FreeCiv scores per model"),
    ("appendix/C_full_results.tex", "\\caption{\\yellow{Provisional: one question per prompt; batched rerun pending.} \\green{FreeCiv mirror set", "\\caption{\\green{FreeCiv mirror set"),
    ("appendix/C_full_results.tex", "\\caption{\\yellow{Provisional: one question per prompt; batched rerun pending.} \\green{FreeCiv bank excess Brier", "\\caption{\\green{FreeCiv bank excess Brier"),
    ("appendix/C_full_results.tex", "\\caption{\\yellow{Provisional: one question per prompt; batched rerun pending.} \\green{FreeCiv continuous nCRPS", "\\caption{\\green{FreeCiv continuous nCRPS"),
    ("appendix/C_full_results.tex", "\\caption{\\yellow{Provisional: one question per prompt; batched rerun pending.} \\green{FreeCiv natural conditionals per model", "\\caption{\\green{FreeCiv natural conditionals per model"),
    ("appendix/C_full_results.tex", "\\caption{\\green{Provisional: one question per prompt; a second run at 50 questions per prompt is pending. FreeCiv scores by resolution turn", "\\caption{\\green{FreeCiv scores by resolution turn"),
    ("appendix/C_full_results.tex", "\\caption{\\green{Provisional: one question per prompt; a second run at 50 questions per prompt is pending. Natural conditionals: how forecasts move", "\\caption{\\green{Natural conditionals: how forecasts move"),
    ("appendix/C_full_results.tex", "\\caption{\\yellow{Provisional: one question per prompt; batched rerun pending.} \\green{Reliability of FreeCiv bank forecasts", "\\caption{\\green{Reliability of FreeCiv bank forecasts"),
    ("appendix/C_full_results.tex", "\\caption{\\yellow{Provisional: one question per prompt; batched rerun pending.} \\green{FreeCiv difficulty by question family", "\\caption{\\green{FreeCiv difficulty by question family"),
    ("appendix/C_full_results.tex", " \\yellow{The FreeCiv rows are provisional (one question per prompt; batched rerun pending).}", ""),
    ("appendix/C_full_results.tex", " The FreeCiv rows are provisional (one question per prompt; batched rerun pending).", ""),
    # Appendix D
    ("appendix/D_ablations.tex", "\\subsection{Reasoning effort (FreeCiv, provisional)}", "\\subsection{Reasoning effort (FreeCiv)}"),
    ("appendix/D_ablations.tex", "The main FreeCiv run set every model to its lowest reasoning effort (\\cref{sec:freeciv}).", "The FreeCiv run of 9 September 2026, one question per prompt, set every model to its lowest reasoning effort (\\cref{sec:freeciv})."),
    ("appendix/D_ablations.tex", "The per-question rows of the check are not on this machine; the table should be regenerated from them with reasoning tokens and parse rates (Jaeho). Provisional: one question per prompt.}", "The per-question rows of the check are not on this machine; the table should be regenerated from them with reasoning tokens and parse rates (Jaeho). The check was made on the run of 9 September, one question per prompt, and not repeated in the batched run.}"),
    ("appendix/D_ablations.tex", "\\subsection{Answering again with nothing revealed (FreeCiv, provisional)}", "\\subsection{Answering again with nothing revealed (FreeCiv)}"),
    ("appendix/D_ablations.tex", " \\yellow{Provisional: one question per prompt.}}", "}"),
]

def fill(s):
    if "{" not in s or "\\cref" in s and not re.search(r"\{[a-z_]+\}", s):
        return s
    for k, v in N.items():
        s = s.replace("{" + k + "}", str(v))
    return s

problems, edits = [], {}
for f, old, new in R:
    path = P / f
    text = edits.get(path, path.read_text())
    c = text.count(old)
    if c != 1:
        problems.append(f"{f}: expected 1 match, found {c}: {old[:90]!r}")
        continue
    new_f = fill(new)
    if re.search(r"\{[a-z_]+\}", new_f.replace("\\cref{app:worlds}", "").replace("\\cref{sec:freeciv}", "").replace("\\cref{app:ablation-effort}", "").replace("\\cref{tab:hosting}", "")):
        problems.append(f"{f}: unfilled placeholder in replacement: {new_f[:100]!r}")
    edits[path] = text.replace(old, new_f)
if problems:
    print("NOT WRITTEN:\n  " + "\n  ".join(problems))
    sys.exit(1)
for path, text in edits.items():
    if a.check:
        print(f"would edit {path.relative_to(P)}")
    else:
        path.write_text(text)
        print(f"edited     {path.relative_to(P)}")
print(f"{len(R)} replacements over {len(edits)} files" + (" (check only)" if a.check else ""))
