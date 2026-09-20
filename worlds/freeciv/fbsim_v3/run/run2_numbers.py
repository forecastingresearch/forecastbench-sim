#!/usr/bin/env python3
"""run2_numbers.py BATCHED_DIR NATCOND_REPINNED_DIR OUT_JSON — the run-2 facts the paper's protocol prose quotes.

From the stored calls of the batched arm and the rows of the natural-conditional reruns: prompt counts and sizes, prompt
token range, parse rates per model, DeepSeek V4 Flash's reasoning tokens, and the costs.  Written as a flat JSON that
paper/apply_run2_prose.py fills into its placeholders, so every number in that prose traces to this file.
"""
import collections, glob, json, statistics as st, sys

B, NC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
calls = []
for f in glob.glob(f"{B}/*/calls.jsonl"):
    for l in open(f):
        calls.append(json.loads(l))
by = collections.defaultdict(list)
for c in calls:
    by[c["model"]].append(c)
assert all(len(v) == 44 for v in by.values()) and len(by) == 24, {m: len(v) for m, v in by.items()}
one = next(iter(by.values()))
bins = sorted(c["n_asked"] for c in one if c["kind"] == "bin")
conts = sorted(c["n_asked"] for c in one if c["kind"] == "cont")
tok = [c["tokens_in"] for c in calls if c.get("tokens_in")]
parse = {m: sum(c.get("n_parsed", 0) for c in v) / sum(c["n_asked"] for c in v) for m, v in by.items()}
worst = min(parse, key=parse.get)
refusals = [(c["model"], c["call_id"]) for c in calls if (c.get("n_parsed", 0) == 0)]
nc_rows = [json.loads(l) for l in open(f"{NC}/results.jsonl")]
dsv4 = [c for c in by["deepseek/deepseek-v4-flash-0731"] if c.get("tokens_reasoning")]
N = dict(
    n_prompts=44, n_bin_prompts=len(bins), bin_lo=bins[0], bin_hi=bins[-1], n_cont_prompts=len(conts), cont_lo=conts[0], cont_hi=conts[-1],
    prompt_tokens_lo=f"{min(tok):,}", prompt_tokens_hi=f"{max(tok):,}",
    parse_min=f"{100 * min(parse.values()):.1f}", parse_mean=f"{100 * st.mean(parse.values()):.1f}", parse_worst_model=worst,
    parse_note=(f"{len(refusals)} prompts returned no readable answer at all ({', '.join(sorted({m.split('/')[1] for m, _ in refusals}))})."
                if refusals else "every prompt returned at least one readable answer."),
    dsv4_reasoning=f"{st.mean(c['tokens_reasoning'] for c in dsv4):,.0f}",
    cost_batched=f"{sum(c.get('cost') or 0 for c in calls):.2f}",
    cost_natcond=f"{sum(r.get('cost') or 0 for r in nc_rows):.2f}",
    natcond_rerun_rows=len(nc_rows), natcond_rerun_unparsed=sum(1 for r in nc_rows if r.get("value") is None),
    per_model_parse={m: round(100 * p, 2) for m, p in sorted(parse.items())},
    refusals=refusals,
)
json.dump(N, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in N.items() if k not in ("per_model_parse", "refusals")}, indent=1))
print("refusals:", refusals)
