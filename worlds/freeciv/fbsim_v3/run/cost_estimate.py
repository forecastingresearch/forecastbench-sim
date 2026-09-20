#!/usr/bin/env python3
"""cost_estimate.py — token counts of every prompt in the study and a per-model cost estimate.

  uv run python cost_estimate.py [--models-file models_v1.csv] [--vis 500] [--reason-low 1200] [--nonreason-out 700] [--high-mult 2] [--live]

Input tokens are counted exactly (tiktoken o200k_base; other tokenizers differ by roughly +-15%). Output tokens are an
assumption: a reasoning model at low effort = REASON_LOW hidden + VIS visible; a non-reasoning model = NONREASON_OUT.
Turn-2 input = the turn-1 prompt + the visible turn-1 answer (VIS) + the follow-up message. No prompt caching is assumed
(the question precedes the report in the verbatim prompt, so provider prefix caches do not hit). --live refreshes prices
from the OpenRouter models endpoint instead of the models file.
"""
import argparse, csv, json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elicit_v1 import build_jobs, load_models
import tiktoken
enc = tiktoken.get_encoding('o200k_base')
def n(s): return len(enc.encode(s))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models-file', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_v1.csv'))
    ap.add_argument('--vis', type=int, default=500); ap.add_argument('--reason-low', type=int, default=1200); ap.add_argument('--nonreason-out', type=int, default=700)
    ap.add_argument('--high-mult', type=float, default=2.0); ap.add_argument('--cache', type=float, default=0.93, help='share of prefix tokens read from cache on providers with a prefix cache (Anthropic/OpenAI/Google); 0 = no caching'); ap.add_argument('--live', action='store_true'); ap.add_argument('--sets', default='bank,tails,mirrors,continuous,natcond'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    models = load_models(a.models_file)
    if a.live:
        live = {m['id']: m for m in json.load(urllib.request.urlopen('https://openrouter.ai/api/v1/models'))['data']}
        for m in models:
            p = live[m['openrouter_id']]['pricing']; m['in_per_M'] = float(p['prompt']) * 1e6; m['out_per_M'] = float(p['completion']) * 1e6
    J = build_jobs(a.sets, smoke=a.smoke)
    t1_tok = {j['item']: n(j['prompt']) for j in J['t1']}
    tok = dict(t1=sum(t1_tok.values()), single=sum(n(j['prompt']) for j in J['single']),
               t2=sum(t1_tok[j['qid']] + a.vis + n(j['followup']) for j in J['t2']), nonews=sum(t1_tok[j['qid']] + a.vis + n(j['followup']) for j in J['nonews']))
    calls = {k: len(v) for k, v in J.items()}
    print(f"calls per model: {calls}  total {sum(calls.values())}")
    print(f"input tokens per model (exact for t1/single; t2/nonews add {a.vis} visible t1 tokens): " + ', '.join(f'{k}={v:,}' for k, v in tok.items()) + f"  total {sum(tok.values()):,}")
    print(f"turn-1 prompt tokens: min {min(t1_tok.values()):,} mean {sum(t1_tok.values()) / len(t1_tok):,.0f} max {max(t1_tok.values()):,}")
    tot_in = sum(tok.values()); tot_calls = sum(calls.values())
    rows = []; grand = grand_hi = 0
    for m in models:
        reasoning = m['reasoning'].lower().startswith('y')
        out_per_call = (a.reason_low + a.vis) if reasoning else a.nonreason_out
        tot_out = out_per_call * tot_calls
        cached_frac = a.cache * 0.9 if any(m['openrouter_id'].startswith(x) for x in ('anthropic/', 'openai/', 'google/')) else 0.0   # prefix ~90% of a prompt
        cache_price = float(m.get('cache_read_per_M') or 0) or float(m['in_per_M']) * 0.1
        in_cost = tot_in * (1 - cached_frac) / 1e6 * float(m['in_per_M']) + tot_in * cached_frac / 1e6 * cache_price
        cost = in_cost + tot_out / 1e6 * float(m['out_per_M'])
        cost_hi = in_cost + tot_out * a.high_mult / 1e6 * float(m['out_per_M'])
        grand += cost; grand_hi += cost_hi
        rows.append(dict(name=m['name'], id=m['openrouter_id'], reasoning='low' if reasoning else 'none', in_M=tot_in / 1e6, out_M=tot_out / 1e6,
                         in_price=float(m['in_per_M']), out_price=float(m['out_per_M']), cost=cost, cost_hi=cost_hi))
    print(f"\n{'model':38s} {'openrouter id':40s} {'reason':6s} {'$in/M':>6s} {'$out/M':>7s} {'in Mtok':>8s} {'out Mtok':>8s} {'cost':>8s} {'high':>8s}")
    for r in rows: print(f"{r['name'][:38]:38s} {r['id']:40s} {r['reasoning']:6s} {r['in_price']:6.2f} {r['out_price']:7.2f} {r['in_M']:8.2f} {r['out_M']:8.2f} {r['cost']:8.2f} {r['cost_hi']:8.2f}")
    print(f"\nTOTAL over {len(rows)} models: ${grand:,.0f} central, ${grand_hi:,.0f} if output runs {a.high_mult:g}x the assumption")
    with open(os.path.join(os.path.dirname(a.models_file), 'cost_estimate.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__ == '__main__': main()
