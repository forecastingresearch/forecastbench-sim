#!/usr/bin/env python3
"""smoke_report_v2.py DIR [DIR ...] — health table for a batched run (elicit_v2.py) or a natcond run (elicit_natcond_v1.py).

Per model: calls, calls complete (every asked question parsed), parse modes, finish reasons, provider served, mean
prompt / cached / completion / reasoning tokens, cost per call, seconds per call, and a projection of the full run's
cost from the per-question cost.  Reads calls.jsonl where present (batched runs), else results.jsonl rows.
"""
import collections, glob, json, os, statistics as st, sys

FULL_BATCHED_CALLS = 34                  # turn-1 prompts per model in run 2
FULL_QUESTIONS = 1400
FULL_NATCOND_CALLS = 854


def load(dirs, name):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", name), recursive=True)):
            for line in open(f):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def mean(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else float("nan")


def main():
    dirs = sys.argv[1:]
    calls = load(dirs, "calls.jsonl")
    rows = load(dirs, "results.jsonl")
    if calls:   # batched
        by = collections.defaultdict(list)
        for c in calls:
            by[c["model"]].append(c)
        print(f"{'model':38s} {'calls':>5s} {'compl':>5s} {'parse':>8s} {'finish':>10s} {'provider':>16s} {'in':>6s} {'cached':>6s} {'out':>6s} {'reas':>6s} {'$/call':>7s} {'s/call':>6s} {'$/q':>7s} {'full$':>6s}")
        tot_full = 0
        for m, cs in sorted(by.items(), key=lambda kv: -mean([c.get("cost") for c in kv[1]])):
            n = len(cs)
            compl = sum(1 for c in cs if c.get("complete"))
            pm = collections.Counter(c.get("parse_mode") for c in cs)
            fin = collections.Counter(c.get("finish") for c in cs)
            prov = collections.Counter(c.get("provider") for c in cs)
            nq = sum(c.get("n_asked", 0) for c in cs)
            cost = sum(c.get("cost") or 0 for c in cs)
            per_q = cost / nq if nq else float("nan")
            full = per_q * FULL_QUESTIONS
            tot_full += full
            print(f"{m:38s} {n:5d} {compl:5d} {'/'.join(f'{k}:{v}' for k, v in pm.items()):>8s} {'/'.join(f'{k}:{v}' for k, v in fin.items()):>10s} "
                  f"{'/'.join(str(k) for k in prov):>16s} {mean([c.get('tokens_in') for c in cs]):6.0f} {mean([c.get('tokens_cached') for c in cs]):6.0f} "
                  f"{mean([c.get('tokens_out') for c in cs]):6.0f} {mean([c.get('tokens_reasoning') for c in cs]):6.0f} {cost / n:7.3f} {mean([c.get('secs') for c in cs]):6.0f} {per_q:7.4f} {full:6.2f}")
        unparsed = [(r["model"], r["item"], r["call_id"]) for r in rows if r.get("value") is None]
        print(f"\nrows: {len(rows)}, unparsed: {len(unparsed)}; spent this smoke: ${sum(c.get('cost') or 0 for c in calls):.2f}; "
              f"projected full batched run (1,400 questions per model at the smoke's cost per question): ${tot_full:.0f}")
        errs = [(c["model"], c.get("error")) for c in calls if c.get("error")]
        if errs:
            print("errors:", errs)
        if unparsed:
            print("unparsed rows (first 12):", unparsed[:12])
    if rows and not calls:  # natcond (v1 format)
        by = collections.defaultdict(list)
        for r in rows:
            by[r["model"]].append(r)
        print(f"{'model':38s} {'rows':>5s} {'t1nc/t2/nn':>12s} {'unparsed':>8s} {'parse':>14s} {'finish':>10s} {'provider':>14s} {'in':>6s} {'out':>6s} {'reas':>6s} {'$/call':>7s} {'s/call':>6s} {'full$':>6s}")
        for m, rs in by.items():
            arms = collections.Counter(r["arm"] for r in rs)
            pm = collections.Counter(r.get("parse_mode") for r in rs)
            fin = collections.Counter(r.get("finish") for r in rs)
            prov = collections.Counter(r.get("provider") for r in rs)
            cost = sum(r.get("cost") or 0 for r in rs)
            print(f"{m:38s} {len(rs):5d} {arms.get('t1nc', 0):>3d}/{arms.get('t2', 0):>3d}/{arms.get('nonews', 0):>3d} {sum(1 for r in rs if r.get('value') is None):8d} "
                  f"{'/'.join(f'{k}:{v}' for k, v in pm.items()):>14s} {'/'.join(f'{k}:{v}' for k, v in fin.items()):>10s} {'/'.join(str(k) for k in prov):>14s} "
                  f"{mean([r.get('tokens_in') for r in rs]):6.0f} {mean([r.get('tokens_out') for r in rs]):6.0f} {mean([r.get('tokens_reasoning') for r in rs]):6.0f} "
                  f"{cost / len(rs):7.4f} {mean([r.get('secs') for r in rs]):6.0f} {cost / len(rs) * FULL_NATCOND_CALLS:6.2f}")
        errs = [(r["model"], r["item"], r.get("error")) for r in rows if r.get("error")]
        if errs:
            print("errors:", errs[:10])


if __name__ == "__main__":
    main()
