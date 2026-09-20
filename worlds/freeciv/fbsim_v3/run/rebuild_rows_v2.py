#!/usr/bin/env python3
"""rebuild_rows_v2.py BATCHED_DIR — regenerate results.jsonl (and the complete/parse_mode fields of calls.jsonl) for every
model folder from the stored calls, with the current parse_block of elicit_v2.py.  No API calls.  Rows are a pure function
of the stored responses, so a parser improvement is applied by re-reading, never by re-asking.  Prints per-model parse counts."""
import glob, gzip, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elicit_v2 as e

D = sys.argv[1]
sets = {}
for name, f in e.FILES.items():
    for i in json.load(open(e.SETS / f"{f}.json")):
        sets[i["id"]] = dict(i, set=name)
jobs, _ = e.build_batches("bank,tails,mirrors,continuous", 50, "grouped", 20)
batches = {b["id"]: b for b in jobs["t1"]}
tot = collections.Counter()
for d in sorted(glob.glob(f"{D}/*/")):
    if os.path.basename(d.rstrip("/")) == "logs" or not (os.path.exists(f"{d}/calls.jsonl") or os.path.exists(f"{d}/calls.jsonl.gz")):
        continue
    src = f"{d}/calls.jsonl" if os.path.exists(f"{d}/calls.jsonl") else f"{d}/calls.jsonl.gz"
    calls = [json.loads(l) for l in (gzip.open(src, "rt") if src.endswith(".gz") else open(src))]
    rows, new_calls, n_ok, n_q = [], [], 0, 0
    for c in calls:
        b = batches.get(c["call_id"].split(":", 1)[1])
        if b is None:
            raise SystemExit(f"{c['call_id']}: batch not in the current design (was the design changed?)")
        asked = {k: i["id"] for k, i in enumerate(b["items"], 1)}
        parsed, mode = e.parse_block(c["text"], set(asked), c["kind"])
        n = len(asked)
        c = dict(c, complete=len(parsed) == n, parse_mode=mode, n_parsed=len(parsed))
        new_calls.append(c)
        n_ok += len(parsed); n_q += n
        for k, item in asked.items():
            v = parsed.get(k)
            if c["kind"] == "cont" and v is not None:
                v = sorted(v)
            rows.append(dict(model=c["model"], item=item, arm="t1", kind=c["kind"], world=b["world"], value=v, parse_mode=mode if v is not None else None,
                             batch=b["id"], call_id=c["call_id"], position=k, n_in_call=n, attempts=c.get("attempts"), ts=c.get("ts"),
                             cost_share=(c.get("cost") or 0) / n, tokens_in=c.get("tokens_in"), tokens_out=c.get("tokens_out"),
                             tokens_reasoning=c.get("tokens_reasoning"), tokens_cached=c.get("tokens_cached"), provider=c.get("provider"),
                             finish=c.get("finish"), error=c.get("error")))
    with open(f"{d}/results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with (gzip.open(src, "wt") if src.endswith(".gz") else open(src, "w")) as f:
        for c in new_calls:
            f.write(json.dumps(c) + "\n")
    tot["calls"] += len(calls); tot["answers"] += n_ok; tot["questions"] += n_q
    print(f"  {os.path.basename(d.rstrip('/')):40s} calls {len(calls):3d}  answers parsed {n_ok:4d} of {n_q}  ({100 * n_ok / n_q:.1f}%)")
print(f"total: {tot['calls']} calls, {tot['answers']} of {tot['questions']} answers parsed ({100 * tot['answers'] / tot['questions']:.2f}%)")
