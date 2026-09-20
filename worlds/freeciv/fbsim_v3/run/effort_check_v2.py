#!/usr/bin/env python3
"""effort_check_v2.py --out DIR [--env-file F] [--models a,b] [--dry-run] [--smoke] [--workers N]

The reasoning-effort check of Appendix D repeated on the grouped prompts (Jaeho, 20 September 2026): the same 200 bank
questions as the check of 9 September (elicit_v1's stratified sample, `--sample bank=200`, seed 11), asked in grouped
prompts of the run-2 kind (one game per prompt, horizon-major order, at most 50 questions, the run-2 wording and answer
block, each prompt sent once) to the same seven models at the next reasoning level: effort medium where the provider
exposes a level, a 2,048-token budget for the budget models (LiteLLM's mapping, as in the check of 9 September).
The low-effort side of the comparison is the main run itself (results/run2_paper), which already holds these questions.

Output: DIR/results.jsonl (one row per model x question, arm t1, the elicit_v2 row layout plus `effort_level`),
DIR/calls.jsonl (one record per prompt), DIR/items.json (the 200 question ids).  Resumable: complete calls are skipped.
"""
import argparse, collections, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import elicit_v2 as ev  # noqa: E402

SEVEN = ["anthropic/claude-fable-5", "anthropic/claude-sonnet-5", "qwen/qwen3-235b-a22b", "openai/gpt-5", "anthropic/claude-haiku-4.5",
         "openai/o3", "openai/gpt-5.6-luna"]


def sample_ids(per_horizon=40, seed=11):
    """The rule of the check of 9 September as Appendix D states it: 40 bank questions per horizon, drawn in turn from each
    family (families in name order, each family's questions shuffled once by Random(seed)).  The exact list of 9 September is
    in the archive and not on this machine, so this is the same rule, not necessarily the same questions."""
    import random
    rng = random.Random(seed)
    items = json.load(open(ev.SETS / "bank_750.json"))
    out = []
    for T in sorted({i["T"] for i in items}):
        groups = collections.defaultdict(list)
        for it in items:
            if it["T"] == T:
                groups[it["family"]].append(it["id"])
        for k in sorted(groups):
            rng.shuffle(groups[k])
        picked, keys = [], sorted(groups)
        while len(picked) < per_horizon and any(groups[k] for k in keys):
            for k in keys:
                if groups[k] and len(picked) < per_horizon:
                    picked.append(groups[k].pop())
        out += picked
    return sorted(out)


def next_level(model):
    m = dict(model)
    mode = (m.get("reasoning_mode") or "none").strip()
    if mode == "effort":
        m["effort"] = "medium"
    elif mode == "budget":
        m["budget"] = "2048"
    else:
        raise SystemExit(f"{m['openrouter_id']} has no reasoning control")
    return m


def build_prompts(ids):
    S = {i["id"]: dict(i, set="bank") for i in json.load(open(ev.SETS / "bank_750.json"))}
    by = collections.defaultdict(list)
    for i in ids:
        by[S[i]["world"]].append(S[i])
    batches = []
    for world in sorted(by):
        items = sorted(by[world], key=lambda i: (i["T"], i["family"], i["id"]))
        chunks = ev.even_split(items, 50)
        for ci, chunk in enumerate(chunks, 1):
            pre, tail = ev.make_prompt("bin", world, chunk)
            batches.append(dict(id=f"{world}_bin_effort_c{ci}of{len(chunks)}", world=world, kind="bin", items=chunk, prefix=pre, tail=tail))
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-file", default="")
    ap.add_argument("--models", default=",".join(SEVEN))
    ap.add_argument("--models-file", default=str(HERE / "models_v2.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one prompt of the first model only")
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and a.env_file:
        for l in open(os.path.expanduser(a.env_file)):
            if l.startswith("OPENROUTER_API_KEY="):
                key = l.split("=", 1)[1].strip().strip("\"'")
    ev.KEY = key
    ids = sample_ids()
    batches = build_prompts(ids)
    want = a.models.split(",")
    models = [next_level(m) for m in ev.load_models(a.models_file) if m["openrouter_id"] in want]
    assert len(models) == len(want), [m["openrouter_id"] for m in models]
    S = {i["id"]: i for i in json.load(open(ev.SETS / "bank_750.json"))}
    byT = collections.Counter(S[i]["T"] for i in ids)
    print(f"{len(ids)} bank questions (per horizon {dict(sorted(byT.items()))}); {len(batches)} prompts of {[len(b['items']) for b in batches]} questions; "
          f"models: {[(m['openrouter_id'], ev.reasoning_param(m)) for m in models]}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(ids, open(out / "items.json", "w"))
    if a.dry_run:
        b = batches[0]
        print(json.dumps(ev.build_body(models[0], [{"role": "user", "content": ev.user_content(b["prefix"][:300] + " ...", b["tail"][:800], models[0]["openrouter_id"])}]), indent=1)[:1500])
        return
    if not key:
        sys.exit("no OPENROUTER_API_KEY")
    if a.smoke:
        models, batches = models[:1], batches[:1]
    calls_f, rows_f = out / "calls.jsonl", out / "results.jsonl"
    done = {}
    if calls_f.exists():
        for line in open(calls_f):
            try:
                c = json.loads(line); done[(c["model"], c["call_id"])] = c
            except Exception:
                pass
    lock = ev.lock
    calls_out, rows_out = open(calls_f, "a"), open(rows_f, "a")

    def run_call(model, b):
        mid = model["openrouter_id"]; call_id = f"t1:{b['id']}"
        if done.get((mid, call_id), {}).get("complete"):
            return
        msgs = [{"role": "user", "content": ev.user_content(b["prefix"], b["tail"], mid)}]
        asked = {k: i["id"] for k, i in enumerate(b["items"], 1)}
        r = ev.call(model, msgs)
        parsed, mode = ev.parse_block(r["text"], set(asked), "bin")
        n = len(asked)
        rec = dict(model=mid, call_id=call_id, arm="t1", kind="bin", n_asked=n, complete=len(parsed) == n, attempts=1, parse_mode=mode, ts=time.time(),
                   messages=msgs, effort_level="next", **r, world=b["world"], batch_id=b["id"], n_parsed=len(parsed))
        rows = [dict(model=mid, item=item, arm="t1", kind="bin", world=b["world"], value=parsed.get(k), parse_mode=mode if parsed.get(k) is not None else None,
                     batch=b["id"], call_id=call_id, position=k, n_in_call=n, attempts=1, ts=rec["ts"], cost_share=(r.get("cost") or 0) / n,
                     tokens_in=r.get("tokens_in"), tokens_out=r.get("tokens_out"), tokens_reasoning=r.get("tokens_reasoning"), tokens_cached=r.get("tokens_cached"),
                     provider=r.get("provider"), finish=r.get("finish"), error=r.get("error"), effort_level="next", reasoning_param=r.get("reasoning_param"))
                for k, item in asked.items()]
        with lock:
            calls_out.write(json.dumps(rec) + "\n"); calls_out.flush()
            for row in rows:
                rows_out.write(json.dumps(row) + "\n")
            rows_out.flush()
        print(f"  [{time.strftime('%H:%M:%S')}] {mid} {call_id} {r.get('secs')}s out={r.get('tokens_out')} reas={r.get('tokens_reasoning')} cost=${r.get('cost') or 0:.4f} "
              f"parsed={len(parsed)}/{n} finish={r.get('finish')} err={r.get('error')}", flush=True)

    for model in models:
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda b: run_call(model, b), batches))
        print(f"{model['openrouter_id']}: ${ev.spent[model['openrouter_id']]:.2f}, {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(ev.spent), open(out / "spend_session.json", "w"), indent=1)
    print("done", out)


if __name__ == "__main__":
    main()
