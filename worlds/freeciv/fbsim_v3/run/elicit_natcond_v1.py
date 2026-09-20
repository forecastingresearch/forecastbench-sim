#!/usr/bin/env python3
"""elicit_natcond_v1.py — the natural-conditional arm exactly as run 1 asked it, for the re-pinned models.

  python elicit_natcond_v1.py --out DIR --models "DeepSeek: DeepSeek V3,DeepSeek: DeepSeek V4 Flash 0731"
                              [--models-file models_v2.csv] [--workers W] [--limit N] [--dry-run] [--env-file PATH]

Run 2 keeps the natural conditionals unbatched: one question per prompt at turn 1, the uniform reveal sentence
and the question again at turn 2, and the no-news control on 100 cells.  The prompt texts, message structure,
parsing and row format are those of elicit_v1.py (run of 9 September), copied verbatim, so the rows for a
re-pinned model are comparable with the run-1 rows reused for the other 22 (rebuild_natcond_rows.py).  Only the
request's provider block differs: it comes from models_v2.csv through elicit_v2.build_body, so the pins match
the Micropolis registry.  Per model: 355 turn-1 calls (276 bank questions and 124 extra value questions used
only by cells), 400 turn-2 calls, 99 no-news calls.  Turn-1 rows carry arm "t1nc" (see score_v2.py).
"""
import argparse, collections, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elicit_v2 import REPORTS, SETS, FIRST_PARTY, call, load_models, spent  # noqa: E402

# ---- run-1 texts, verbatim (elicit_v1.py v1.8) ----
BASE_PREFIX = """You will be asked a binary forecasting question about a simulation in progress. Your task is to estimate the probability that the event described in the question occurs. Please provide your reasoning before stating your final probability.

Question Background: This is a partial report on a FreeCiv game simulation in progress, observed at turn 60. Five AI civilizations are competing.

{report}
"""
BASE_TAIL = """
Question Title: {question}
{fact}
Resolution Criteria: {criteria}

Think step by step about the information provided, reason about uncertainty, and put your final confidence that the question resolves YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

Your answer will be evaluated with a proper scoring rule, so your best strategy is to report your honest probability estimate.

Your final answer should be the probability that the event resolves YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""
TURN2 = """{preamble}

Given this, please answer the same question again. Reason about what, if anything, this changes, and end your response with your updated probability in <probability> </probability> tags."""
NO_NEWS = "The game has continued. No new information about turns 61 through 90 is available. If you wish, revise your forecast."
TURN2_NONEWS = NO_NEWS + "\n\nPlease answer the same question again and end your response with your probability in <probability> </probability> tags."

_rep = {}


def report(world):
    if world not in _rep:
        _rep[world] = (REPORTS / world / "turn_060_report.txt").read_text()
    return _rep[world]


def make_prompt(world, question, criteria, fact=""):
    return BASE_PREFIX.format(report=report(world)), BASE_TAIL.format(question=question, fact=fact, criteria=criteria)


def user_content(prefix, tail, model_id):
    if model_id.startswith(FIRST_PARTY):
        return [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": tail}]
    return prefix + tail


def _prob(v):
    v = float(v)
    v = v / 100 if v > 1 else v
    return v if 0 <= v <= 1 else None


def parse_prob(t):
    """run-1 parser: strict = the last <probability> tag; lenient = 'probability ... 0.xx', a bare '<0.3>' or a final bare number."""
    m = re.findall(r"<probability>\s*([0-9]*\.?[0-9]+)\s*%?\s*</probability>", t or "")
    if m:
        return _prob(m[-1]), "strict"
    tail = (t or "")[-400:]
    t = re.sub(r"(?i)\s*</?probability>\s*$", "", tail.strip()).rstrip(".* ")
    m = (re.findall(r"<\s*([0-9]*\.?[0-9]+)\s*%?\s*>\s*$", t)
         or re.findall(r"(?i)probability[^0-9]{0,40}([0-9]*\.?[0-9]+)\s*%?", t)
         or re.findall(r"(?:^|[\s:=~≈])(0?\.[0-9]+|1\.0+|1|0)\s*$", t))
    v = _prob(m[-1]) if m else None
    return v, ("lenient" if v is not None else None)


def build_jobs(limit=0):
    nc = json.load(open(SETS / "natcond_600.json"))
    bank = {i["id"]: i for i in json.load(open(SETS / "bank_750.json"))}
    extra = {i["id"]: i for i in json.load(open(SETS / "natcond_extra_turn1.json"))}
    t1, seen, t2, nonews, nn_seen = [], set(), [], [], set()
    for c in nc:
        q = bank.get(c["qid"]) or extra[c["qid"]]
        if q["id"] not in seen:
            seen.add(q["id"])
            pre, tail = make_prompt(q["world"], q["text"], q["criteria"])
            t1.append(dict(item=q["id"], kind="bin", world=q["world"], set="bank" if c["from_bank"] else "extra", prefix=pre, tail=tail))
        t2.append(dict(item=c["qid"] + "|" + c["rev_id"], kind="bin", world=q["world"], qid=q["id"], block=c["block"], followup=TURN2.format(preamble=c["turn2_preamble"])))
        if c["control_no_news"] and q["id"] not in nn_seen:
            nn_seen.add(q["id"])
            nonews.append(dict(item=q["id"], kind="bin", world=q["world"], qid=q["id"], followup=TURN2_NONEWS))
    if limit:
        keep = {j["item"] for j in t1[:limit]}
        t1 = t1[:limit]
        t2 = [j for j in t2 if j["qid"] in keep]
        nonews = [j for j in nonews if j["qid"] in keep]
    t1.sort(key=lambda j: j["world"]); t2.sort(key=lambda j: j["world"]); nonews.sort(key=lambda j: j["world"])
    return dict(t1=t1, t2=t2, nonews=nonews)


lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--models-file", default=str(Path(__file__).resolve().parent / "models_v2.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N turn-1 questions and their cells")
    ap.add_argument("--reask", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--env-file", default="")
    a = ap.parse_args()
    import elicit_v2
    elicit_v2.KEY = os.environ.get("OPENROUTER_API_KEY")
    if not elicit_v2.KEY and a.env_file:
        for l in open(os.path.expanduser(a.env_file)):
            if l.startswith("OPENROUTER_API_KEY="):
                elicit_v2.KEY = l.split("=", 1)[1].strip().strip("\"'")
    want = a.models.split(",")
    models = [m for m in load_models(a.models_file) if m["name"] in want or m["openrouter_id"] in want]
    missing = [w for w in want if not any(m["name"] == w or m["openrouter_id"] == w for m in models)]
    if missing:
        sys.exit(f"not in models file: {missing}")
    jobs = build_jobs(a.limit)
    print(f"jobs per model: t1={len(jobs['t1'])} t2={len(jobs['t2'])} nonews={len(jobs['nonews'])} total={sum(len(v) for v in jobs.values())}; models={[m['name'] for m in models]}", flush=True)
    if a.dry_run:
        return
    if not elicit_v2.KEY:
        sys.exit("OPENROUTER_API_KEY missing (export it or pass --env-file)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = out / "results.jsonl"; done = {}
    if ck.exists():
        for line in open(ck):
            try:
                r = json.loads(line); done[(r["model"], r["item"], r["arm"])] = r
            except Exception:
                pass
    f = open(ck, "a")

    def emit(row):
        with lock:
            f.write(json.dumps(row) + "\n"); f.flush()
        done[(row["model"], row["item"], row["arm"])] = row

    for model in models:
        mid = model["openrouter_id"]; t0 = time.time()
        W = a.workers or int((model.get("workers") or "").strip() or 16)

        def run(job, arm, msgs_fn, extra):
            key = (mid, job["item"], arm)
            if key in done and done[key].get("value") is not None:
                return done[key]
            row = None
            for attempt in range(1 + a.reask):
                r = call(model, msgs_fn())
                value, pmode = parse_prob(r["text"])
                row = dict(model=mid, item=job["item"], arm=arm, kind="bin", world=job["world"], reasoning="low", value=value, parse_mode=pmode,
                           attempts=attempt + 1, ts=time.time(), source="run2:natcond_v1", **extra, **{k: v for k, v in r.items() if k != "usage_raw"})
                if value is not None or r.get("error"):
                    break
            emit(row)
            print(f"  [{time.strftime('%H:%M:%S')}] {mid} {arm} {job['item']} {row.get('secs')}s out={row.get('tokens_out')} reas={row.get('tokens_reasoning')} value={value} mode={pmode} finish={row.get('finish')} err={row.get('error')}", flush=True)
            return row

        t1_by = {j["item"]: j for j in jobs["t1"]}
        t1_msgs = lambda j: (lambda: [{"role": "user", "content": user_content(j["prefix"], j["tail"], mid)}])
        with ThreadPoolExecutor(W) as ex:
            list(ex.map(lambda j: run(j, "t1nc", t1_msgs(j), {}), jobs["t1"]))

        def follow_msgs(job):
            t1 = done.get((mid, job["qid"], "t1nc")); j1 = t1_by[job["qid"]]
            return lambda: [{"role": "user", "content": user_content(j1["prefix"], j1["tail"], mid)}, {"role": "assistant", "content": t1["text"]}, {"role": "user", "content": job["followup"]}]

        with ThreadPoolExecutor(W) as ex:
            for arm in ("t2", "nonews"):
                ready = [j for j in jobs[arm] if (done.get((mid, j["qid"], "t1nc")) or {}).get("text")]
                list(ex.map(lambda j: run(j, arm, follow_msgs(j), dict(t1_value=(done.get((mid, j["qid"], "t1nc")) or {}).get("value"))), ready))
        rows = [r for (m, _, _), r in done.items() if m == mid]
        print(f"{model['name']} ({mid}): {len(rows)} rows, {sum(1 for r in rows if r.get('value') is None)} unparsed/errors, ${spent[mid]:.2f} this session, {time.time() - t0:.0f}s, workers={W}", flush=True)
    json.dump(dict(spent), open(out / "spend_session.json", "w"), indent=1)
    print("done; rows in", ck)


if __name__ == "__main__":
    main()
