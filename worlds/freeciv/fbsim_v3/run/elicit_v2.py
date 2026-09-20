#!/usr/bin/env python3
"""elicit_v2.py — batched elicitation over the draw v1 sets via OpenRouter (run 2, 2026-09-19).

Run 2 (Jaeho, 2026-09-19): bank, tails, mirrors and continuous are batched here; the natural-conditional arm stays
unbatched as in run 1 (elicit_natcond_v1.py for the re-pinned models, rebuild_natcond_rows.py for the rest).  The
grouped turn-2 code below is kept for a fully batched variant (--sets ...,natcond) but is not the run-2 design.

  python elicit_v2.py --out DIR [--models NAME_OR_ID,...] [--models-file models_v2.csv]
                      [--sets bank,tails,mirrors,continuous,natcond] [--per-prompt 50] [--t2 grouped|percell]
                      [--workers W] [--limit-batches N] [--smoke] [--dry-run] [--env-file PATH]

What changed from elicit_v1.py (one question per prompt): questions that share a world and an answer kind share
one prompt, up to --per-prompt of them, following Fabio's Micropolis batching:
  * grouping key = (world, kind): every binary question of a world (bank, tails, mirrors and the natural-
    conditional extra questions) forms one group; the continuous questions of a world form another;
  * a group larger than the cap is split into ceil(n / cap) consecutive chunks whose sizes differ by at most one
    (36 to 49 questions here), each its own prompt, cache key and retry unit;
  * inside a group the questions are in one random order fixed by seed 2026, so tail, bank and mirror items are
    mixed and a batch never consists of one set (a deliberate difference from Micropolis, whose corpus order
    already mixes templates).
The prompt keeps the run-1 wording: intro, "Question Background", the world report, then the numbered questions
each with its resolution criteria, then the instruction and an answer block modelled on Fabio's
(<<<PROBABILITIES>>> Q1: 0.65 ... <<<END>>>, or <<<PERCENTILES>>> Q1: p5=.., p25=.., p50=.., p75=.., p95=.. <<<END>>>).
The report prefix is one content part with a cache_control breakpoint on first-party hosts.

Natural conditionals (turn 2).  --t2 grouped (default): one continuation per (batch, revealed fact): the fact is
stated once and the model re-answers only the questions of that batch that have a cell with that fact, by number.
--t2 percell: one continuation per cell, re-asking one question, as in run 1.  The no-news control is one
continuation per batch re-asking its control questions.  Scoring is unchanged: one row per (model, item, arm).

Rows.  results.jsonl has one row per (model, item, arm) with `value`, as score_v1.py expects, plus the batch id,
the position in the batch and the call's usage divided by the number of questions asked (`cost_share`).  calls.jsonl
has one record per API call: the messages sent, the response text and reasoning, usage, cost, provider, timing.
Both are append-only and the run resumes from them.  A call whose answer block misses any asked question is
re-asked once (fresh call); the response that answers more questions is kept.

Provider routing follows Fabio's registry (models_v2.csv: endpoint, quantizations, ignore -> provider.only /
quantizations / ignore, allow_fallbacks false, require_parameters true).  Reasoning as in run 1 (lowest level
per model; DeepSeek V4 Flash at effort low, which it ignores, as in the other two worlds).

The API key comes from OPENROUTER_API_KEY or --env-file (a file outside this repository with KEY=value lines).
Nothing under this folder holds a credential.
"""
import argparse, collections, csv, hashlib, json, math, os, random, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
SETS = V3 / "sets" / "draw_v1"
REPORTS = V3 / "reports"
SEED = 2026
FILES = {"bank": "bank_750", "tails": "tails_300", "mirrors": "mirrors_50", "continuous": "continuous_300"}
FIRST_PARTY = ("anthropic/", "openai/", "google/")

# ---------------------------------------------------------------- prompts (run-1 wording, many questions)
BIN_PREFIX = """You will be asked several binary forecasting questions about a simulation in progress. For each question, your task is to estimate the probability that the event described in the question occurs. Please provide your reasoning before stating your final probabilities.

Question Background: This is a partial report on a FreeCiv game simulation in progress, observed at turn 60. Five AI civilizations are competing.

{report}
"""
BIN_TAIL = """
Questions ({n} in total):

{questions}
Think step by step about the information provided, reason about uncertainty, and then state your final confidence that each question resolves YES. Each probability should be a number between 0 and 1.

Your answers will be evaluated with a proper scoring rule, so your best strategy is to report your honest probability estimate for every question. You MUST provide a probability for every question.

Your response SHOULD STRICTLY END with your probabilities in this exact format, one line per question, in order:
<<<PROBABILITIES>>>
Q1: 0.65
Q2: 0.03
<<<END>>>
Provide one such line for each of the {n} questions, replacing the example values with your probability that the question resolves YES."""
CONT_PREFIX = """You will be asked several numeric forecasting questions about a simulation in progress. For each question, your task is to estimate the full range of plausible values for the quantity described, expressed as percentiles of your forecast distribution. Please provide your reasoning before stating your final percentiles.

Question Background: This is a partial report on a FreeCiv game simulation in progress, observed at turn 60. Five AI civilizations are competing.

{report}
"""
CONT_TAIL = """
Questions ({n} in total):

{questions}
Think step by step about the information provided, reason about uncertainty, and give your final answer to each question as five percentiles of your forecast distribution: p5, p25, p50, p75, p95. "p5" means you estimate a 5% chance the true value falls below that number; "p50" is your median. Your percentiles must be non-decreasing (p5 <= p25 <= p50 <= p75 <= p95). Every answer is a whole number.

Your answers will be evaluated with a proper scoring rule, so your best strategy is to report your honest estimate of each distribution - percentiles that are too narrow and percentiles that are too wide will both cost you. You MUST provide percentiles for every question.

Your response SHOULD STRICTLY END with your percentiles in this exact format, one line per question, in order:
<<<PERCENTILES>>>
Q1: p5=5, p25=10, p50=15, p75=20, p95=25
Q2: p5=100, p25=200, p50=300, p75=400, p95=500
<<<END>>>
Provide one such line for each of the {n} questions, replacing the example values with your actual percentile estimates."""
QUESTION = "Q{k}. Question Title: {question}\nResolution Criteria: {criteria}\n"
TURN2 = """{preamble}

Given this, please answer the following question{s} from the list above again: {qlist}. Reason about what, if anything, this changes, and end your response with your updated probabilit{ies} in this exact format:
<<<PROBABILITIES>>>
{example}
<<<END>>>"""
NO_NEWS = """The game has continued. No new information about turns 61 through 90 is available. If you wish, revise your forecast{s} for the following question{s} from the list above: {qlist}. Please answer {them} again and end your response with your probabilit{ies} in this exact format:
<<<PROBABILITIES>>>
{example}
<<<END>>>"""

_rep = {}


def report(world):
    if world not in _rep:
        _rep[world] = (REPORTS / world / "turn_060_report.txt").read_text()
    return _rep[world]


def qblock(items):
    return "".join(QUESTION.format(k=k, question=i["text"], criteria=i["criteria"]) + "\n" for k, i in enumerate(items, 1))


def make_prompt(kind, world, items):
    pre, tail = (CONT_PREFIX, CONT_TAIL) if kind == "cont" else (BIN_PREFIX, BIN_TAIL)
    return pre.format(report=report(world)), tail.format(n=len(items), questions=qblock(items))


def user_content(prefix, tail, model_id):
    if model_id.startswith(FIRST_PARTY):
        return [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": tail}]
    return prefix + tail


def followup(template, preamble, positions, kind="bin"):
    ql = ", ".join(f"Q{p}" for p in positions)
    ex = "\n".join(f"Q{p}: 0.xx" for p in positions)
    many = len(positions) > 1
    return template.format(preamble=preamble, s="s" if many else "", ies="ies" if many else "y", them="them" if many else "it",
                           qlist=ql, example=ex)


# ---------------------------------------------------------------- parsing
_num = r"([0-9]*\.?[0-9]+)\s*%?"


def _prob(v):
    v = float(v)
    v = v / 100 if v > 1 else v
    return v if 0 <= v <= 1 else None


def parse_block(text, positions, kind):
    """-> ({position: value}, mode). Strict: the last <<<PROBABILITIES>>>/<<<PERCENTILES>>> block. Lenient: 'Qk: ...' lines anywhere
    (last occurrence wins). Positions not found are absent from the dict."""
    text = text or ""
    tag = "PERCENTILES" if kind == "cont" else "PROBABILITIES"
    blocks = re.findall(rf"<<<{tag}>>>(.*?)<<<END>>>", text, re.S)
    src, mode = (blocks[-1], "block") if blocks else (text[-6000:], "lenient")
    out = {}
    if kind == "cont":
        for m in re.finditer(r"Q(\d+)\s*[:.)-]?\s*(.*)", src):
            k = int(m.group(1))
            if k not in positions:
                continue
            vals = dict(re.findall(r"\b(p5|p25|p50|p75|p95)\s*[=:]\s*(-?[0-9][0-9,]*\.?[0-9]*)", m.group(2)))
            try:
                out[k] = [float(vals[q].replace(",", "")) for q in ("p5", "p25", "p50", "p75", "p95")]
            except (KeyError, ValueError):
                continue
    else:
        for m in re.finditer(rf"Q(\d+)\s*[:.)-]\s*{_num}", src):
            k = int(m.group(1))
            if k in positions:
                v = _prob(m.group(2))
                if v is not None:
                    out[k] = v
    return out, (mode if out else None)


# ---------------------------------------------------------------- jobs
def load_models(path):
    return [r for r in csv.DictReader(open(path)) if r.get("openrouter_id")]


def even_split(seq, cap):
    """Fabio's _split_evenly: ceil(n/cap) consecutive chunks whose sizes differ by at most one."""
    n = len(seq)
    if cap <= 0 or n <= cap:
        return [list(seq)]
    k = math.ceil(n / cap)
    q, r = divmod(n, k)
    out, pos = [], 0
    for i in range(k):
        sz = q + (1 if i < r else 0)
        out.append(list(seq[pos:pos + sz]))
        pos += sz
    return out


def build_batches(sets, cap, t2_mode):
    """-> dict(t1=[batch...], t2=[job...], nonews=[job...]).  A batch: id, world, kind, items (list of set items),
    prefix, tail.  A t2 job: id, batch_id, cells [(cell, position)], preamble.  A nonews job: id, batch_id, [(qid, position)]."""
    sets = set(sets.split(",") if isinstance(sets, str) else sets)
    S = {}
    for name, f in FILES.items():
        if name in sets or (name in ("bank",) and "natcond" in sets):
            for i in json.load(open(SETS / f"{f}.json")):
                S.setdefault(i["id"], dict(i, set=name))
    nc, extra = [], {}
    if "natcond" in sets:
        nc = json.load(open(SETS / "natcond_600.json"))
        extra = {i["id"]: dict(i, set="extra") for i in json.load(open(SETS / "natcond_extra_turn1.json"))}
        for c in nc:                     # every natcond question needs a turn-1 answer: bank items or the extra questions
            if c["qid"] not in S:
                S[c["qid"]] = extra[c["qid"]]
    by = collections.defaultdict(list)
    for i in S.values():
        kind = "cont" if i["set"] == "continuous" else "bin"
        by[(i["world"], kind)].append(i)
    batches, pos_of = [], {}
    for (world, kind) in sorted(by):
        items = sorted(by[(world, kind)], key=lambda i: i["id"])
        random.Random(f"{SEED}:{world}:{kind}").shuffle(items)
        chunks = even_split(items, cap)
        for ci, chunk in enumerate(chunks, 1):
            bid = f"{world}_{kind}_c{ci}of{len(chunks)}"
            pre, tail = make_prompt(kind, world, chunk)
            batches.append(dict(id=bid, world=world, kind=kind, items=chunk, prefix=pre, tail=tail))
            for k, i in enumerate(chunk, 1):
                pos_of[i["id"]] = (bid, k)
    t2, nonews = [], []
    if nc:
        if t2_mode == "grouped":
            groups = collections.OrderedDict()
            for c in nc:
                bid, k = pos_of[c["qid"]]
                groups.setdefault((bid, c["rev_id"]), []).append((c, k))
            for (bid, rev), cells in groups.items():
                cells.sort(key=lambda ck: ck[1])
                t2.append(dict(id=f"t2:{bid}:{rev}", batch_id=bid, world=bid.split("_")[0], cells=cells, preamble=cells[0][0]["turn2_preamble"]))
        else:
            for c in nc:
                bid, k = pos_of[c["qid"]]
                t2.append(dict(id=f"t2:{bid}:{c['rev_id']}:{c['qid']}", batch_id=bid, world=bid.split("_")[0], cells=[(c, k)], preamble=c["turn2_preamble"]))
        nn = collections.OrderedDict()
        for c in nc:
            if c["control_no_news"]:
                bid, k = pos_of[c["qid"]]
                nn.setdefault(bid, {})[c["qid"]] = k
        for bid, qs in nn.items():
            nonews.append(dict(id=f"nonews:{bid}", batch_id=bid, world=bid.split("_")[0], qids=sorted(qs.items(), key=lambda x: x[1])))
    return dict(t1=batches, t2=t2, nonews=nonews), pos_of


def smoke_filter(jobs, n_bin=1, n_cont=1):
    """The first binary and the first continuous batch of one world, with their turn-2 and no-news jobs."""
    keep = [b for b in jobs["t1"] if b["kind"] == "bin"][:n_bin] + [b for b in jobs["t1"] if b["kind"] == "cont"][:n_cont]
    ids = {b["id"] for b in keep}
    return dict(t1=keep, t2=[j for j in jobs["t2"] if j["batch_id"] in ids], nonews=[j for j in jobs["nonews"] if j["batch_id"] in ids])


# ---------------------------------------------------------------- OpenRouter
KEY = None
lock = threading.Lock()
spent = collections.defaultdict(float)


def reasoning_param(model):
    mode = (model.get("reasoning_mode") or "none").strip()
    if mode == "effort":
        return {"effort": (model.get("effort") or "low").strip()}
    if mode == "budget":
        return {"max_tokens": int((model.get("budget") or "1024").strip())}
    if mode == "disabled":
        return {"enabled": False}
    return None


def build_body(model, messages, max_tokens_default=32768):
    body = {"model": model["openrouter_id"], "messages": messages, "usage": {"include": True}}
    rp = reasoning_param(model)
    if rp:
        body["reasoning"] = rp
    mt = (model.get("max_tokens") or "").strip() or max_tokens_default
    if mt:
        body["max_tokens"] = int(mt)
    prov = {"require_parameters": True}
    if (model.get("endpoint") or "").strip():
        prov["only"] = [model["endpoint"].strip()]
        prov["allow_fallbacks"] = False
    if (model.get("quantizations") or "").strip():
        prov["quantizations"] = [q.strip() for q in model["quantizations"].split("|") if q.strip()]
    ign = [x.strip() for x in (model.get("ignore") or "").split("|") if x.strip()]
    if ign:
        prov["ignore"] = ign
    body["provider"] = prov
    return body


STREAM = True          # --no-stream turns it off
IDLE_TIMEOUT = 180     # seconds without a byte from the server before the attempt is abandoned and retried


def call(model, messages):
    """One OpenRouter chat completion.  Streamed by default: the socket timeout then applies to every read, so a
    connection that goes silent is abandoned after IDLE_TIMEOUT seconds and retried, while a long but live generation
    keeps arriving (OpenRouter also sends ': OPENROUTER PROCESSING' keep-alive comments).  The smoke of 2026-09-19 saw
    non-streamed calls hang for over 20 minutes with the connection established and nothing coming back."""
    body = build_body(model, messages)
    if STREAM:
        body["stream"] = True
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "Accept": "text/event-stream" if STREAM else "application/json",
                                          "HTTP-Referer": "https://forecastingresearch.org", "X-Title": "fbsim-v3-run2"})
    err, transient, t0 = None, [], time.time()
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=IDLE_TIMEOUT if STREAM else 900) as r:
                if not STREAM:
                    resp = json.load(r)
                    if "error" in resp and not resp.get("choices"):
                        raise RuntimeError(json.dumps(resp["error"])[:300])
                    ch = resp.get("choices", [{}])[0]
                    msg = ch.get("message") or {}
                    txt, rtxt, fin, u, prov = msg.get("content") or "", msg.get("reasoning"), ch.get("finish_reason"), resp.get("usage") or {}, resp.get("provider")
                else:
                    text, reasoning, fin, u, prov, n_chunks = [], [], None, None, None, 0
                    for raw in r:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            j = json.loads(data)
                        except Exception:
                            continue
                        n_chunks += 1
                        if "error" in j and not j.get("choices"):
                            raise RuntimeError(json.dumps(j["error"])[:300])
                        prov = j.get("provider") or prov
                        for ch in j.get("choices", []):
                            d = ch.get("delta") or {}
                            if d.get("content"):
                                text.append(d["content"])
                            if d.get("reasoning"):
                                reasoning.append(d["reasoning"])
                            fin = ch.get("finish_reason") or fin
                        if j.get("usage"):
                            u = j["usage"]
                    txt, rtxt, u = "".join(text), "".join(reasoning) or None, u or {}
                    if not txt and fin is None:
                        raise RuntimeError(f"stream ended after {n_chunks} chunks with no content and no finish_reason")
            cost = float(u.get("cost") or 0)
            with lock:
                spent[model["openrouter_id"]] += cost
            return dict(text=txt, reasoning_text=rtxt, tokens_in=u.get("prompt_tokens"), tokens_out=u.get("completion_tokens"),
                        tokens_reasoning=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        tokens_cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), usage_raw=u, cost=cost, finish=fin,
                        provider=prov, reasoning_param=body.get("reasoning"), max_tokens=body.get("max_tokens"), provider_pref=body["provider"],
                        stream=STREAM, error=None, http_retries=transient, attempts_http=attempt + 1, secs=round(time.time() - t0, 1))
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: " + e.read().decode(errors="replace")[:300]
            transient.append(e.code)
            if e.code in (400, 401, 404) and attempt >= 1:
                break
            if e.code in (402, 429):
                time.sleep(15 + 10 * attempt + 10 * random.random())
                continue
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:250]}"
            transient.append(type(e).__name__)
        time.sleep(min(60, 3 * 2 ** attempt) + random.random())
    return dict(text="", cost=0, error=err, reasoning_param=body.get("reasoning"), http_retries=transient, stream=STREAM, secs=round(time.time() - t0, 1))


# ---------------------------------------------------------------- main
def main():
    global KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma list of names or OpenRouter ids from the models file; default = every row")
    ap.add_argument("--models-file", default=str(HERE / "models_v2.csv"))
    ap.add_argument("--sets", default="bank,tails,mirrors,continuous", help="run 2 default: the natural conditionals stay unbatched (elicit_natcond_v1.py); add natcond to batch them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-prompt", type=int, default=50)
    ap.add_argument("--t2", default="grouped", choices=["grouped", "percell"])
    ap.add_argument("--workers", type=int, default=0, help="override the models-file workers column")
    ap.add_argument("--limit-batches", type=int, default=0, help="only the first N turn-1 batches (and their follow-ups)")
    ap.add_argument("--smoke", action="store_true", help="one binary and one continuous batch of one world, with follow-ups")
    ap.add_argument("--dry-run", action="store_true", help="build the jobs, print counts and write PREFLIGHT_v2.md, make no calls")
    ap.add_argument("--env-file", default="", help="file with OPENROUTER_API_KEY=...; kept outside the repository")
    ap.add_argument("--no-stream", action="store_true", help="plain (non-streamed) completions")
    ap.add_argument("--idle-timeout", type=int, default=180, help="seconds without data before a streamed attempt is retried")
    a = ap.parse_args()
    global STREAM, IDLE_TIMEOUT
    STREAM, IDLE_TIMEOUT = not a.no_stream, a.idle_timeout
    KEY = os.environ.get("OPENROUTER_API_KEY")
    if not KEY and a.env_file:
        for l in open(os.path.expanduser(a.env_file)):
            if l.startswith("OPENROUTER_API_KEY="):
                KEY = l.split("=", 1)[1].strip().strip("\"'")
    models = load_models(a.models_file)
    if a.models:
        want = a.models.split(",")
        models = [m for m in models if m["name"] in want or m["openrouter_id"] in want]
        missing = [w for w in want if not any(m["name"] == w or m["openrouter_id"] == w for m in models)]
        if missing:
            sys.exit(f"not in models file: {missing}")
    jobs, pos_of = build_batches(a.sets, a.per_prompt, a.t2)
    if a.smoke:
        jobs = smoke_filter(jobs)
    if a.limit_batches:
        keep = {b["id"] for b in jobs["t1"][:a.limit_batches]}
        jobs = dict(t1=jobs["t1"][:a.limit_batches], t2=[j for j in jobs["t2"] if j["batch_id"] in keep], nonews=[j for j in jobs["nonews"] if j["batch_id"] in keep])
    n_q = sum(len(b["items"]) for b in jobs["t1"])
    sizes = sorted(len(b["items"]) for b in jobs["t1"])
    print(f"per model: turn-1 prompts={len(jobs['t1'])} ({n_q} questions; batch sizes {sizes[0]}-{sizes[-1]}) "
          f"turn-2 calls={len(jobs['t2'])} ({sum(len(j['cells']) for j in jobs['t2'])} cells; mode {a.t2}) "
          f"no-news calls={len(jobs['nonews'])} ({sum(len(j['qids']) for j in jobs['nonews'])} questions); "
          f"total calls={len(jobs['t1']) + len(jobs['t2']) + len(jobs['nonews'])}; models={len(models)}", flush=True)
    if a.dry_run:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        b = jobs["t1"][0]
        bc = next(x for x in jobs["t1"] if x["kind"] == "cont")
        t2j = next((j for j in jobs["t2"] if j["batch_id"] == b["id"]), None)
        nnj = next((j for j in jobs["nonews"] if j["batch_id"] == b["id"]), None)
        L = ["# Preflight, elicit_v2 (batched)", "", f"Jobs per model: {len(jobs['t1'])} turn-1 prompts, {len(jobs['t2'])} turn-2 calls, "
             f"{len(jobs['nonews'])} no-news calls.", "", "## Example request body (first model, first batch; report truncated)", "```json",
             json.dumps(build_body(models[0], [{"role": "user", "content": user_content(b["prefix"][:600] + " ...", b["tail"], models[0]["openrouter_id"])}]), indent=1)[:4000],
             "```", "", f"## Turn-1 binary prompt, batch {b['id']} ({len(b['items'])} questions)", "```", b["prefix"][:1200] + "\n[... report continues ...]\n" + b["tail"], "```",
             "", f"## Turn-1 continuous prompt, batch {bc['id']} ({len(bc['items'])} questions): tail only", "```", bc["tail"], "```"]
        if t2j:
            L += ["", f"## Turn-2 follow-up ({t2j['id']}, {len(t2j['cells'])} cell(s))", "```", followup(TURN2, t2j["preamble"], [k for _, k in t2j["cells"]]), "```"]
        if nnj:
            L += ["", f"## No-news follow-up ({nnj['id']}, {len(nnj['qids'])} question(s))", "```", followup(NO_NEWS, "", [k for _, k in nnj["qids"]]), "```"]
        (out / "PREFLIGHT_v2.md").write_text("\n".join(L) + "\n")
        print("wrote", out / "PREFLIGHT_v2.md")
        return
    if not KEY:
        sys.exit("OPENROUTER_API_KEY missing (export it or pass --env-file)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f, calls_f = out / "results.jsonl", out / "calls.jsonl"
    calls_done = {}
    if calls_f.exists():
        for line in open(calls_f):
            try:
                c = json.loads(line)
                calls_done[(c["model"], c["call_id"])] = c
            except Exception:
                pass
    rows_out = open(rows_f, "a")
    calls_out = open(calls_f, "a")

    def emit_rows(rows):
        with lock:
            for r in rows:
                rows_out.write(json.dumps(r) + "\n")
            rows_out.flush()

    def emit_call(c):
        with lock:
            calls_out.write(json.dumps(c) + "\n")
            calls_out.flush()
        calls_done[(c["model"], c["call_id"])] = c

    for model in models:
        mid = model["openrouter_id"]
        t0 = time.time()
        W = a.workers or int((model.get("workers") or "").strip() or 16)

        def run_call(call_id, arm, messages, asked, kind, meta):
            """asked: {position: item_id}. Makes the call (re-asking once if any position is unparsed), stores it, emits rows."""
            key = (mid, call_id)
            prev = calls_done.get(key)
            if prev and prev.get("complete"):
                return prev
            best, best_parsed = None, {}
            for attempt in range(2):
                r = call(model, messages)
                parsed, mode = parse_block(r["text"], set(asked), kind)
                if len(parsed) > len(best_parsed) or best is None:
                    best, best_parsed, best_mode, best_attempt = r, parsed, mode, attempt + 1
                if len(parsed) == len(asked) or r.get("error"):
                    break
            n = len(asked)
            rec = dict(model=mid, call_id=call_id, arm=arm, kind=kind, n_asked=n, complete=len(best_parsed) == n, attempts=best_attempt,
                       parse_mode=best_mode, ts=time.time(), messages=messages, **{k: v for k, v in best.items()}, **meta)
            emit_call(rec)
            print(f"  [{time.strftime('%H:%M:%S')}] {mid} {call_id} {best.get('secs')}s out={best.get('tokens_out')} reas={best.get('tokens_reasoning')} "
                  f"cost=${best.get('cost') or 0:.4f} parsed={len(best_parsed)}/{n} finish={best.get('finish')} retries={best.get('http_retries')} err={best.get('error')}", flush=True)
            rows = []
            for k, item in asked.items():
                v = best_parsed.get(k)
                if kind == "cont" and v is not None:
                    v = sorted(v)
                rows.append(dict(model=mid, item=item, arm=arm, kind=kind, world=meta["world"], value=v, parse_mode=best_mode if v is not None else None,
                                 batch=meta["batch_id"], call_id=call_id, position=k, n_in_call=n, attempts=best_attempt, ts=rec["ts"],
                                 cost_share=(best.get("cost") or 0) / n, tokens_in=best.get("tokens_in"), tokens_out=best.get("tokens_out"),
                                 tokens_reasoning=best.get("tokens_reasoning"), tokens_cached=best.get("tokens_cached"), provider=best.get("provider"),
                                 finish=best.get("finish"), error=best.get("error"), **({"t1_value": meta["t1_values"].get(item)} if "t1_values" in meta else {})))
            emit_rows(rows)
            return rec

        # turn 1: the first batch of every (world, kind) first so its report prefix is cached before the rest arrive
        firsts, rest, seen = [], [], set()
        for b in jobs["t1"]:
            g = (b["world"], b["kind"])
            (rest if g in seen else firsts).append(b)
            seen.add(g)
        t1_msgs = {b["id"]: [{"role": "user", "content": user_content(b["prefix"], b["tail"], mid)}] for b in jobs["t1"]}
        with ThreadPoolExecutor(W) as ex:
            list(ex.map(lambda b: run_call(f"t1:{b['id']}", "t1", t1_msgs[b["id"]], {k: i["id"] for k, i in enumerate(b["items"], 1)}, b["kind"],
                                           dict(world=b["world"], batch_id=b["id"])), firsts + rest))
        # follow-ups continue the stored turn-1 exchange of their batch
        def t1_text(bid):
            c = calls_done.get((mid, f"t1:{bid}"))
            return c["text"] if c and c.get("text") else None

        def t1_values(bid):
            c = calls_done.get((mid, f"t1:{bid}"))
            if not c:
                return {}
            b = next(x for x in jobs["t1"] if x["id"] == bid)
            parsed, _ = parse_block(c["text"], set(range(1, len(b["items"]) + 1)), "bin")
            return {i["id"]: parsed.get(k) for k, i in enumerate(b["items"], 1)}

        def follow(job, arm):
            bid = job["batch_id"]
            t1 = t1_text(bid)
            if t1 is None:
                return None
            if arm == "t2":
                positions = [k for _, k in job["cells"]]
                asked = {k: c["qid"] + "|" + c["rev_id"] for c, k in job["cells"]}
                text = followup(TURN2, job["preamble"], positions)
            else:
                positions = [k for _, k in job["qids"]]
                asked = {k: qid for qid, k in job["qids"]}
                text = followup(NO_NEWS, "", positions)
            msgs = t1_msgs[bid] + [{"role": "assistant", "content": t1}, {"role": "user", "content": text}]
            return run_call(job["id"], arm, msgs, asked, "bin", dict(world=job["world"], batch_id=bid, t1_values=t1_values(bid)))

        with ThreadPoolExecutor(W) as ex:
            list(ex.map(lambda j: follow(j, "t2"), jobs["t2"]))
            list(ex.map(lambda j: follow(j, "nonews"), jobs["nonews"]))
        mine = [c for (m, _), c in calls_done.items() if m == mid]
        inc = sum(1 for c in mine if not c.get("complete"))
        print(f"{model['name']} ({mid}): {len(mine)} calls, {inc} with unparsed answers, ${spent[mid]:.2f} this session, {time.time() - t0:.0f}s, workers={W}", flush=True)
    json.dump(dict(spent), open(out / "spend_session.json", "w"), indent=1)
    print("done; rows in", rows_f, "calls in", calls_f)


if __name__ == "__main__":
    main()
