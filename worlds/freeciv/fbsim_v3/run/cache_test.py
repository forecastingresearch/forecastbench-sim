#!/usr/bin/env python3
"""cache_test.py [MODEL_ID ...] — three consecutive turn-1 calls per model on the same world and kind (same prefix, different
questions); prints prompt tokens, cached tokens, cost and provider per call so a prefix-cache hit is visible on calls 2 and 3.
Default: one model per pinned provider family. ~$1.5 for the default list."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
from elicit_v1 import build_jobs, call, load_models, user_content, parse_prob
DEFAULT = ['anthropic/claude-fable-5', 'anthropic/claude-opus-5', 'anthropic/claude-sonnet-5', 'anthropic/claude-haiku-4.5', 'openai/gpt-5.5', 'openai/gpt-5.6-sol',
           'openai/gpt-5', 'openai/o3', 'openai/gpt-5-nano', 'google/gemini-3.7-flash', 'google/gemini-3-flash-preview', 'google/gemini-2.5-flash',
           'qwen/qwen3-235b-a22b', 'deepseek/deepseek-chat', 'meta-llama/llama-4-scout', 'moonshotai/kimi-k2']
ids = sys.argv[1:] or DEFAULT
models = {m['openrouter_id']: m for m in load_models(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_v1.csv'))}
J = build_jobs('bank'); jobs = [j for j in J['t1'] if j['world'] == 'seed7003' and j['kind'] == 'bin'][:3]
def one(mid):
    m = models[mid]; lines = []
    for k, j in enumerate(jobs):
        t0 = time.time(); r = call(m, [{"role": "user", "content": user_content(j['prefix'], j['tail'])}], 'low')
        u = r.get('usage_raw') or {}
        lines.append(f"  call {k + 1}: in={r.get('tokens_in')} cached={r.get('tokens_cached')} cache_write={(u.get('prompt_tokens_details') or {}).get('cache_write_tokens') or u.get('cache_creation_input_tokens')} out={r.get('tokens_out')} "
                     f"cost=${r.get('cost') or 0:.5f} prov={r.get('provider')} {time.time() - t0:.0f}s p={parse_prob(r.get('text'))} err={(r.get('error') or '')[:100]}")
    return f"{mid}\n" + "\n".join(lines)
with ThreadPoolExecutor(len(ids)) as ex:
    for out in ex.map(one, ids): print(out, flush=True)
