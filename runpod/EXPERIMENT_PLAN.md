# Session 2 Experiment Plan — RLVR-on-Freeciv: testing the transfer-failure hypothesis

**Intended reader**: a fresh LLM agent (probably Sonnet) with no prior context on this project, executing this plan autonomously. Every command below should be runnable as-is. Every decision should have a stated motivation. If something fails, the plan includes recovery procedures.

---

## 0. Required reading before you start

Before executing anything, read these in order. They'll take ~20 min total and will save you from repeating mistakes from session 1.

1. **[runpod/README.md](README.md)** — the project's pipeline overview (~5 min). Establishes what the SFT/RL/eval scripts are, the decision-gate logic, and the cost model.
2. **[runpod/SESSION_REPORT.md](SESSION_REPORT.md)** — narrative log of the first session (~10 min). Read especially "What didn't work / what I'd do differently" — those are the footguns this plan exists to fix.
3. **[runpod/WRITEUP.md](WRITEUP.md)** — polished research summary with plots (~5 min). The TL;DR + headline numbers tell you what the first session actually found and what we need to disambiguate.

After reading those: if you understand why we're running E1–E5 and what each one tests, you're ready. If not, re-read the "Three hypotheses" section below.

---

## 1. Context: what session 1 found, and why we're running session 2

**Session 1 result in one sentence**: SFT on dense Monte-Carlo `p_mc` targets learned the held-out Freeciv game (Brier 0.230 → 0.146, beating constant-baseline), but did **not transfer** to ForecastBench (Brier got worse, 0.173 → 0.210). Subsequent RL undid most of the SFT lesson rather than improving it.

**Three competing hypotheses for why** — each of which session 2 tests directly:

- **H1 — Format gap, not learning gap.** Our eval pipeline passes raw text to vLLM, bypassing Qwen3's native `<think>` reasoning trigger. Maybe the base model degenerates on FB because it can't activate thinking mode. **Test: E1** — eval base model with chat template, see if FB parse rate and Brier improve.

- **H2 — SFT-init harmed RL transfer.** The SFT ep4 adapter was already worse on FB than baseline. RL started from there with `KL=0.04` to that adapter; that KL pull-back trapped RL in the FB-degraded basin. **Test: E5** — RL from base, KL=0, with the same Brier reward. If E5 beats E3 (RL-from-SFT) on FB, SFT is harmful for transfer.

- **H3 — In-distribution memorization, not learning.** SFT's 37% Brier reduction on Freeciv-val might just be the model memorizing the 10 training-game world reports, not learning generalizable forecasting. **Test: E4** — hold out 3 templates entirely, eval SFT on the held-out templates. If Brier doesn't improve on held-out templates, the "win" was memorization.

There's also **a clean rerun (E2 + E3)** because session 1 made several mistakes that confound the result:

- Used `--max-completion 256` for RL but evaluated with `max_new_tokens 1024` — train/eval mismatch.
- Used `--num-generations 2` for GRPO — the [gradient-starvation paper](https://arxiv.org/abs/2505.07689) shows G=2 collapses advantage to a sign flip, erasing reward magnitudes.
- Used `--unparseable-penalty -1.0` for RL — ~10× the magnitude of typical `−Brier` rewards, dominating the calibration signal. [Turtel et al. 2025](https://arxiv.org/abs/2505.17989) (the only published forecasting-RLVR work analogous to ours) uses `−0.25` (= "max Brier") for malformed outputs.
- Used `β=0.04` for GRPO KL — outdated; modern consensus is `β∈{0, 0.001}` ([DAPO](https://arxiv.org/abs/2503.14476), [Dr.GRPO](https://arxiv.org/abs/2503.20783)).
- No Brier-on-val callback during SFT — the per-epoch CE wasn't a useful early-stopping signal (epoch 1 best CE, epoch 4 best Brier).

E2 redoes SFT with chat-template applied and a per-epoch Brier callback. E3 redoes RL with all four hyperparameter corrections.

---

## 2. Experiment list and dependency graph

| ID | what | hardware | wall | cost (est.) | tests |
|---|---|---|---|---|---|
| E1 | chat-template eval on base Qwen3-8B at t=0 and t=0.6 | A100 80GB SXM ($1.49/hr) | 30 min | $0.75 | H1 |
| E2 | re-SFT with chat-template + Brier-on-val callback | H200 ($4.39/hr) | 2 h | $9 | (unblocks E3) |
| E3 | RL on E2-best adapter with fixed hyperparameters | H200 | 3 h | $13 | session-1 confound resolution |
| E4 | template-holdout SFT (train on 7/10, eval on 3) | H200 | 2 h | $9 | H3 |
| E5 | RL from base, KL=0 | H200 | 3 h | $13 | H2 |

**Dependency graph**:

```
E1 ──── independent ──────────────────────────────────► report
E2 ──┬─ E3 (requires E2 adapter) ─────────────────────► report
     └─ also produces brier-curve diagnostic           
E4 ──── independent (different data split) ───────────► report
E5 ──── independent (uses base, no SFT) ──────────────► report
```

**Total cost** (either ordering): **~$45**. Parallel and sequential cost about the same in dollars because total GPU-hours are similar; parallel just compresses wall-clock from ~10 h to ~5 h.

---

## 3. Parallel vs sequential execution

### 3a. Parallel (RECOMMENDED — same cost, half the wall-clock)

Launch four pods at once. E1 finishes first (eval-only). E4, E5 finish around the same time (~3h each). E2 and E3 finish last (E2 must complete before E3 starts) at ~5 h total.

**Pod-to-experiment assignment**:

| pod | hardware | runs |
|---|---|---|
| pod-E1 | A100 80GB SXM secure | E1 |
| pod-E2-E3 | H200 secure | E2 then E3 on the same pod (don't terminate between) |
| pod-E4 | H200 secure | E4 |
| pod-E5 | H200 secure | E5 |

Wall-clock: max(0.5, 5, 2, 3) = **~5 h from kickoff**. Cost: ~$45.

### 3b. Sequential (single pod, fallback if parallel pods are supply-constrained)

Run all five on **one H200 pod**. Order: E1 → E2 → E3 → E4 → E5. **Do not terminate the pod between experiments** — keep the HF cache, vLLM compile cache, and venv warm.

```
E1 (30 min, eval) → E2 (2 h, SFT) → E3 (3 h, RL) → E4 (2 h, SFT) → E5 (3 h, RL)
```

Wall-clock: **~10.5 h**. Cost: ~$48.

**Concrete sequential execution**: after pod setup (§5) and code patches deployed (§6), put the five experiment scripts on the pod and chain them. The chain is just `&&`-joined bash, so a failure in any step halts the rest:

```bash
# On the pod, in the persistent SSH session OR via nohup so you can disconnect:
cd /workspace/civbench
nohup bash -c '
set -e
echo "=== sequential chain start $(date) ===" | tee runs/SEQUENTIAL_LOG.txt

# E1: chat-template eval on base (~30 min)
bash /workspace/scripts/run_E1.sh 2>&1 | tee -a runs/SEQUENTIAL_LOG.txt

# E2: re-SFT with chat-template (~2 h)
bash /workspace/scripts/run_E2.sh 2>&1 | tee -a runs/SEQUENTIAL_LOG.txt

# E3: RL on E2-best (~3 h). Depends on E2 having finished. Reads
# runs/E2/brier_curve.jsonl to pick the best epoch automatically:
BEST_CKPT=$(python -c "
import json
rows = [json.loads(l) for l in open(\"runs/E2/brier_curve.jsonl\") if l.strip()]
best = min((r for r in rows if r.get(\"val_brier\") is not None), key=lambda r: r[\"val_brier\"])
print(f\"runs/E2/checkpoint-{best[\\\"step\\\"]}\")
")
export BEST_CKPT
bash /workspace/scripts/run_E3.sh 2>&1 | tee -a runs/SEQUENTIAL_LOG.txt

# E4: template-holdout SFT (~2 h, independent of E2/E3)
bash /workspace/scripts/run_E4.sh 2>&1 | tee -a runs/SEQUENTIAL_LOG.txt

# E5: RL from base (~3 h, independent)
bash /workspace/scripts/run_E5.sh 2>&1 | tee -a runs/SEQUENTIAL_LOG.txt

echo "=== sequential chain done $(date) ===" | tee -a runs/SEQUENTIAL_LOG.txt
' > /workspace/run_chain.log 2>&1 &
disown
echo "started sequential chain in background PID=$!"
```

Each `run_E?.sh` script is just the commands from §7 below, wrapped in a heredoc-written script file. **Create all five scripts before kicking off the chain** so any typos surface immediately, not 5 hours in.

Five reasons sequential is cost-competitive with parallel:
1. The HF model cache (16 GB Qwen3-8B) is downloaded once vs 4 times.
2. The vLLM compile graph (~30 sec) is cached and reused across evals.
3. The installed Python stack (~5 min) is set up once.
4. The pod boot + sshd init time (~3 min) happens once.
5. No need to rsync data five times.

**Why I still recommend parallel**: with parallel pods you get the results 2× faster, and the dollar difference is <$3. Time saved >> $3.

---

## 4. Cost-saving techniques worth applying (from research, not speculation)

Two ideas to apply across all experiments, regardless of parallel-vs-sequential:

### 4a. RunPod Network Volume for cross-pod cache reuse

Source: [RunPod docs on volumes](https://docs.runpod.io/pods/storage/types) and [Axolotl GH issue #216](https://github.com/axolotl-ai-cloud/axolotl/issues/216).

Standard RunPod pod-attached volumes are wiped when you terminate. A **Network Volume** ($0.07 GB/mo, region-locked, cheap) persists across pod terminations. You attach it to multiple pods over time and reuse its contents.

For our workflow, **put on the Network Volume**: `~/.cache/uv`, `~/.cache/vllm`, the HF cache (`$HF_HOME = /workspace/.hf_cache`), and the venv. Then each new pod mounts the Network Volume at `/workspace` and is "ready" in ~1 min instead of ~15 min.

**However** — for this 5-experiment session, the cost savings are marginal ($1–2 across the whole session). Recommend creating a network volume only if you anticipate running more experiments later. **For now, skip this and accept the per-pod 5-minute setup overhead**.

### 4b. vLLM colocate tuning

Source: [HF blog on vllm colocate](https://github.com/huggingface/blog/blob/main/vllm-colocate.md), [verl perf docs](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

For GRPO with `use_vllm=True` and `vllm_mode="colocate"` (the only viable option on a single GPU):

- **`vllm_gpu_memory_utilization=0.25` for 7-8B models** (not the vLLM default of 0.9, and not session-1's 0.35). The HF/TRL example for Phi-4-mini uses 0.25, for Qwen2.5-7B uses 0.30. Lower means more memory for training; we have headroom on H200.
- **`enforce_eager=False`** (the default) — keeps CUDA graphs on. Reported 39% throughput penalty if disabled. We're already correct here.
- **vLLM 0.11+ on H200 uses FlashAttention 3 by default** — no install needed. We're on 0.11.2; already covered.
- ⚠️ **DO NOT set `vllm_sleep_level=2` in TRL 0.21**. vLLM `sleep()` integration in TRL is incomplete — TRL maintainers are "waiting for an official fix before integrating sleep() fully" due to a segfault at training shutdown ([HF colocate blog](https://huggingface.co/blog/vllm-colocate), [vLLM issue #16993](https://github.com/vllm-project/vllm/issues/16993)). My earlier recommendation here was wrong. Leave it unset.

Net expected speedup vs session-1 GRPO: ~10–15% from `vllm_gpu_memory_utilization=0.25` alone.

### 4c. FlashAttention 2 pre-built wheel (recommended ~5 min install)

Session 1 tried to install `flash-attn` from source (no wheel for `cu128+torch2.8` triggered a 20+ min nvcc compile). I was wrong about that — **Dao-AILab ships pre-built wheels on [GitHub Releases](https://github.com/Dao-AILab/flash-attention/releases)** matching `cp{python}-cu{cuda}-torch{ver}` tags. With torch 2.8 + cu128, find the matching wheel URL and `pip install <wheel-url>` — installs in <1 min, no nvcc.

**Expected speedup**: FA2 vs SDPA on Qwen3-8B at 8K context: ~1.3-1.5× attention throughput per [Dao-AILab benchmarks](https://github.com/Dao-AILab/flash-attention) and the [HF perf-infer docs](https://huggingface.co/docs/transformers/main/perf_infer_gpu_one). Net SFT wall-clock improvement: ~15-25%. Our `02_sft.py` already auto-detects flash_attn (falls back to SDPA if missing), so just install the wheel and rerun.

```bash
# On the pod, after main stack install:
TORCH_VER=$(python -c "import torch; v=torch.__version__.split('+')[0]; print('.'.join(v.split('.')[:2]))")
# Visit https://github.com/Dao-AILab/flash-attention/releases and pick the wheel matching cp311 + cu128 + torch${TORCH_VER}
# As of 2026-06, e.g.:
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install --system --no-cache-dir "$WHEEL_URL" || echo "FA2 wheel install failed — will fall back to SDPA, no fatal"
```

⚠️ **Caveat**: open [Dao-AILab issue #2151](https://github.com/Dao-AILab/flash-attention/issues/2151) reports FA2 produces measurably higher training loss vs SDPA on some setups. If you see clearly higher train loss with FA2 vs SDPA on first few steps of E2, fall back to SDPA. (Not blocking — train-time loss is not our north star; Brier-on-val is.)

### 4d. Skip `packing=True`

TRL's `packing=True` only helps when sequences are short relative to `max_seq_length`. Our prompts are 6.8K tokens at p95 and max_seq=8192 (83% fill, low variance) — expected speedup ≈ 1.0× to 1.1× per [IBM/HF packing benchmarks](https://huggingface.co/blog/packing-with-FA2). Not worth the risk of cross-document-attention bugs. **Keep `packing=False`** (the current default).

### 4e. Optional: Liger-Kernel (big speedup, but adds a dep)

[Liger-Kernel](https://github.com/linkedin/Liger-Kernel) is LinkedIn's Triton kernel library for LLM training. Reported numbers on H100 Llama-3.1-8B: **+22% throughput, −60% VRAM** at the same effective batch size ([Spheron benchmark](https://www.spheron.network/blog/liger-kernel-llm-training-gpu-cloud/)). Plugs into TRL's SFTTrainer with one flag.

**Don't apply by default** — it's an additional dep that needs Qwen3 support verified. If E2 SFT takes longer than 2.5h, optionally add `--use-liger-kernel` to the trainer config and re-test. Documented but not blocking.

### 4f. HF_HUB_OFFLINE after first download

After the initial Qwen3-8B download completes, set `HF_HUB_OFFLINE=1` in subsequent scripts so transformers loads from local cache instead of re-querying HF Hub. This avoids any further rate-limiting risk and shaves ~5-10 sec off every model load.

```bash
export HF_HUB_OFFLINE=1  # add to E1-E5 scripts AFTER the initial Qwen/Qwen3-8B download
```

---

## 5. Shared pod setup (run once per pod)

Every pod follows the same setup sequence. This is fully automatable.

### 5a. Provision (via RunPod GraphQL API)

```python
# Run from your local shell. Reads RunPod API key from .env.
# Each pod gets the same image, same env vars, same volume size.

import os
import subprocess
import json
import urllib.request

api_key = subprocess.check_output(
    ["grep", "^RUNPOD_API_KEY=", ".env"], cwd=".").decode().strip().split("=", 1)[1]
hf_token = subprocess.check_output(
    ["grep", "^HF_ACCESS_TOKEN=", ".env"], cwd=".").decode().strip().split("=", 1)[1]
pubkey = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()

POD_SPECS = {
    "E1":    {"gpu": "NVIDIA A100-SXM4-80GB", "name": "civbench-E1"},
    "E2-E3": {"gpu": "NVIDIA H200",            "name": "civbench-E2E3"},
    "E4":    {"gpu": "NVIDIA H200",            "name": "civbench-E4"},
    "E5":    {"gpu": "NVIDIA H200",            "name": "civbench-E5"},
}

def provision(spec):
    mutation = f"""mutation {{
        podFindAndDeployOnDemand(input: {{
            cloudType: SECURE,
            gpuCount: 1,
            volumeInGb: 200,
            containerDiskInGb: 100,
            minVcpuCount: 8,
            minMemoryInGb: 64,
            gpuTypeId: \\"{spec['gpu']}\\",
            name: \\"{spec['name']}\\",
            imageName: \\"runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04\\",
            ports: \\"22/tcp\\",
            volumeMountPath: \\"/workspace\\",
            env: [
              {{key: \\"PUBLIC_KEY\\", value: \\"{pubkey}\\"}},
              {{key: \\"HF_TOKEN\\", value: \\"{hf_token}\\"}}
            ]
        }}) {{ id costPerHr machine {{ location }} }}
    }}"""

    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": mutation}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    resp = json.loads(urllib.request.urlopen(req).read())
    return resp["data"]["podFindAndDeployOnDemand"]
```

**Polling for SSH readiness** (each provisioned pod takes ~1-3 min to bring up sshd). Wait until you get a TCP port mapping for port 22 AND `ssh <ip> 'echo ok'` works.

### 5b. Install stack (each pod, ~5 min — repeat for every pod)

```bash
# SSH into the pod. Replace <IP> and <PORT> from the GraphQL response.
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_runpod -p <PORT> -i $HOME/.ssh/id_ed25519"
ssh $SSH_OPTS root@<IP> 'bash -s' <<'EOF'
set -e

# 1. Install uv (fast Python package manager — pip's resolver OOMs on community pods)
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH

# 2. apt deps (rsync needed for file transfer)
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq rsync >/dev/null 2>&1

# 3. Install training stack
# Pin reasons:
#   vllm==0.11.2  — has FA3 on Hopper/H200, Qwen3 support, TRL 0.21 compat
#   trl==0.21.0   — has vllm_mode="colocate" arg (was added in 0.18+)
#   transformers>=4.51 — has Qwen3 model_type
# Install vllm with --no-deps first to avoid resolver hangs, then full deps.
uv pip install --system --no-cache-dir --no-deps "vllm==0.11.2"
uv pip install --system --no-cache-dir \
  "transformers>=4.51,<4.55" "trl==0.21.0" "peft>=0.13,<0.20" \
  "accelerate>=1.0" "datasets>=3.0" "bitsandbytes>=0.44" \
  "huggingface_hub[cli,hf_transfer]" cachetools
uv pip install --system --no-cache-dir "vllm==0.11.2"

# 4. HF auth (premium account, no rate limits)
# Note: user named env var HF_TOKEN here even though .env has HF_ACCESS_TOKEN.
# The setup-time provisioning step above must rename it. If running manually,
# do: huggingface-cli login --token "$HF_ACCESS_TOKEN" instead.
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# 5. Optional: install FA2 pre-built wheel (15-25% SFT speedup, <1 min install)
# Check the latest matching wheel at https://github.com/Dao-AILab/flash-attention/releases
# Pattern: flash_attn-<VER>+cu128torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
FA2_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install --system --no-cache-dir "$FA2_WHEEL" 2>&1 | tail -3 || \
  echo "[setup] FA2 wheel install failed — will fall back to SDPA"
python -c "import flash_attn; print('flash_attn version:', flash_attn.__version__)" 2>&1 | head -1

# 6. Pre-download Qwen3-8B (16GB; ~1-2 min with HF premium)
export HF_HOME=/workspace/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download Qwen/Qwen3-8B
du -sh /workspace/.hf_cache

# 7. After download done, switch to offline mode for the experiments
unset HF_HUB_ENABLE_HF_TRANSFER  # avoid hf_transfer issues on resume reads
# Add to subsequent scripts: export HF_HUB_OFFLINE=1

# 6. Create work directories
mkdir -p /workspace/civbench/data/forecastbench
mkdir -p /workspace/civbench/data/training
mkdir -p /workspace/civbench/runpod
mkdir -p /workspace/civbench/scripts
mkdir -p /workspace/civbench/runs

echo "[setup] complete $(date)"
EOF
```

### 5c. Rsync repo + data (each pod)

```bash
# From your local repo root
RSYNC_SSH="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_runpod -p <PORT> -i $HOME/.ssh/id_ed25519"
RSYNC_FLAGS="-rltDz"  # no -p/-o/-g because /workspace is a network volume on RunPod
rsync $RSYNC_FLAGS -e "$RSYNC_SSH" --exclude='__pycache__' runpod/        root@<IP>:/workspace/civbench/runpod/
rsync $RSYNC_FLAGS -e "$RSYNC_SSH" scripts/build_forecastbench_eval.py    root@<IP>:/workspace/civbench/scripts/
rsync $RSYNC_FLAGS -e "$RSYNC_SSH" data/forecastbench/eval_post2025.csv   root@<IP>:/workspace/civbench/data/forecastbench/
rsync $RSYNC_FLAGS -e "$RSYNC_SSH" data/training/                          root@<IP>:/workspace/civbench/data/training/
```

**Common failure**: BSD rsync on macOS doesn't support `--no-owner --no-group`. Use only `-rltDz` (no `-a`). The `chown` errors that result are harmless (network volume rejects chown but the file transfer succeeds).

---

## 6. Code changes required (apply locally, rsync to pod)

Three patches needed to existing scripts. **Apply these in the local repo before rsyncing.**

### 6a. `runpod/04_eval.py` — `--chat-template` flag (already applied in session 1, verify it's there)

This flag should already be in the script as of session 1. Verify by grepping:

```bash
grep -c "chat_template" runpod/04_eval.py  # should be > 0
grep -c "llm.chat" runpod/04_eval.py        # should be > 0
```

If missing, the patch:

```python
# In argparse:
ap.add_argument("--chat-template", action="store_true",
                help="Wrap prompts as user messages and apply the model's chat template.")

# Replacement for the generate call:
if args.chat_template:
    messages = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(messages, sp, lora_request=lora_req)
else:
    outs = llm.generate(prompts, sp, lora_request=lora_req)
```

### 6b. `runpod/02_sft.py` — switch to messages format + add Brier-on-val callback

The current script passes a `text` field to TRL, which TRL doesn't actually chat-template-wrap (despite the misleading "Converting train dataset to ChatML" log line — see [SESSION_REPORT.md](SESSION_REPORT.md) addendum). To force chat-template application during SFT, switch to the `messages` field.

Apply this diff:

```python
# Replace the to_text() function with:
def to_messages(ex: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": fmt_target(ex["p_mc"])},
        ],
        "weight": ex.get("weight", 1.0),
    }

# Replace the dataset-building lines:
train_recs = [to_messages(r) for r in train]
val_recs   = [to_messages(r) for r in val]

# weight-expansion same as before, but build dicts with "messages":
if args.use_weights:
    K = 4
    expanded = []
    for r in train_recs:
        n = max(1, round(r["weight"] * K))
        expanded.extend([r] * n)
    train_recs = expanded

train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_recs])
val_ds   = Dataset.from_list([{"messages": r["messages"]} for r in val_recs])

# In SFTConfig, REMOVE `dataset_text_field="text"`. TRL auto-detects messages format.
```

Add the Brier-on-val callback (paste below the existing `JsonlLogCallback`):

```python
class BrierOnValCallback(TrainerCallback):
    """At each epoch end, generate completions on a val subset and compute Brier
    on the parsed answers. Writes a line to runs/<output>/brier_curve.jsonl per epoch.

    Why this exists: per-token cross-entropy diverges from Brier in this setup
    (session 1 epoch 1 had best CE but epoch 4 had best Brier on Freeciv-val).
    CE-based early stopping killed the experiment too early in session 1.
    """
    PROB_RE = __import__("re").compile(
        r"PROBABILITY:\s*([01](?:\.\d+)?|\.\d+)", __import__("re").IGNORECASE)

    def __init__(self, val_records: list, tokenizer, output_dir: str, max_eval: int = 30):
        self.val_records = val_records[:max_eval]
        self.tok = tokenizer
        self.out_path = Path(output_dir) / "brier_curve.jsonl"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        open(self.out_path, "w").close()

    def on_epoch_end(self, args, state, control, model=None, **kw):
        if model is None:
            return
        import torch
        model.eval()
        squared_errors, n_parsed = [], 0
        with torch.no_grad():
            for ex in self.val_records:
                msgs = [{"role": "user", "content": ex["prompt"]}]
                ids = self.tok.apply_chat_template(
                    msgs, return_tensors="pt", add_generation_prompt=True
                ).to(model.device)
                try:
                    out = model.generate(
                        ids, max_new_tokens=256, do_sample=False,
                        pad_token_id=self.tok.pad_token_id,
                    )
                    text = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                    m = self.PROB_RE.findall(text)
                    if not m:
                        continue
                    pred = float(m[-1])
                    if not (0.0 <= pred <= 1.0):
                        continue
                    squared_errors.append((pred - ex["p_mc"]) ** 2)
                    n_parsed += 1
                except Exception as e:
                    print(f"[brier-callback] error on val example: {e}")
                    continue
        brier = sum(squared_errors) / max(1, n_parsed) if squared_errors else None
        with open(self.out_path, "a") as f:
            f.write(__import__("json").dumps({
                "epoch": state.epoch, "step": state.global_step,
                "val_brier": brier, "n_parsed": n_parsed,
                "n_total": len(self.val_records),
            }) + "\n")
        print(f"[brier-callback] epoch {state.epoch:.1f}: "
              f"val_brier={brier} (n_parsed={n_parsed}/{len(self.val_records)})")
        model.train()

# Wire it up where you currently create the trainer:
# Note: pass the ORIGINAL val records (with 'prompt' and 'p_mc' fields), not val_recs
val_for_callback = val   # the list of raw jsonl records loaded earlier
brier_cb = BrierOnValCallback(val_for_callback, tok, args.output)
log_cb = JsonlLogCallback(metrics_path)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                     eval_dataset=val_ds, processing_class=tok,
                     callbacks=[log_cb, brier_cb])
```

### 6c. `runpod/03_rl.py` — chat-template prompts + new defaults + `--no-adapter` for E5

Three changes:

```python
# Change 1: in argparse, make --adapter optional and add no-adapter behavior
ap.add_argument("--adapter", default=None,
                help="Path to SFT LoRA adapter to start from. If omitted, RL from base.")

# Change 2: in model loading, support no-adapter path
if args.adapter is None:
    print(f"[rl] starting from base model (no SFT adapter)")
    lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
else:
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)

# Change 3: apply chat template at dataset build time (replaces raw prompts)
def to_record(ex):
    msgs = [{"role": "user", "content": ex["prompt"]}]
    prompt_text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt_text, "p_mc": float(ex["p_mc"]),
            "weight": float(ex.get("weight", 1.0)),
            "qid": ex["qid"], "game_id": ex["game_id"]}

# Change 4: vLLM colocate tuning in GRPOConfig
# Lower gpu_memory_utilization (was 0.35 in session 1; HF colocate blog
# recommends 0.25-0.30 for 7-8B models). DO NOT set vllm_sleep_level — TRL 0.21
# integration is incomplete (segfault at training shutdown per vLLM #16993).
cfg = GRPOConfig(
    ...,
    use_vllm=args.use_vllm,
    vllm_mode="colocate" if args.use_vllm else None,
    vllm_gpu_memory_utilization=0.25 if args.use_vllm else None,
    ...
)
```

Verify all three changes are applied before rsyncing.

---

## 7. The five experiments — detailed commands

Each subsection below is **self-contained**. Each defines its purpose, preconditions, exact commands to run on the pod, expected output files, and success criterion.

### E1 — chat-template eval on base Qwen3-8B

**Tests**: H1 — does activating Qwen3's `<think>` reasoning trigger fix FB at temperature 0?

**Preconditions**: pod with Qwen3-8B cached, `runpod/04_eval.py` has `--chat-template` flag verified.

**Commands** (run on the pod, ~25 min wall):

```bash
cd /workspace/civbench
export HF_HOME=/workspace/.hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Run 1: greedy + chat template
mkdir -p runs/E1/chat_t0
python runpod/04_eval.py \
  --base Qwen/Qwen3-8B \
  --forecastbench data/forecastbench/eval_post2025.csv \
  --freeciv-val data/training/val.jsonl \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --chat-template \
  --output runs/E1/chat_t0/eval.json 2>&1 | tee runs/E1/chat_t0/log.txt

# Run 2: sampled (t=0.6) + chat template
mkdir -p runs/E1/chat_t06
python runpod/04_eval.py \
  --base Qwen/Qwen3-8B \
  --forecastbench data/forecastbench/eval_post2025.csv \
  --freeciv-val data/training/val.jsonl \
  --max-new-tokens 1024 \
  --temperature 0.6 \
  --chat-template \
  --output runs/E1/chat_t06/eval.json 2>&1 | tee runs/E1/chat_t06/log.txt
```

**Expected output**:
- `runs/E1/chat_t0/eval.json` and `runs/E1/chat_t06/eval.json` with `forecastbench` and `freeciv_val` sections.

**Success criterion**:
- Both exit 0.
- `n_parsed/n ≥ 0.85` on FB at both temperatures (if `<think>` trigger fixes loops, expect ≥0.90 at t=0).

**Reporting**: compare FB Brier and parse-rate to the without-template session-1 baselines: t=0 was 36% parsed / 0.182 Brier; t=0.6 was 93% parsed / 0.173 Brier. Substantial movement (especially at t=0) supports H1.

### E2 — re-SFT with chat-template + Brier-on-val callback

**Tests**: rebuild SFT-ep4 adapter under correct format (the session-1 adapter weights are gone). Unblocks E3.

**Preconditions**: pod with Qwen3-8B cached, `02_sft.py` patched per §6b.

**Commands** (~2 h wall):

```bash
cd /workspace/civbench
export HF_HOME=/workspace/.hf_cache
export TOKENIZERS_PARALLELISM=false

# Important: --no-quant uses bf16 base (faster on H200), --grad-checkpoint is
# required at seq=8192 to fit memory (session 1 OOM'd at no_grad_ckpt).
python runpod/02_sft.py \
  --train data/training/train.jsonl \
  --val   data/training/val.jsonl \
  --model Qwen/Qwen3-8B \
  --output runs/E2 \
  --epochs 4 \
  --batch 2 \
  --grad-accum 8 \
  --lr 2e-4 \
  --lora-r 16 \
  --max-seq 8192 \
  --use-weights \
  --no-quant \
  --grad-checkpoint 2>&1 | tee runs/E2/log.txt
```

**Expected output**:
- 4 epoch checkpoints: `runs/E2/checkpoint-{69,138,207,276}/`
- `runs/E2/sft_metrics.jsonl` per-step metrics
- `runs/E2/brier_curve.jsonl` per-epoch Brier-on-val (the new metric — this is the key diagnostic)
- `runs/E2/adapter/` (alias of last checkpoint)

**Success criterion**:
- All 4 epochs complete.
- `brier_curve.jsonl` has 4 lines, each with non-null `val_brier`.
- Best `val_brier` < 0.230 (= base-model Brier on Freeciv-val) — i.e., SFT moved the model at all.

**Reporting**: print the brier_curve.jsonl contents in the experiment log. Identify best epoch by `val_brier`. **Pass that checkpoint name to E3** (replace `BEST_CKPT` below).

### E3 — RL on E2-best adapter with corrected hyperparameters

**Tests**: does RL move the FB and Freeciv Brier when reward, KL, G, and token budget are all properly tuned?

**Preconditions**: E2 finished. **Note the best-epoch checkpoint name from E2** — likely `checkpoint-276` based on session 1, but verify.

**Commands** (~3 h wall):

```bash
cd /workspace/civbench
export HF_HOME=/workspace/.hf_cache
export TOKENIZERS_PARALLELISM=false

# Replace with actual best-epoch path from E2 brier_curve.jsonl
BEST_CKPT=runs/E2/checkpoint-276

python runpod/03_rl.py \
  --train data/training/train.jsonl \
  --val   data/training/val.jsonl \
  --base  Qwen/Qwen3-8B \
  --adapter $BEST_CKPT \
  --output runs/E3 \
  --num-generations 4 \
  --batch 1 \
  --grad-accum 16 \
  --steps 300 \
  --max-prompt 7600 \
  --max-completion 1024 \
  --kl-coef 0.001 \
  --unparseable-penalty -0.25 \
  --dr-grpo \
  --use-weights \
  --use-vllm \
  --no-quant \
  --grad-checkpoint 2>&1 | tee runs/E3/log.txt

# After RL completes, eval the final adapter with chat template:
python runpod/04_eval.py \
  --base Qwen/Qwen3-8B \
  --adapter runs/E3/adapter \
  --forecastbench data/forecastbench/eval_post2025.csv \
  --freeciv-val data/training/val.jsonl \
  --max-new-tokens 1024 \
  --temperature 0.6 \
  --chat-template \
  --output runs/E3/eval_chat_t06.json 2>&1 | tee runs/E3/eval_log.txt
```

**Hyperparameter notes** (these differ from session 1; cite if challenged):
- `--num-generations 4` (was 2): G=2 collapsed advantage to ±sign per [gradient-starvation paper](https://arxiv.org/abs/2505.07689)
- `--max-completion 1024` (was 256): matches eval-time budget — session 1 had train/eval mismatch
- `--kl-coef 0.001` (was 0.04): modern consensus from [DAPO](https://arxiv.org/abs/2503.14476) / [Dr.GRPO](https://arxiv.org/abs/2503.20783) is β≤0.001
- `--unparseable-penalty -0.25` (was -1.0): matches `Soft Brier` from [Turtel et al. 2025](https://arxiv.org/abs/2505.17989)

**Success criterion**:
- All 300 steps complete; final adapter saved.
- Mean reward in last 20 steps > mean reward in first 20 steps (i.e., learning actually happened; session 1 RL was flat).
- Post-RL eval Brier on Freeciv-val ≤ E2 best-epoch Brier (i.e., RL preserves SFT gain — session 1 lost it).

### E4 — template-holdout SFT

**Tests**: H3 — does SFT generalize across templates within Freeciv?

**Preconditions**: split the data locally before rsyncing (do this BEFORE provisioning the pod, since data needs to be on the pod at start). Then standard SFT pod.

**Local data prep** (run once on your laptop):

```bash
mkdir -p data/training_E4
uv run python <<'EOF'
import json
HOLD_OUT = ['wonder_completed', 'territory_comparative', 'tech_comparative']
# Reason for these three: wonder_completed had the worst SFT Brier in session 1
# (0.45), tech_comparative was the 3rd-worst on RL (0.32), territory_comparative
# is a mid-difficulty geometric template. Together they span easy/hard, and
# none is overly trivial (no extreme p_mc clustering).
train = [json.loads(l) for l in open('data/training/train.jsonl')]
val   = [json.loads(l) for l in open('data/training/val.jsonl')]
all_recs = train + val
train_new = [r for r in all_recs if r.get('template_id') not in HOLD_OUT]
heldout   = [r for r in all_recs if r.get('template_id') in HOLD_OUT]
with open('data/training_E4/train.jsonl','w') as f:
    for r in train_new: f.write(json.dumps(r)+'\n')
with open('data/training_E4/val.jsonl','w') as f:
    for r in heldout: f.write(json.dumps(r)+'\n')
print(f"train: {len(train_new)} ({len({r['template_id'] for r in train_new})} templates)")
print(f"held-out: {len(heldout)} ({HOLD_OUT})")
EOF
```

Then `rsync data/training_E4/ root@<IP>:/workspace/civbench/data/training_E4/` to the pod.

**Commands** (~2 h wall):

```bash
cd /workspace/civbench
export HF_HOME=/workspace/.hf_cache

python runpod/02_sft.py \
  --train data/training_E4/train.jsonl \
  --val   data/training_E4/val.jsonl \
  --model Qwen/Qwen3-8B \
  --output runs/E4 \
  --epochs 4 \
  --batch 2 \
  --grad-accum 8 \
  --lr 2e-4 \
  --lora-r 16 \
  --max-seq 8192 \
  --use-weights \
  --no-quant \
  --grad-checkpoint 2>&1 | tee runs/E4/log.txt

# After SFT: eval each saved checkpoint on the held-out templates
for ckpt in runs/E4/checkpoint-*; do
  name=$(basename $ckpt)
  mkdir -p runs/E4/eval_$name
  python runpod/04_eval.py \
    --base Qwen/Qwen3-8B \
    --adapter $ckpt \
    --freeciv-val data/training_E4/val.jsonl \
    --max-new-tokens 1024 \
    --temperature 0.6 \
    --chat-template \
    --output runs/E4/eval_$name/eval.json 2>&1 | tee runs/E4/eval_$name/log.txt
done
```

**Success criterion**:
- 4 epochs complete; brier_curve.jsonl has 4 lines.
- 4 per-epoch eval.json files on the held-out templates.

**Reporting**: report Brier per template per epoch (matrix). If held-out template Brier improves substantially (≥10% relative vs base), the SFT signal is generalizable. If held-out Brier doesn't move, SFT memorized world reports per template — strong negative result and the headline of session 2.

### E5 — RL from base (no SFT)

**Tests**: H2 — does the SFT step actively harm transfer? Direct ablation against E3.

**Preconditions**: pod with Qwen3-8B cached, `03_rl.py` patched per §6c.

**Commands** (~3 h wall):

```bash
cd /workspace/civbench
export HF_HOME=/workspace/.hf_cache
export TOKENIZERS_PARALLELISM=false

# No --adapter flag — uses the new from-base path.
# kl-coef=0.0 per DAPO/Dr.GRPO — from-base RL doesn't need KL constraint.
python runpod/03_rl.py \
  --train data/training/train.jsonl \
  --val   data/training/val.jsonl \
  --base  Qwen/Qwen3-8B \
  --output runs/E5 \
  --num-generations 4 \
  --batch 1 \
  --grad-accum 16 \
  --steps 300 \
  --max-prompt 7600 \
  --max-completion 1024 \
  --kl-coef 0.0 \
  --unparseable-penalty -0.25 \
  --dr-grpo \
  --use-weights \
  --use-vllm \
  --no-quant \
  --grad-checkpoint 2>&1 | tee runs/E5/log.txt

# Post-RL eval with chat template
python runpod/04_eval.py \
  --base Qwen/Qwen3-8B \
  --adapter runs/E5/adapter \
  --forecastbench data/forecastbench/eval_post2025.csv \
  --freeciv-val data/training/val.jsonl \
  --max-new-tokens 1024 \
  --temperature 0.6 \
  --chat-template \
  --output runs/E5/eval_chat_t06.json 2>&1 | tee runs/E5/eval_log.txt
```

**Success criterion**:
- All 300 steps complete; reward not flat (last-20 mean > first-20 mean).
- Post-RL eval Brier comparable to E1 baseline (i.e., RL-from-base at least matches the base model).

**Interpretation**:
- If E5 FB Brier < E3 FB Brier → SFT actively hurt transfer; drop SFT for FB-focused work.
- If E5 FB Brier ≈ E3 FB Brier → SFT vs no-SFT doesn't matter; bottleneck is elsewhere (probably the task itself).
- If E5 FB Brier > E3 FB Brier → SFT init genuinely helped; session-1's RL just had bad hyperparameters.

---

## 8. Post-experiment: collect, compare, terminate

After all experiments complete:

```bash
# Pull all results back to local
mkdir -p tmp/session2
for pod in <E1_IP:E1_PORT> <E2E3_IP:E2E3_PORT> <E4_IP:E4_PORT> <E5_IP:E5_PORT>; do
  ip=${pod%:*}; port=${pod#*:}
  rsync -rltDz -e "ssh -p $port -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new" \
    --include='*/' --include='eval.json' --include='*.jsonl' --include='*.json' \
    --include='checkpoint-*/adapter_model.safetensors' \
    --include='checkpoint-*/adapter_config.json' \
    --include='adapter/adapter_model.safetensors' \
    --include='adapter/adapter_config.json' \
    --exclude='*' \
    root@$ip:/workspace/civbench/runs/ tmp/session2/
done
```

**Important**: `--include='checkpoint-*/adapter_model.safetensors'` (and same for `adapter/`) — session 1 forgot to pull the actual LoRA weights and they were lost when pods terminated. DO NOT make that mistake again.

Then terminate all pods (don't leave them up overnight):

```python
# For each pod_id you tracked at provision time:
import json, urllib.request
api_key = open(".env").read().split("RUNPOD_API_KEY=", 1)[1].split("\n", 1)[0]
for pod_id in pod_ids_to_terminate:
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}'}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    print(urllib.request.urlopen(req).read())
```

### 8a. Compile the results table

Create `runpod/SESSION_REPORT_V2.md` and add a table with these columns:

| condition | FB parse | FB Brier | FB vs const | Freeciv parse | Freeciv Brier | Freeciv vs const |

Populate with:
- Session 1 baselines (from `runpod/WRITEUP.md` headline table)
- E1: chat template at t=0 and t=0.6
- E2: best-epoch by val_brier
- E3: final RL adapter
- E4: per-checkpoint, focus on held-out templates
- E5: final RL-from-base adapter

### 8b. Address each hypothesis in turn

For each of H1/H2/H3 (defined in §1 of this doc), write a paragraph stating:
- What the relevant experiment(s) measured
- The numeric result
- Whether the hypothesis is supported, refuted, or inconclusive
- What the next experiment after this would be

### 8c. Update the writeup with new plots

The session-1 plotting script is at `runpod/make_plots.py`. Extend it to add panels for session-2 data: brier curves for E2 and E4, FB Brier across all session-1 + session-2 conditions, per-template breakdown for E4 held-out vs trained-on templates.

---

## 9. Common pitfalls / debugging

These all bit session 1; document so they don't bite again.

### 9a. SSH timeouts during training

vLLM colocate is GPU+CPU heavy; sshd can stall. **This does not mean training crashed**. Pattern: `ssh: connect to host ... port ...: Operation timed out` while the training script is still running. Wait 1-2 min and retry with `ConnectTimeout=60`. If 5 retries all fail, check pod status via the GraphQL API — `desiredStatus: RUNNING` with `runtime: null` for ~1 min is a transient blip, not a crash.

### 9b. `pkill -9` doesn't kill orphan procs

Background processes spawned via `nohup ... & disown` and then their PARENTS killed often leave nvcc / huggingface-cli / vLLM-engine zombies holding GPU memory. Use `pkill -9 -f <pattern>` to nuke all matching processes. To verify clean state before launching: `nvidia-smi --query-gpu=memory.used --format=csv,noheader` should return `0 MiB` (or `1 MiB`).

### 9c. Heredoc commands in SSH sometimes truncate

When running `ssh root@... 'cat > /workspace/foo.sh <<EOF ... EOF'`, the heredoc can be eaten by SSH or by zsh quoting. Verify with `ssh root@... 'ls -la /workspace/foo.sh'`. If missing, use `printf '%s\n' '#!/bin/bash' 'line1' 'line2' > /workspace/foo.sh` instead.

### 9d. HF Hub rate-limiting was the worst session-1 failure

Symptoms: download stalls at random partial-file sizes, multiple concurrent `huggingface-cli` processes, .incomplete files growing slowly or not at all. **Mitigation**: use the HF premium token (this session has it). Also set `HF_HUB_ENABLE_HF_TRANSFER=1` for the first download to use parallel chunks; turn it OFF for resumes (`unset HF_HUB_ENABLE_HF_TRANSFER`) — session 1 found hf_transfer reliable for initial downloads but flaky on resumes.

### 9e. Pod gets stuck and Jupyter is "taking longer than expected"

This shows up in the RunPod web UI when the container hasn't finished booting. Wait 2-3 min after provision before attempting SSH. If still failing after 5 min, terminate and re-provision (the pod's sshd init script probably failed; usually a different machine works).

### 9f. `enable_input_require_grads` needed for LoRA + grad_checkpoint without 4-bit

If `--no-quant --grad-checkpoint` is used (bf16 base + gradient checkpointing), the model's input embeddings need `requires_grad` set so the LoRA gradients can flow back. Session 1 added this in `02_sft.py`:

```python
if args.no_quant and args.grad_checkpoint:
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
```

If you see `RuntimeError: element 0 of tensors does not require grad`, this is the cause. Already patched in our `02_sft.py`; if you regenerate the script, keep this.

### 9g. TRL 0.21's "Converting train dataset to ChatML" log is misleading

When `dataset_text_field="text"` is set in SFTConfig, TRL does NOT actually apply the chat template despite logging "Converting to ChatML". Switching to `messages` field (per §6b) is the only way to actually wrap in chat template at SFT time.

### 9h. `save_total_limit=1` deletes earlier checkpoints

Session 1's original config had this; we lost all but the last checkpoint mid-run. The patched `02_sft.py` uses `save_total_limit=10` (effectively unlimited for 4 epochs). Keep it that way.

### 9i. RL adapter weights need to be in the rsync-back filter

Session 1's terminal failure: I rsynced back `eval.json` and `trainer_state.json` but NOT `adapter_model.safetensors`. When the pod was terminated, the actual LoRA weights were gone forever. Session 2 must rsync `--include='*/adapter_model.safetensors'` explicitly (see §8).

---

## 10. Pre-flight checklist (run through before launching any pod)

Tick each before kickoff:

- [ ] **RunPod balance ≥ $60** (current spend estimate $45 + margin)
- [ ] **`.env` has `RUNPOD_API_KEY=` and `HF_ACCESS_TOKEN=hf_...`**
- [ ] **Local disk ≥ 10 GB free** (rsync needs scratch; session 1 hit 100% disk twice)
- [ ] **SSH key at `~/.ssh/id_ed25519`** matches the pubkey on the RunPod account
- [ ] **Code changes applied to local repo**:
  - [ ] `runpod/04_eval.py` has `--chat-template` flag (§6a)
  - [ ] `runpod/02_sft.py` uses `messages` field + `BrierOnValCallback` (§6b)
  - [ ] `runpod/03_rl.py` has `--adapter` optional + chat-template wrap + vllm tuning (§6c)
- [ ] **E4 data split prepped locally**: `data/training_E4/train.jsonl` + `val.jsonl` exist
- [ ] **Choose parallel or sequential** mode and decide pod count

If you're parallel: provision four pods at the same time (each gets the same setup script). If sequential: one pod, run experiments in order without terminating between.

---

## 11. Total spend forecast

| item | parallel | sequential |
|---|---:|---:|
| E1 pod time | 0.5 h × $1.49 = $0.75 | (folded into single pod) |
| E2 + E3 pod time | 5 h × $4.39 = $22 | (folded) |
| E4 pod time | 2 h × $4.39 = $9 | (folded) |
| E5 pod time | 3 h × $4.39 = $13 | (folded) |
| single H200 (sequential only) | — | 10.5 h × $4.39 = $46 |
| single A100 for E1 (sequential only) | — | 0.5 h × $1.49 = $0.75 |
| **total** | **~$45** | **~$47** |
| **wall-clock** | **~5 h** | **~10.5 h** |

**Recommendation: parallel.** Cost difference is negligible (<$3), wall-clock is half. The only reason to go sequential is supply constraints on the H200 SKU.

---

## 12. What you (the executing agent) should report back

After all experiments complete, write a single report at `runpod/SESSION_REPORT_V2.md` containing:

1. **Headline table** — Brier + parse-rate for all session-1 + session-2 conditions, on the same scale.
2. **Per-hypothesis verdict** — one paragraph each on H1, H2, H3.
3. **Surprise findings** — anything you didn't expect from the literature or from the plan.
4. **Recommended next experiments** — three at most, in priority order.
5. **Plots** — extend `runpod/make_plots.py` to include the new session-2 conditions, save under `runpod/plots/v2/`.
6. **Cost actuals** — what you actually spent vs the $45 forecast.

Total report length: aim for 1500 words + plots. Match the style of `runpod/WRITEUP.md`.

---

*This plan was written to be executable by a fresh LLM agent without prior conversational context. Read §0 (Required reading) and §1 (Context) before doing anything. The motivation for every command and every hyperparameter is in the plan; if anything is unclear, the answer is in the cited references.*
