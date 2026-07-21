#!/bin/bash
# Pod bootstrap for the OpenForecaster-replication A/B. Idempotent.
# Fetches the OF repo, installs it via its own setup.sh (with the interactive
# OpenRouter prompt pre-empted), applies the 2-line reward patch, pulls the
# FB-Sim datasets + scripts from the public civbench branch, downloads models.
set -euxo pipefail

BRANCH="${BRANCH:-whatif-exploration}"
RAW="https://raw.githubusercontent.com/forecastingresearch/forecastbench-sim/$BRANCH"
cd /workspace

if [ ! -d scaling-forecasting-training ]; then
  git clone --depth 1 https://github.com/OpenForecaster/scaling-forecasting-training
fi
cd scaling-forecasting-training
mkdir -p qgen/config
[ -f qgen/config/openrouter_key.py ] || echo 'API_KEY = ""' > qgen/config/openrouter_key.py
bash setup.sh < /dev/null
source forecast/bin/activate

mkdir -p /workspace/data /workspace/logs
for f in train_hard train_dense val_hard val_dense; do
  [ -s /workspace/data/$f.jsonl ] || \
    (curl -fsSL "$RAW/data/uplift_v2/verl_v2/$f.jsonl.gz" | gunzip > /workspace/data/$f.jsonl)
done
wc -l /workspace/data/*.jsonl
python - <<'PYEOF'
import pandas as pd, json
for f in ('train_dense','train_hard','val_dense','val_hard'):
    rows=[json.loads(l) for l in open(f'/workspace/data/{f}.jsonl')]
    pd.DataFrame(rows).to_parquet(f'/workspace/data/{f}.parquet')
    print(f, len(rows))
PYEOF

for f in patch_of_reward.py train_arm_v2.sh eval_checkpoints.py monitor_val.py; do
  curl -fsSL "$RAW/scripts/uplift_v2/runpod/$f" -o /workspace/$f
done

python /workspace/patch_of_reward.py \
  /workspace/scaling-forecasting-training/libraries/verl

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B', local_dir='/workspace/models/Qwen3-4B')
EOF

touch /workspace/BOOTSTRAP_DONE
echo "BOOTSTRAP_DONE"
