#!/usr/bin/env python3
"""Stopping-rule + checkpoint-hygiene sidecar for train_arm_v2.sh.

Watches the verl console log for validation metric lines, and:
  1. EARLY STOP: if the best val score hasn't improved for PATIENCE epochs,
     kills the training process group (checkpoints already on disk).
  2. DISK HYGIENE: after each new checkpoint, deletes optimizer/extra states
     from all checkpoints except the most recent (only model shards are needed
     for eval/deploy; optimizer only needed to resume from the latest).
     Also deletes model shards of checkpoints that are neither best-val,
     most recent, nor second-most-recent.

verl val lines look like: "val/test_score/<data_source>:0.8123" (or metric
dicts printed at test_freq steps). We treat max-so-far as the target metric
(score = 1 - Brier-style, higher better).

Usage: python monitor_val.py --log /workspace/logs/EXP.log \
         --ckpt-dir /workspace/ckpts/EXP --steps-per-epoch 26 \
         --test-freq 13 --patience 3 --pgid <train pgid>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

HF_REPO = os.environ.get("HF_BACKUP_REPO", "")  # e.g. user/civbench-stage3


def hf_upload(local: Path, dest: str):
    """Best-effort upload to the private HF backup repo. Never raises."""
    if not HF_REPO:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        if local.is_dir():
            api.upload_folder(folder_path=str(local), path_in_repo=dest,
                              repo_id=HF_REPO, repo_type="model")
        else:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=dest,
                            repo_id=HF_REPO, repo_type="model")
        print(f"[monitor] backed up {local} -> {HF_REPO}/{dest}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[monitor] HF upload failed (non-fatal): {e}", flush=True)

VAL_RE = re.compile(r"val/test_score[^:\s]*[:=]\s*([0-9.]+)")
STEP_RE = re.compile(r"step:(\d+)")


def checkpoints(d: Path):
    return sorted((p for p in d.glob("global_step_*") if p.is_dir()),
                  key=lambda p: int(p.name.split("_")[-1]))


def prune(ckpt_dir: Path, best_step: int | None):
    if best_step is None:
        return  # never prune before the val pipeline has proven itself (#8)
    cks = checkpoints(ckpt_dir)
    if len(cks) < 3:
        return
    keep_full = {cks[-1].name, cks[-2].name}         # resume points (#10)
    keep_model = set(keep_full)
    if len(cks) > 1:
        keep_model.add(cks[-2].name)
    if best_step is not None:
        keep_model.add(f"global_step_{best_step}")
    for c in cks:
        # strip optimizer/extra state everywhere except the resume point
        if c.name not in keep_full:
            for sub in c.rglob("optim*"):
                (shutil.rmtree(sub, ignore_errors=True) if sub.is_dir()
                 else sub.unlink(missing_ok=True))
            for sub in c.rglob("extra_state*"):
                (shutil.rmtree(sub, ignore_errors=True) if sub.is_dir()
                 else sub.unlink(missing_ok=True))
        # drop model weights of checkpoints we'll never use
        if c.name not in keep_model:
            shutil.rmtree(c, ignore_errors=True)
            print(f"[monitor] pruned {c.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--steps-per-epoch", type=int, required=True)
    ap.add_argument("--test-freq", type=int, required=True)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--pgid", type=int, required=True)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    best, best_step, last_seen_step = -1.0, None, 0
    vals = []  # (step, score)
    pos = 0
    print(f"[monitor] watching {args.log}, patience={args.patience} epochs "
          f"({args.patience * args.steps_per_epoch} steps)", flush=True)
    while True:
        time.sleep(30)
        try:
            with open(args.log) as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except FileNotFoundError:
            continue
        steps_all = [int(x) for x in STEP_RE.findall(chunk)]
        if steps_all:
            last_seen_step = max(last_seen_step, max(steps_all))
        for m in VAL_RE.finditer(chunk):
            score = float(m.group(1))
            # attribute the score to the latest step BEFORE the val line (#9)
            prior = [int(x) for x in STEP_RE.findall(chunk[:m.start()])]
            step_at = max([last_seen_step] + prior)
            vals.append((step_at, score))
            last_seen_step = step_at
            if score > best + 1e-4:
                best, best_step = score, step_at
            print(f"[monitor] val at step~{step_at}: {score:.4f} "
                  f"(best {best:.4f} @ {best_step})", flush=True)
            json.dump({"vals": vals, "best": best, "best_step": best_step},
                      open(ckpt_dir / "val_history.json", "w"))
            hf_upload(ckpt_dir / "val_history.json",
                      f"{ckpt_dir.name}/val_history.json")

        # retrying best-checkpoint backup every cycle until it succeeds (#9)
        if best_step is not None:
            cand = ckpt_dir / f"global_step_{best_step}"
            marker = ckpt_dir / f".uploaded_{best_step}"
            if cand.exists() and not marker.exists():
                hf_upload(cand, f"{ckpt_dir.name}/best_global_step_{best_step}")
                marker.touch()
        prune(ckpt_dir, best_step)
        # stopping rule: no new best for `patience` epochs' worth of steps
        if (best_step is not None and
                last_seen_step - best_step >= args.patience * args.steps_per_epoch
                and len(vals) >= args.patience + 1):
            print(f"[monitor] EARLY STOP: no improvement since step {best_step} "
                  f"(now ~{last_seen_step}). Killing training pgid {args.pgid}.",
                  flush=True)
            try:
                os.killpg(args.pgid, signal.SIGTERM)
                time.sleep(60)
                os.killpg(args.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            json.dump({"vals": vals, "best": best, "best_step": best_step,
                       "early_stopped": True},
                      open(ckpt_dir / "val_history.json", "w"))
            hf_upload(ckpt_dir / "val_history.json",
                      f"{ckpt_dir.name}/val_history.json")
            hf_upload(Path(args.log), f"{ckpt_dir.name}/train.log")
            return
        # training ended on its own?
        try:
            os.killpg(args.pgid, 0)
        except ProcessLookupError:
            print("[monitor] training process gone; exiting.", flush=True)
            hf_upload(ckpt_dir / "val_history.json",
                      f"{ckpt_dir.name}/val_history.json")
            hf_upload(Path(args.log), f"{ckpt_dir.name}/train.log")
            return


if __name__ == "__main__":
    main()
