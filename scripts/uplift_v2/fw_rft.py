#!/usr/bin/env python3
"""Fireworks RFT job manager for the A2 three-way experiment.

Subcommands:
  upload   <dataset_id> <path.jsonl>      create + upload + validate a dataset
  launch   <job_prefix> <dataset_id> [--epochs N] [--lr F] [--seed N]
           create an RFT job mirroring calibration job nnwolhuh's config
           (qwen3-4b, G=8, temp 1.0, maxOut 8192, ctx 24576, LoRA r8)
  status   [job_id]                       list jobs / show one job
  cancel   <job_id>

Env: FIREWORKS_API_KEY, FIREWORKS_ACCOUNT_ID.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = "https://api.fireworks.ai/v1"
EVALUATOR = "test-forecast-test-forecast"


def req(method: str, path: str, body: dict | None = None,
        raw_url: str | None = None, headers: dict | None = None,
        data: bytes | None = None):
    url = raw_url or f"{BASE}{path}"
    h = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
         "Content-Type": "application/json",
         "User-Agent": "civbench-fw-rft/1.0"}
    if headers:
        h = headers
    r = urllib.request.Request(
        url, method=method,
        data=data if data is not None else (
            json.dumps(body).encode() if body is not None else None),
        headers=h)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            txt = resp.read()
            return resp.status, (json.loads(txt) if txt.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read() or b"{}"
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw.decode(errors="replace")[:600]}


def acct() -> str:
    return os.environ["FIREWORKS_ACCOUNT_ID"]


def upload(dataset_id: str, path: str) -> int:
    p = Path(path)
    n = sum(1 for _ in open(p))
    size = p.stat().st_size
    code, d = req("POST", f"/accounts/{acct()}/datasets", {
        "datasetId": dataset_id,
        "dataset": {"displayName": dataset_id, "userUploaded": {},
                    "exampleCount": str(n)}})
    if code >= 400 and "already exists" not in json.dumps(d):
        print(f"create failed {code}: {json.dumps(d)[:300]}")
        return 1
    code, d = req("POST",
                  f"/accounts/{acct()}/datasets/{dataset_id}:getUploadEndpoint",
                  {"filenameToSize": {"dataset.jsonl": size}})
    if code >= 400:
        print(f"getUploadEndpoint failed {code}: {json.dumps(d)[:300]}")
        return 1
    url = d["filenameToSignedUrls"]["dataset.jsonl"]
    code, _ = req("PUT", "", raw_url=url, data=p.read_bytes(), headers={
        "Content-Type": "application/octet-stream",
        "x-goog-content-length-range": f"{size},{size}"})
    if code >= 400:
        print(f"PUT failed {code}")
        return 1
    code, d = req("POST",
                  f"/accounts/{acct()}/datasets/{dataset_id}:validateUpload", {})
    code, d = req("GET", f"/accounts/{acct()}/datasets/{dataset_id}")
    print(f"dataset {dataset_id}: state={d.get('state')} examples={n}")
    return 0 if d.get("state") == "READY" else 1


def launch(job_id: str, dataset_id: str, epochs: int, lr: float,
           output_model: str) -> int:
    body = {
            "dataset": f"accounts/{acct()}/datasets/{dataset_id}",
            "evaluator": f"accounts/{acct()}/evaluators/{EVALUATOR}",
            "evalAutoCarveout": False,
            "chunkSize": 200,
            "trainingConfig": {
                "baseModel": "accounts/fireworks/models/qwen3-4b",
                "epochs": epochs,
                "learningRate": lr,
                "loraRank": 8,
                "maxContextLength": 24576,
                "batchSize": 32768,
            },
            "inferenceParameters": {
                "maxOutputTokens": 8192,
                "responseCandidatesCount": 8,
                "temperature": 1.0,
                "topP": 1.0,
            },
    }
    code, d = req("POST",
                  f"/accounts/{acct()}/reinforcementFineTuningJobs"
                  f"?reinforcementFineTuningJobId={job_id}", body)
    if code >= 400:
        print(f"launch failed {code}: {json.dumps(d)[:800]}")
        return 1
    print(f"launched: {d.get('name')} state={d.get('state')}")
    return 0


def status(job_id: str | None) -> int:
    if job_id:
        code, d = req("GET",
                      f"/accounts/{acct()}/reinforcementFineTuningJobs/{job_id}")
        keep = {k: d.get(k) for k in
                ("name", "state", "status", "createTime", "completedTime",
                 "dataset", "outputModel", "jobProgress")}
        print(json.dumps(keep, indent=1))
    else:
        code, d = req("GET",
                      f"/accounts/{acct()}/reinforcementFineTuningJobs?pageSize=50")
        for j in d.get("reinforcementFineTuningJobs", []):
            print(j["name"].split("/")[-1], "|", j.get("state"), "|",
                  j.get("createTime", "")[:16], "|",
                  j.get("dataset", "").split("/")[-1])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload")
    u.add_argument("dataset_id")
    u.add_argument("path")
    l = sub.add_parser("launch")
    l.add_argument("job_id")
    l.add_argument("dataset_id")
    l.add_argument("--epochs", type=int, default=4)
    l.add_argument("--lr", type=float, default=1e-4)
    l.add_argument("--output-model", required=True)
    s = sub.add_parser("status")
    s.add_argument("job_id", nargs="?")
    c = sub.add_parser("cancel")
    c.add_argument("job_id")
    args = ap.parse_args()

    if args.cmd == "upload":
        return upload(args.dataset_id, args.path)
    if args.cmd == "launch":
        return launch(args.job_id, args.dataset_id, args.epochs, args.lr,
                      args.output_model)
    if args.cmd == "status":
        return status(args.job_id)
    if args.cmd == "cancel":
        code, d = req("POST",
                      f"/accounts/{acct()}/reinforcementFineTuningJobs/"
                      f"{args.job_id}:cancel", {})
        print(code, json.dumps(d)[:300])
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
