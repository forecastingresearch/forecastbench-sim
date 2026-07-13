#!/usr/bin/env python3
"""Minimal Fireworks on-demand deployment manager for the step-0 eval.

Small catalog models (qwen3-4b, qwen3-8b, llama-v3p1-8b-instruct) are tunable but
not serverless — inference requires an on-demand deployment billed per GPU-hour.
Pattern cribbed from slop-forensics' fireworks_control.py.

Usage (env: FIREWORKS_API_KEY, FIREWORKS_ACCOUNT_ID):
  uv run python scripts/uplift_v2/fw_deploy.py create qwen3-4b-instruct-2507
  uv run python scripts/uplift_v2/fw_deploy.py status cb-qwen3-4b-instruct-2507
  uv run python scripts/uplift_v2/fw_deploy.py delete cb-qwen3-4b-instruct-2507
  uv run python scripts/uplift_v2/fw_deploy.py list
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://api.fireworks.ai/v1"


def _req(method: str, path: str, body: dict | None = None, params: dict | None = None):
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    acct = os.environ["FIREWORKS_ACCOUNT_ID"]
    cmd = sys.argv[1]

    if cmd == "list":
        code, d = _req("GET", f"/accounts/{acct}/deployments")
        for dep in d.get("deployments", []):
            print(dep["name"], "|", dep.get("baseModel"), "|", dep.get("state"))
        if not d.get("deployments"):
            print("(no deployments)")
        return 0

    model_or_dep = sys.argv[2]

    if cmd == "create":
        model = f"accounts/fireworks/models/{model_or_dep}"
        dep_id = f"cb-{model_or_dep}"[:40]
        body = {
            "baseModel": model,
            "displayName": f"civbench step0 {model_or_dep}"[:63],
            "description": "civbench uplift-v2 step-0 baseline eval (temporary)",
            "minReplicaCount": 0,
            "maxReplicaCount": 1,
            "acceleratorCount": 1,
            "acceleratorType": os.getenv("FIREWORKS_ACCELERATOR_TYPE",
                                         "NVIDIA_H100_80GB"),
            "precision": "BF16",
            "enableSessionAffinity": True,
        }
        code, d = _req("POST", f"/accounts/{acct}/deployments", body,
                       {"deploymentId": dep_id})
        print(f"create -> HTTP {code}")
        if code >= 400:
            print(json.dumps(d, indent=1)[:800])
            return 1
        name = d.get("name", f"accounts/{acct}/deployments/{dep_id}")
        print("deployment:", name)
        # wait for READY
        for _ in range(60):
            code, d = _req("GET", f"/accounts/{acct}/deployments/{dep_id}")
            state = d.get("state")
            print("  state:", state)
            if state == "READY":
                print(f"\nUse as model id for inference:\n  {d.get('baseModel')}"
                      f"#{name}")
                return 0
            if state in ("FAILED", "DELETING"):
                print(json.dumps(d, indent=1)[:800])
                return 1
            time.sleep(15)
        print("timed out waiting for READY (still creating; check status later)")
        return 1

    dep_id = model_or_dep
    if cmd == "status":
        code, d = _req("GET", f"/accounts/{acct}/deployments/{dep_id}")
        print(json.dumps({k: d.get(k) for k in
                          ("name", "baseModel", "state", "acceleratorType",
                           "minReplicaCount", "maxReplicaCount")}, indent=1))
        return 0
    if cmd == "delete":
        code, d = _req("DELETE", f"/accounts/{acct}/deployments/{dep_id}",
                       params={"ignoreChecks": "true"})
        print(f"delete -> HTTP {code}")
        if code >= 400:
            print(json.dumps(d, indent=1)[:400])
        # verify
        time.sleep(3)
        code, d = _req("GET", f"/accounts/{acct}/deployments/{dep_id}")
        print("post-delete state:", d.get("state", d.get("code", "GONE")))
        return 0
    print(f"unknown command {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
