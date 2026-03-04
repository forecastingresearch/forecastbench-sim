#!/usr/bin/env python3
"""Quick environment sanity checks for LoRA/QLoRA training."""

from __future__ import annotations

import importlib
import json

import torch


def pkg_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostics script
        return f"MISSING ({exc})"


def main() -> None:
    info = {
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "packages": {
            "transformers": pkg_version("transformers"),
            "trl": pkg_version("trl"),
            "peft": pkg_version("peft"),
            "bitsandbytes": pkg_version("bitsandbytes"),
            "datasets": pkg_version("datasets"),
            "accelerate": pkg_version("accelerate"),
        },
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
