"""Build non-editable wheels in temporary copies, preserving tracked metadata."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offline", action="store_true", help="Use only cached build dependencies")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fbsim-wheel-build-") as temporary:
        for name in ("fbsim-core", "fbsim-benchmark"):
            source = root / "packages" / name
            copied = Path(temporary) / name
            shutil.copytree(source, copied, ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"))
            command = ["uv", "build", "--wheel", "--no-sources", "--python", sys.executable, "--out-dir", str(output)]
            if args.offline:
                command.append("--offline")
            subprocess.run([*command, str(copied)], cwd=temporary, check=True)


if __name__ == "__main__":
    main()
