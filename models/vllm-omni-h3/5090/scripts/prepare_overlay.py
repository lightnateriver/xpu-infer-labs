#!/usr/bin/env python3
"""Build a private import overlay from the pinned vLLM-Omni package."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "patches"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("v1-sp8", "sp4-alternate"))
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source", type=Path, required=True, help="Clean pinned vLLM-Omni checkout")
    args = parser.parse_args()
    target = args.destination.expanduser().resolve()
    if target.exists():
        parser.error(f"destination already exists: {target}")
    for name in ("vllm", "vllm-omni"):
        version = importlib.metadata.version(name)
        if version != "0.32.0":
            parser.error(f"{name} must be 0.32.0, found {version}")
    spec = importlib.util.find_spec("vllm_omni")
    if not spec or not spec.origin:
        parser.error("vllm_omni package not found")
    package = args.source.expanduser().resolve() / "vllm_omni"
    if not (package / "__init__.py").is_file():
        parser.error("--source must contain vllm_omni/")
    hashes = json.loads((PATCHES / "patched-sha256.json").read_text(encoding="utf-8-sig"))[args.profile]
    patch = PATCHES / f"{args.profile}.patch"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="omni-h3-") as scratch:
        staged = Path(scratch) / "overlay"
        shutil.copytree(package, staged / "vllm_omni", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        version_file = Path(spec.origin).with_name("_version.py")
        if version_file.is_file():
            shutil.copyfile(version_file, staged / "vllm_omni" / "_version.py")
        for flags in (["--check"], []):
            subprocess.run(["git", "apply", "--no-index", *flags, str(patch)], cwd=staged, check=True)
        for name, expected in hashes.items():
            actual = hashlib.sha256((staged / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Patched source hash mismatch: {name}")
        shutil.copyfile(ROOT / "configs" / f"{args.profile}.mode", staged / "mode.txt")
        shutil.move(staged, target)
    print(f"Overlay ready: {target}")


if __name__ == "__main__":
    main()
