"""Check the recorded package versions without creating a CUDA context."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def check():
    reference = json.loads((ROOT/'environment/versions.json').read_text())
    actual = {}
    for package, expected in reference['packages'].items():
        actual[package] = importlib.metadata.version(package)
        assert actual[package] == expected, f'{package}: expected {expected}, got {actual[package]}'
    assert sys.version_info[:2] == (3, 10), 'Recorded Python version is 3.10'
    for binary in ['nvidia-smi', 'ffmpeg', 'ffprobe']:
        assert shutil.which(binary), f'Missing executable: {binary}'
    return dict(python=sys.version.split()[0], packages=actual, recorded_versions_match=True)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
