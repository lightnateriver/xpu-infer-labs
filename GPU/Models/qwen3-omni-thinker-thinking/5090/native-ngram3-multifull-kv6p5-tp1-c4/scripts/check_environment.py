"""Check the recorded package versions without creating a CUDA context."""
import importlib.metadata
import hashlib
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
    distribution=importlib.metadata.distribution('vllm')
    source_hashes=json.loads((ROOT/'environment/package-source-manifest.json').read_text())
    for filename,digest in source_hashes.items():
        path=Path(distribution.locate_file(filename))
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'Recorded source mismatch: '+filename
    return dict(python=sys.version.split()[0], packages=actual, recorded_versions_match=True, source_hashes_match=True)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
