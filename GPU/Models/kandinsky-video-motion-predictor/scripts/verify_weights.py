"""Verify the two existing checkpoint files; never downloads weights."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-dir',type=Path,required=True)
a=p.parse_args()
manifest=json.loads((Path(__file__).resolve().parents[1]/'environment/upstream.json').read_text())
for name,expected in manifest['weights'].items():
    h=hashlib.sha256()
    with (a.model_dir/name).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError('Weight hash mismatch: '+name)
print('Verified both checkpoint files')
