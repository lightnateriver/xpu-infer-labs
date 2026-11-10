"""Fetch the eight verified upstream model files at a pinned public revision."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[1]/'environment/upstream.json').read_text())
    for name,expected in manifest['files'].items():
        output=a.destination/name
        if output.is_file():
            payload=output.read_bytes()
        else:
            url='https://raw.githubusercontent.com/gen-ai-team/kandinsky-video-tools/'+manifest['revision']+'/'+name
            with urllib.request.urlopen(url,timeout=60) as response:payload=response.read()
        if hashlib.sha256(payload).hexdigest()!=expected:
            raise ValueError('Upstream file hash mismatch: '+name)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_bytes(payload)
    (a.destination/'VERIFIED.json').write_text(json.dumps(manifest,indent=2))
    print('Verified',len(manifest['files']),'upstream model files')

if __name__=='__main__':main()
