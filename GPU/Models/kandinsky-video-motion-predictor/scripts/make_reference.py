"""Generate private serial GPU references from caller-owned physical MP4 inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(line) for line in a.manifest.read_text().splitlines() if line.strip()]
    if not rows or len({r['clip_id'] for r in rows})!=len(rows):raise ValueError('Empty manifest or duplicate clip_id')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'validation').mkdir()
    sys.path.insert(0,str(a.code))
    import numpy as np
    import torch
    from motion_only import MotionScorer
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    scorer=MotionScorer(device='cuda:0',max_segments=16)
    records=[]
    with (a.output/'cases.jsonl').open('w') as file:
        for i,row in enumerate(rows):
            start=time.perf_counter();x=scorer.prep({'clip_path':row['path']});prep=time.perf_counter()-start
            torch.cuda.synchronize();start=time.perf_counter();score=scorer.infer_batch([x])[0];torch.cuda.synchronize()
            record=dict(clip_id=row['clip_id'],shape=list(x.shape),input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                        scores=score,prepare_s=prep,infer_s=time.perf_counter()-start)
            file.write(json.dumps(record)+'\n');file.flush();records.append(record)
            if i==0:
                np.save(a.output/'validation/input.npy',x.numpy(),allow_pickle=False)
                (a.output/'validation/tensor-result.json').write_text(json.dumps(dict(scores=score)))
    (a.output/'scores-array.json').write_text(json.dumps(records,indent=2))
    print('Saved private reference records:',len(records))

if __name__=='__main__':main()
