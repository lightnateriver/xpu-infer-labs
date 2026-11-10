import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from motion_only import MotionScorer
p=argparse.ArgumentParser();p.add_argument('--video');p.add_argument('--tensor');p.add_argument('--output',required=True);p.add_argument('--save-tensor');p.add_argument('--max-segments',type=int,default=16);a=p.parse_args()
assert bool(a.video)!=bool(a.tensor),'Choose exactly one of --video or --tensor'
t=time.perf_counter();sc=MotionScorer(device='cuda:0',max_segments=a.max_segments);torch.cuda.synchronize();load=time.perf_counter()-t
t=time.perf_counter()
x=torch.from_numpy(np.load(a.tensor,allow_pickle=False)) if a.tensor else sc.prep({'clip_path':a.video})
assert x.dtype==torch.uint8 and x.ndim==4 and tuple(x.shape[1:])==(3,224,224) and len(x)%16==0
prep=time.perf_counter()-t
if a.save_tensor:np.save(a.save_tensor,x.numpy(),allow_pickle=False)
torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();result=sc.infer_batch([x])[0];torch.cuda.synchronize();elapsed=time.perf_counter()-t
assert all(np.isfinite(v) for v in result.values())
o={'input':a.video or a.tensor,'shape':list(x.shape),'scores':result,'seconds':{'load':load,'prepare':prep,'infer':elapsed},'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'status':'complete'}
Path(a.output).write_text(json.dumps(o,indent=2));print(json.dumps(o))
