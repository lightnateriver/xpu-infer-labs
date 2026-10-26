"""Count actual vLLM CUDA graph calls using per-process shared memory, without GPU synchronization."""
import mmap
import os
from pathlib import Path
import struct
from functools import wraps

NAMES=['capture_FULL','capture_PIECEWISE','replay_FULL','replay_PIECEWISE',
       'fallback_NONE','delegation_other_mode','fallback_no_context','other']
_pid=None
_map=None
_counts=None

def _ensure():
 global _pid,_map,_counts
 if _pid==os.getpid():return
 if _map is not None:_map.close()
 _pid=os.getpid();_counts=[0]*len(NAMES)
 namespace=os.stat('/proc/self/ns/pid').st_ino
 init_start=Path('/proc/1/stat').read_text().rsplit(')',1)[1].split()[19]
 path=Path('/dev/shm')/f'codex-omni-graph-audit-{namespace}-{init_start}-{_pid}.bin'
 with path.open('w+b') as stream:
  stream.truncate(8*len(NAMES));_map=mmap.mmap(stream.fileno(),8*len(NAMES))

def record(event,mode=None):
 _ensure()
 name=event+'_'+mode if mode is not None else event
 index=NAMES.index(name) if name in NAMES else NAMES.index('other')
 _counts[index]+=1
 struct.pack_into('<Q',_map,index*8,_counts[index])

_native_v2_installed=False

def _instrument_native_v2(manager,breakable,graph_type):
 capture=manager.capture
 @wraps(capture)
 def capture_audit(self,*args,**kwargs):
  before={id(value) for value in self.graphs.values()}
  try:return capture(self,*args,**kwargs)
  finally:
   for graph in self.graphs.values():
    if id(graph) not in before:record('capture','FULL')
 manager.capture=capture_audit
 replay=manager.run_fullgraph
 @wraps(replay)
 def replay_audit(self,*args,**kwargs):
  result=replay(self,*args,**kwargs)
  record('replay','FULL')
  return result
 manager.run_fullgraph=replay_audit
 dispatch=manager.dispatch
 @wraps(dispatch)
 def dispatch_audit(self,*args,**kwargs):
  result=dispatch(self,*args,**kwargs)
  if result.cg_mode.name=='NONE':record('fallback_NONE')
  return result
 manager.dispatch=dispatch_audit
 end_segment=breakable._end_segment
 @wraps(end_segment)
 def end_segment_audit(self,*args,**kwargs):
  before=self.num_graphs
  result=end_segment(self,*args,**kwargs)
  for _ in range(self.num_graphs-before):record('capture','PIECEWISE')
  return result
 breakable._end_segment=end_segment_audit
 # Each bound CUDA graph replay in segments is counted after it succeeds.
 @wraps(breakable.replay)
 def segment_replay_audit(self):
  for operation in self.segments:
   operation()
   if getattr(operation,'__name__',None)=='replay' and isinstance(getattr(operation,'__self__',None),graph_type):
    record('replay','PIECEWISE')
 breakable.replay=segment_replay_audit

def install_native_v2():
 global _native_v2_installed
 if _native_v2_installed:return
 from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
 from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
 import torch
 _instrument_native_v2(CudaGraphManager,BreakableCUDAGraphCapture,torch.cuda.CUDAGraph)
 _native_v2_installed=True
 import json
 namespace=os.stat('/proc/self/ns/pid').st_ino
 init_start=Path('/proc/1/stat').read_text().rsplit(')',1)[1].split()[19]
 (Path('/dev/shm')/f'codex-omni-v2-audit-{namespace}-{init_start}-{os.getpid()}.json').write_text(json.dumps(dict(
   installed=True,methods=['CudaGraphManager.capture','CudaGraphManager.run_fullgraph','CudaGraphManager.dispatch',
                           'BreakableCUDAGraphCapture._end_segment','BreakableCUDAGraphCapture.replay'],
   semantics='successful registered FULL captures and replay calls; breakable PIECEWISE segments; NONE includes disabled graph and prefill dispatches')))

_ensure()
