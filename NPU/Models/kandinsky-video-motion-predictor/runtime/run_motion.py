"""Standalone resident batch runner using the retained motion pipeline options."""
import argparse
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time

PREP=None
def process_memory(pid):
    try:
        values=dict(line.split(':',1) for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines() if ':' in line)
        return dict(pss_kib=int(values['Pss'].split()[0]),rss_kib=int(values['Rss'].split()[0]))
    except (OSError,KeyError):return {}
def worker_init(code,ready,warm):
    global PREP
    os.environ['ASCEND_RT_VISIBLE_DEVICES']='';os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    import torch_npu
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(code))
    from motion_pipeline import Preprocessor
    PREP=Preprocessor()
    PREP(warm,{})
    ready.put(dict(pid=os.getpid(),npu_initialized=torch.npu.is_initialized()))

def probe():return os.getpid()

def prepare(item,config,verify):
    t=time.perf_counter()
    tensor=PREP(item,config)
    array=tensor.numpy()
    return dict(index=item['_index'],array=array,prepare_s=time.perf_counter()-t,
                input_sha256=hashlib.sha256(array.tobytes()).hexdigest() if verify else None)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True,help='Original handoff code directory')
    p.add_argument('--manifest',type=Path,required=True,help='JSONL with clip_id/path and optional duration metadata')
    p.add_argument('--config',type=Path)
    p.add_argument('--weights-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-pixels',type=Path,help='Optional original cases.jsonl')
    p.add_argument('--verify-scores',type=Path,help='Optional frozen-single.json NPU reference')
    p.add_argument('--warmup',action='store_true',help='Warm representative shapes and report the cost separately')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    os.environ['QC_CKPT_DIR']=str(a.weights_root)
    sys.path.insert(0,str(a.code))
    import numpy as np
    import torch
    import torch_npu
    from motion_pipeline import OptimizedMotionScorer,validate_config,order_inputs,inference_context
    config=validate_config(json.loads(a.config.read_text()) if a.config else None)
    items=[dict(json.loads(line),_index=i) for i,line in enumerate(a.manifest.read_text().splitlines()) if line.strip()]
    if not items or len({r['clip_id'] for r in items})!=len(items):raise ValueError('Empty manifest or duplicate clip_id')
    pixel_reference={r['clip_id']:r['input_sha256'] for r in map(json.loads,a.verify_pixels.read_text().splitlines())} if a.verify_pixels else {}
    score_reference={r['clip_id']:r['scores'] for r in json.loads(a.verify_scores.read_text())} if a.verify_scores else {}
    torch.set_num_threads(4);torch.npu.set_device(0)
    t=time.perf_counter();scorer=OptimizedMotionScorer(device='npu:0',max_segments=16)
    scorer.head_fast=config['head_fast']
    torch.npu.synchronize();model_load_s=time.perf_counter()-t
    ctx=mp.get_context('spawn');ready=ctx.Queue();pool_start=time.perf_counter()
    warm=min(items,key=lambda r:r.get('duration_s',float('inf')))
    pool=ProcessPoolExecutor(max_workers=config['workers'],mp_context=ctx,initializer=worker_init,
                             initargs=(str(a.code),ready,warm))
    stop=threading.Event();monitor_thread=None;samples=[]
    try:
        probes=[pool.submit(probe) for _ in range(config['workers'])]
        infos=[ready.get(timeout=180) for _ in probes]
        for f in probes:f.result()
        assert all(not info['npu_initialized'] for info in infos)
        pool_ready_s=time.perf_counter()-pool_start
        warmup_s=0
        if a.warmup:
            t=time.perf_counter();examples={}
            for item in items:
                duration=item.get('duration_s')
                frames=int((min(4.0*int(duration),240)//16+1)*16) if duration is not None else 16
                cost=item.get('width',224)*item.get('height',224)*item.get('num_frames',16)
                if frames not in examples or cost<examples[frames][0]:examples[frames]=(cost,item)
            for _,item in examples.values():
                result=pool.submit(prepare,item,config,False).result()
                x=torch.from_numpy(result['array'])
                for _ in range(2):scorer.infer_batch([x])
            torch.npu.synchronize();warmup_s=time.perf_counter()-t
        ordered=order_inputs(items,config);pending={};cursor=0;records=[]
        def monitor():
            pids=[os.getpid()]+[r['pid'] for r in infos]
            with (a.output/'memory.jsonl').open('w') as file:
                while not stop.is_set():
                    values={pid:process_memory(pid) for pid in pids}
                    row=dict(time=time.time(),total_pss_kib=sum(v.get('pss_kib',0) for v in values.values()),pids=values)
                    samples.append(row);file.write(json.dumps(row)+'\n');file.flush();stop.wait(1)
        monitor_thread=threading.Thread(target=monitor);monitor_thread.start()
        def submit():
            nonlocal cursor
            while cursor<len(ordered) and len(pending)<2*config['workers']:
                row=ordered[cursor];cursor+=1
                pending[pool.submit(prepare,row,config,bool(pixel_reference))]=row
        start=time.perf_counter()
        with inference_context(config):
            submit()
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    item=pending.pop(future);result=future.result();array=result.pop('array')
                    if pixel_reference:assert result['input_sha256']==pixel_reference[item['clip_id']],item['clip_id']
                    score=scorer.infer_batch([torch.from_numpy(array)])[0]
                    torch.npu.synchronize()
                    if not all(np.isfinite(v) for v in score.values()):raise ValueError('Non-finite score')
                    if score_reference:assert score==score_reference[item['clip_id']],item['clip_id']
                    result.update(clip_id=item['clip_id'],scores=score)
                    records.append(result);submit()
        seconds=time.perf_counter()-start
        records.sort(key=lambda r:r['index'])
        with (a.output/'scores.jsonl').open('w') as file:
            for result in records:file.write(json.dumps(result)+'\n')
        report=dict(count=len(records),config=config,model_load_s=model_load_s,pool_ready_s=pool_ready_s,warmup_s=warmup_s,
                    request_wall_s=seconds,clips_per_s=len(records)/seconds,model_initializations=1,
                    all_pixel_hashes_exact=bool(pixel_reference),all_scores_exact=bool(score_reference),
                    sampled_peak_pss_kib=max(r['total_pss_kib'] for r in samples),
                    peak_allocated_bytes=torch.npu.max_memory_allocated(),peak_reserved_bytes=torch.npu.max_memory_reserved())
        (a.output/'summary.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:
        stop.set()
        if monitor_thread is not None:monitor_thread.join()
        pool.shutdown(wait=True,cancel_futures=True);ready.close();ready.join_thread()
        scorer.model.to('cpu');torch.npu.empty_cache();torch.npu.synchronize()

if __name__=='__main__':main()
