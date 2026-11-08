"""Diagnostic-only memory accounting over the unchanged multi-FULL worker."""
from collections import defaultdict
from pathlib import Path
import functools,json,os,threading,time
import torch
from multigraph_worker import MultiShapeAuditWorker


class MemoryAuditWorker(MultiShapeAuditWorker):
    def load_model(self,*args,**kwargs):
        result=super().load_model(*args,**kwargs)
        if os.environ.get('KV_MEMORY_PROBE')!='1':return result
        root=Path(os.environ['MULTIGRAPH_DIR']);device=self.model_runner.device
        records=[];lock=threading.RLock();reset_id='startup';capture_ready=threading.Event()

        def sample(event,pools=False):
            with lock:
                free,total=torch.cuda.mem_get_info(device)
                allocated=torch.cuda.memory_allocated(device);reserved=torch.cuda.memory_reserved(device)
                row=dict(epoch=time.time(),event=event,reset_id=reset_id,
                    phase=(root/'phase.txt').read_text().strip() if (root/'phase.txt').exists() else 'startup',
                    cuda_free_bytes=free,cuda_total_bytes=total,allocated_bytes=allocated,reserved_bytes=reserved,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    non_torch_bytes=max(0,total-free-reserved))
                if pools:
                    groups=defaultdict(lambda:dict(total=0,allocated=0,active=0,segments=0))
                    snapshots=torch.cuda.memory_snapshot();has_pool_ids=all('segment_pool_id' in v for v in snapshots)
                    for segment in snapshots:
                        key=','.join(map(str,segment.get('segment_pool_id',(0,0))))
                        group=groups[key];group['segments']+=1
                        for key2,key3 in [('total','total_size'),('allocated','allocated_size'),('active','active_size')]:group[key2]+=segment.get(key3,0)
                    row.update(pools=dict(groups),pool_ids_available=has_pool_ids)
                records.append(row)
                temp=root/'memory-probe.tmp';temp.write_text(json.dumps(records,indent=2));temp.replace(root/'memory-probe.json')
                return row

        original_capture=self.model_runner.capture_model
        @functools.wraps(original_capture)
        def capture(*a,**kw):
            sample('before_capture',True);value=original_capture(*a,**kw);sample('after_capture',True);capture_ready.set();return value
        self.model_runner.capture_model=capture

        def background():
            nonlocal reset_id
            capture_ready.wait()
            torch.cuda.set_device(device);seen=None
            while True:
                try:
                    path=root/'memory-command.json'
                    command=json.loads(path.read_text()) if path.exists() else None
                    if command and command['id']!=seen:
                        seen=command['id'];before=sample('before_'+seen,True)
                        if command['command'] in ['trim_and_reset','reset_peaks']:
                            torch.cuda.synchronize(device)
                            if command['command']=='trim_and_reset':torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats(device);reset_id=seen
                        after=sample('after_'+seen,True)
                        dest=root/('memory-result-'+seen+'.json');tmp=dest.with_suffix('.tmp')
                        tmp.write_text(json.dumps(dict(command=command,before=before,after=after),indent=2));tmp.replace(dest)
                    sample('sample')
                except Exception as error:
                    print('KV_MEMORY_PROBE_ERROR',repr(error),flush=True)
                time.sleep(1)
        sample('after_load_model',True)
        threading.Thread(target=background,daemon=True).start()
        return result
