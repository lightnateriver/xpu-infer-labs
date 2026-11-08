"""Local Docker-host configuration and benchmark isolation helpers."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

PROJECT=Path(__file__).resolve().parents[1]
_LOCK=None

def command(args,timeout=90):
    if args and args[0]=='docker':
        from native_adapter import command_native
        receipt=os.environ.get('BENCH_NATIVE_RECEIPT')
        if not receipt:raise RuntimeError('This release requires a native service receipt')
        return command_native(args,receipt)
    return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=timeout).strip()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)

def load_env(path):
    if not path.exists():return
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        key,separator,value=line.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',key.strip()):
            raise ValueError('Invalid environment assignment')
        value=value.strip()
        if len(value)>=2 and value[0]==value[-1] and value[0] in ['"',"'"]:value=value[1:-1]
        os.environ.setdefault(key.strip(),value)

def gpu_ids(value):
    result=[int(v.strip()) for v in value.split(',')]
    if not result or len(set(result))!=len(result) or any(v<0 for v in result):
        raise ValueError('Specify distinct nonnegative physical GPU indices')
    return result

def inspect(container):
    value=json.loads(command(['docker','inspect',container]))
    return value[0] if isinstance(value,list) else value

def snapshot(container=None):
    raw=command(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
    cards=[dict(gpu=int(r[0]),uuid=r[1].strip(),memory_mib=int(r[2]),utilization=int(r[3])) for r in csv.reader(raw.splitlines())]
    raw=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
    processes=[dict(uuid=r[0].strip(),pid=int(r[1]),memory=r[2].strip()) for r in csv.reader(raw.splitlines()) if len(r)==3]
    own=set()
    if container:
        own={int(v.strip()) for v in command(['docker','top',container,'-eo','pid']).splitlines()[1:] if v.strip().isdigit()}
    return dict(epoch=time.time(),cards=cards,processes=processes,own_pids=sorted(own))

def wait_idle(container,endpoint,headers,gpus,timeout=300):
    info=inspect(container)
    if not info['State']['Running']:raise RuntimeError('Container is not running')
    started_at=info['State']['StartedAt']
    history=[];consecutive=0;deadline=time.monotonic()+timeout
    while consecutive<3:
        sample=snapshot(container)
        selected={v['uuid'] for v in sample['cards'] if v['gpu'] in gpus}
        if len(selected)!=len(gpus):raise RuntimeError('Selected GPUs are missing')
        foreign=[v for v in sample['processes'] if v['uuid'] in selected and v['pid'] not in sample['own_pids']]
        if foreign:raise RuntimeError('Foreign compute process on selected GPUs; nothing was stopped')
        request=urllib.request.Request(endpoint+'/metrics',headers=headers)
        with urllib.request.urlopen(request,timeout=10) as response:metrics=response.read().decode()
        terminal=sum(float(v.rsplit(' ',1)[-1]) for v in metrics.splitlines() if v.startswith('vllm:request_success_total{'))
        running=sum(float(v.rsplit(' ',1)[-1]) for v in metrics.splitlines() if v.startswith('vllm:num_requests_running{'))
        waiting=sum(float(v.rsplit(' ',1)[-1]) for v in metrics.splitlines() if v.startswith('vllm:num_requests_waiting{'))
        log=command(['docker','logs',container],timeout=120)
        marker='Application startup complete.'
        assert marker in log,'No current API startup marker; reject request-count audit'
        log=log.rsplit(marker,1)[1]
        accepted=len(re.findall(r'"POST /v1/chat/completions HTTP/1\.[01]" 200',log))
        finished=len(set(re.findall(r'\[OmniTiming\] req=(\S+)',log)))
        if os.environ.get('BENCH_NATIVE_RECEIPT'):
            # Native vLLM has no OmniTiming records: completed-engine counter
            # parity with accepted HTTP requests is the completion gate.
            finished=terminal
        # Omni's running gauge may retain the last active scheduler sample.
        # Require API acceptance, engine completion and terminal counter parity.
        idle=accepted==finished==terminal and waiting==0 and all(v['utilization']<=3 for v in sample['cards'] if v['gpu'] in gpus)
        sample.update(accepted=accepted,finished=finished,running=running,waiting=waiting,terminal=terminal,idle=idle,foreign=foreign,started_at=started_at)
        history.append(sample);consecutive=consecutive+1 if idle else 0
        if time.monotonic()>deadline:raise RuntimeError('Service did not become idle; request counters or GPU utilization remain active')
        if consecutive<3:time.sleep(2)
    return dict(terminal=terminal,started_at=started_at,samples=history)

def benchmark_setup(kind):
    global _LOCK
    parser=argparse.ArgumentParser(description='Thinking-enabled '+kind+' benchmark on the local Docker/GPU host')
    parser.add_argument('--env-file',type=Path,default=PROJECT/'.env')
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--container')
    parser.add_argument('--endpoint')
    parser.add_argument('--alias')
    parser.add_argument('--gpus')
    parser.add_argument('--name',default=kind)
    parser.add_argument('--concurrency',type=int,default=8)
    parser.add_argument('--phase',choices=['preflight','warmup','measured'] if kind=='caption' else ['warmup','quality'],required=True)
    parser.add_argument('--sample-ids',default='')
    parser.add_argument('--warmup-sample',default='')
    parser.add_argument('--requests',type=int,default=0)
    if kind=='audio':
        parser.add_argument('--manifest',type=Path,default=PROJECT/'data/manifests/audio-256.json')
        parser.add_argument('--protocol',type=Path,default=PROJECT/'configs/audio-classification-protocol.json')
    args=parser.parse_args();load_env(args.env_file)
    args.container=args.container or os.environ.get('CONTAINER_NAME','qwen3-omni-thinker-tp4')
    args.endpoint=(args.endpoint or os.environ.get('API_BASE','http://localhost:8097')).rstrip('/')
    args.alias=args.alias or os.environ.get('MODEL_ALIAS','qwen3-omni-bf16-thinking-tp4')
    args.gpus=gpu_ids(args.gpus or os.environ.get('GPU_IDS','0,1,2,3'))
    if args.concurrency<1:parser.error('--concurrency must be positive')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.name):parser.error('--name must be a simple directory name')
    args.run_dir=args.run_dir.resolve();args.data_dir=args.data_dir.resolve()
    args.run_dir.mkdir(parents=True,exist_ok=True)
    out=args.run_dir/'runs'/args.name/args.phase
    if out.exists() and any(out.iterdir()):raise FileExistsError('Phase output already exists; choose a fresh --name or --run-dir')
    import fcntl
    lock_path=PROJECT/'.local/benchmark.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    _LOCK=lock_path.open('a+')
    fcntl.flock(_LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info=inspect(args.container)
    key=os.environ.get('VLLM_API_KEY','')
    if not key:raise RuntimeError('VLLM_API_KEY is required')
    headers={'Content-Type':'application/json','Authorization':'Bearer '+key}
    audit=wait_idle(args.container,args.endpoint,headers,args.gpus)
    session=dict(name=args.name,container=args.container,gpus=args.gpus,model_alias=args.alias,alias=args.alias,
        port=int(args.endpoint.rsplit(':',1)[-1]),context_length=65536,concurrency=args.concurrency)
    if kind=='caption':
        items=json.loads((args.run_dir/'prepared/manifest.json').read_text(encoding='utf-8'))
        selected=args.sample_ids.split(',') if args.sample_ids else [v['sample_id'] for v in items]
        if not selected or not set(selected)<={v['sample_id'] for v in items}:raise ValueError('Unknown sample selection')
        counts={v:json.loads((args.run_dir/'capacity'/(v+'.json')).read_text(encoding='utf-8'))['input_tokens'] for v in selected if (args.run_dir/'capacity'/(v+'.json')).exists()}
        warm=args.warmup_sample or (max(counts,key=counts.get) if counts else selected[0])
        if warm not in selected:raise ValueError('Warmup sample is not selected')
        session.update(sample_ids=selected,warmup_sample_id=warm,measured_request_count=args.requests or len(selected))
        (args.run_dir/'capacity').mkdir(exist_ok=True)
    save(args.run_dir/'active-service.json',session)
    save(args.run_dir/('before-'+args.name+'-'+args.phase+'-isolation.json'),audit)
    return args,session,headers,audit

def finish_audit(args,session,headers,before,out,expected):
    after=wait_idle(session['container'],args.endpoint,headers,session['gpus'])
    actual=after['terminal']-before['terminal']
    valid=after['started_at']==before['started_at'] and actual==expected
    save(out/'api-counter-audit.json',dict(before=before['terminal'],after=after['terminal'],expected=expected,actual=actual,
        same_container_start=after['started_at']==before['started_at'],no_other_generation_requests=valid))
    save(out/'after-isolation.json',after)
    if not valid:raise RuntimeError('API request-count audit failed; preserve results but do not accept performance')
