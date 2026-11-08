"""Portable orchestration of the unchanged recorded Caption clients.

New publication wrapper: no claim of a new GPU benchmark of this wrapper.
"""
import argparse,datetime,fcntl,json,os,re,subprocess,sys,time
from pathlib import Path
from ops import ROOT,api,owned,save
from memory_capacity import memory_command,capacity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=os.environ.get('DATA_DIR'))
    parser.add_argument('--prepared-dir',type=Path,default=os.environ.get('PREPARED_DIR'))
    parser.add_argument('--receipt',type=Path,default=ROOT/'.local/service/receipt.json')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=3,help='Required fully valid natural rounds')
    parser.add_argument('--max-attempts',type=int,default=9)
    args=parser.parse_args()
    assert args.data_dir and args.data_dir.is_dir()
    assert args.prepared_dir and (args.prepared_dir/'manifest.json').is_file()
    assert 1<=args.rounds<=args.max_attempts
    assert os.environ.get('VLLM_API_KEY'),'Provide this service API key in the environment'
    receipt=json.loads(args.receipt.read_text());owned(receipt)
    name=receipt['variant'];variant=json.loads((ROOT/'configs/experiment-plan.json').read_text())['variants'][name]
    lock=ROOT/'.local/orchestration.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args,receipt,name,variant)


def execute(args,receipt,name,variant):
    run=args.run_dir.resolve();assert not run.exists(),'Choose a fresh run directory'
    run.mkdir(parents=True,mode=0o700)
    (run/'prepared').symlink_to(args.prepared_dir.resolve(),target_is_directory=True)
    assert len(json.loads((run/'prepared/manifest.json').read_text()))==15,'This suite expects 15 frozen inputs'
    (run/'capacity').mkdir()
    graphs=Path(receipt['deploy_dir'])/'graphs'
    sizes=re.findall(r'GPU KV cache size: ([\d,]+) tokens',Path(receipt['log']).read_text())
    assert sizes,'No actual KV capacity found in service log'
    actual_capacity=int(sizes[-1].replace(',',''))
    save(run/'benchmark-plan.json',dict(variant=name,**variant,actual_capacity=actual_capacity,
        required_complete_rounds=args.rounds,max_attempts=args.max_attempts,thinking=True,
        fixed_warmup=dict(samples=4,output_tokens=4096,excluded_from_ranking=True),diagnostic_only=name=='memory'))
    save(run/'service-configuration.json',json.loads((Path(receipt['deploy_dir'])/'deploy.json').read_text()))
    env=os.environ.copy();env.update(BENCH_NATIVE_VLLM='1',BENCH_NATIVE_RECEIPT=str(args.receipt.resolve()),
        BENCH_LOCK_PATH=str(ROOT/'.local/benchmark.lock'),BENCH_KV_CAPACITY=str(actual_capacity),
        BENCH_OUTPUT_RESERVE=str(variant['reserve']),BENCH_ADMISSION_RATIO='0')
    common=['--env-file',os.devnull,'--run-dir',str(run),'--data-dir',str(args.data_dir.resolve()),
        '--container',receipt['container'],'--endpoint',receipt['endpoint'],'--alias',receipt['alias'],
        '--gpus',str(receipt['gpu']),'--concurrency',str(variant['concurrency'])]

    def invoke(client,label,phase='measured',audit=False,ratio=0):
        owned(receipt);(graphs/'phase.txt').write_text(label)
        marker=graphs/'audit-enabled'
        if audit:marker.touch()
        else:marker.unlink(missing_ok=True)
        local_env=dict(env,BENCH_ADMISSION_RATIO=str(ratio))
        print(json.dumps(dict(variant=name,name=label,phase=phase)),flush=True)
        before=api(receipt['endpoint'],'/metrics');(run/(label+'-before.prom')).write_text(before)
        try:
            with (run/(label+'.log')).open('w') as log:
                result=subprocess.run([sys.executable,'-u',str(ROOT/'benchmark'/client),*common,'--name',label,'--phase',phase],
                    env=local_env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
            folder=run/'runs'/label/phase;a=json.loads((folder/'api-counter-audit.json').read_text())
            assert a['no_other_generation_requests'] and a['same_container_start'] and a['actual']==a['expected']
            s=json.loads((folder/'summary.json').read_text())
            if phase!='preflight':assert s['failed']==0 and s['all_gpu_isolation_valid'] and s['all_tokens_match']
            if phase=='measured':assert s['thinking_observed_samples']==s['samples']
            if audit:
                checks=json.loads((graphs/'live-forward-audit.json').read_text())
                assert checks and all(v['exact_all'] for v in checks)
                save(run/(label+'-graph-audit.json'),checks)
            return result.returncode,s
        finally:
            marker.unlink(missing_ok=True)
            (run/(label+'-after.prom')).write_text(api(receipt['endpoint'],'/metrics'))

    code,_=invoke('bench_caption.py','input-preflight','preflight');assert code==0
    code,warm=invoke('bench_fixed.py','fixed-warmup',audit=True);assert code==0 and warm['valid']==warm['samples']==4
    if name=='memory':
        memory_command(receipt,'diagnostic-start','trim_and_reset')
        invoke('bench_caption.py','memory-diagnostic',audit=True)
        memory_command(receipt,'diagnostic-end','snapshot')
        save(run/'capacity-decision.json',capacity(graphs))
        print('Memory diagnosis complete; this run is excluded from performance ranking.')
        return
    complete=0
    for attempt in range(1,args.max_attempts+1):
        code,s=invoke('bench_caption.py',f'{name}-r{attempt}',ratio=variant['ratio'])
        qualified=code==0 and s['valid']==s['samples'] and s['truncated']==0
        save(run/(f'{name}-r{attempt}-status.json'),dict(qualified=qualified,returncode=code))
        complete+=int(qualified)
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize.py'),'--run-dir',str(run)],check=True)
        if complete>=args.rounds:break
    assert complete>=args.rounds,'Not enough complete rounds; all attempts retained in productive throughput'


if __name__=='__main__':main()
