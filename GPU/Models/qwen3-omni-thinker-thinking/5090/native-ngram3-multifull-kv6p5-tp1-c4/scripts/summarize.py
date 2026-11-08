"""Keep every natural attempt in effective Caption throughput."""
import argparse,json,statistics
from pathlib import Path
from ops import save

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True,type=Path);args=parser.parse_args()
    records=[]
    for p in sorted(args.run_dir.glob('*-r*-status.json')):
        n=p.name.removesuffix('-status.json');s=json.loads((args.run_dir/'runs'/n/'measured/summary.json').read_text())
        a=json.loads((args.run_dir/'runs'/n/'measured/api-counter-audit.json').read_text())
        assert a['actual']==a['expected']==s['samples'] and a['no_other_generation_requests'] and a['same_container_start']
        records.append(dict(name=n,qualified=json.loads(p.read_text())['qualified'],summary=s))
    assert records
    complete=[r['summary'] for r in records if r['qualified']];wall=sum(r['summary']['wall_seconds'] for r in records)
    result=dict(attempts=len(records),qualified_rounds=len(complete),requests=sum(r['summary']['samples'] for r in records),
        valid_captions=sum(r['summary']['valid'] for r in records),total_wall_seconds=wall,
        output_tokens=sum(r['summary']['output_tokens'] for r in records),thinking_included=True,
        preemptions=sum(r['summary']['observed_preemptions_delta'] or 0 for r in records),rounds=records)
    result['effective_items_per_minute']=result['valid_captions']/wall*60
    result['all_attempt_output_tps_per_gpu']=result['output_tokens']/wall
    result['median_complete_round_tps']=statistics.median(s['output_tps_per_gpu'] for s in complete) if complete else None
    save(args.run_dir/'pooled-results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rounds'},indent=2))
