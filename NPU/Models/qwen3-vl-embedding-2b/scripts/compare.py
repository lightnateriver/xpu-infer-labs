"""Compare complete benchmark outputs; publish aggregates, keep vectors private."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def unit(x):
    x=x.astype(np.float64)
    return x/np.linalg.norm(x,axis=1,keepdims=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,help='label=/path/to/output')
    p.add_argument('--reference-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    refs=json.loads(a.reference_map.read_text())
    report={'date':'2026-11-08','state':'complete','analysis_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runs':{},'precision_scope':'combined cohorts; all directed nonself pairs; frozen GPU queries and gallery; threshold >=0.99',
        'dataset':'private frozen 2x64 cohorts, 16 separate warmup; both cohorts previously used for evaluation, not blind',
        'performance_scope':'sum of cohort medians, not a new 128-concurrent batch; resident warmed pipeline, no profiler'}
    boundaries={}
    for item in a.runs:
        label,value=item.split('=',1);root=Path(value)
        s=json.loads((root/'status.json').read_text());assert s['state']=='complete'
        names=[c['name'] for c in s['cohorts']]
        q=unit(np.concatenate([np.load(refs[n]['query']) for n in names]))
        g=unit(np.concatenate([np.load(refs[n]['gallery']) for n in names]))
        stable=np.concatenate([np.load(refs[n]['stable']) for n in names])
        baseline=q@g.T;previous=unit(stable)@g.T
        off=~np.eye(len(q),dtype=bool)
        quality=[];first=None
        for i in range(s['repeats']):
            x=np.concatenate([np.load(root/f'{n}-{i}.npy') for n in names]);u=unit(x);scores=u@g.T;cos=np.sum(u*q,axis=1)
            quality.append({'repeat':i,'count':len(x),'same_input_cosine_min':float(cos.min()),'vectors_below099':int((cos<.99).sum()),
                'fixed_old_gpu_gallery_crossings':int((off&((scores>=.99)!=(baseline>=.99))).sum()),
                'decision_changes_vs_stable_npu':int((off&((scores>=.99)!=(previous>=.99))).sum()),
                'max_pair_score_delta_vs_frozen_gpu':float(np.abs(scores-baseline)[off].max()),
                'max_abs_vs_stable_npu':float(np.abs(x-stable).max()),'exact_vs_stable_npu':bool(np.array_equal(x,stable)),
                'byte_equal_vs_first':first is None or bool(np.array_equal(x,first))})
            if first is None:first=x
        total=sum(c['median_s'] for c in s['cohorts']);count=sum(c['count'] for c in s['cohorts'])
        r={k:s[k] for k in ['parameters','versions','model_sha256','code_sha256','startup_s','pool_warmup','cohorts','repeats','timing']}
        r.update(combined_median_sum_s=total,combined_clips_s=count/total,quality=quality,
            same_input_cosine_gate_pass=all(v['vectors_below099']==0 for v in quality),
            frozen_gallery_zero_crossing_gate_pass=all(v['fixed_old_gpu_gallery_crossings']==0 for v in quality))
        if 'graph_final' in s:r['graph_final']=s['graph_final']
        report['runs'][label]=r
        boundaries[label]={}
        for n in names:
            b=json.loads((root/f'{n}-boundary.json').read_text())
            c=json.loads((root/f'{n}-contract.json').read_text())
            boundaries[label][n]=(b,c)
    reference=next(iter(boundaries.values()))
    report['all_runs_pixels_metadata_contract_equal']=all(v==reference for v in boundaries.values())
    report['boundary_comparisons']=[]
    for label,values in boundaries.items():
        for name,(b,c) in values.items():
            rb,rc=reference[name]
            report['boundary_comparisons'].append({'run':label,'cohort':name,'count':len(b),
                'pixels_metadata_hash_differences_vs_first_run':sum(b.get(k)!=v for k,v in rb.items()),
                'frame_shape_prompt_contract_differences_vs_first_run':sum(x!=y for x,y in zip(c,rc,strict=True))})
    labels=list(boundaries)
    report['boundary_pairwise']=[]
    for i,left in enumerate(labels):
        for right in labels[i+1:]:
            pixels=contracts=0
            for name,(b,c) in boundaries[left].items():
                rb,rc=boundaries[right][name]
                pixels+=sum(b.get(k)!=v for k,v in rb.items())
                contracts+=sum(x!=y for x,y in zip(c,rc,strict=True))
            report['boundary_pairwise'].append({'left':left,'right':right,'pixels_metadata_hash_differences':pixels,'frame_shape_prompt_contract_differences':contracts})
    # No sample-level data or local paths are included in the exported report.
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'state':report['state'],'all_runs_pixels_metadata_contract_equal':report['all_runs_pixels_metadata_contract_equal'],
        'throughput':{k:v['combined_clips_s'] for k,v in report['runs'].items()}}))


if __name__=='__main__':main()
