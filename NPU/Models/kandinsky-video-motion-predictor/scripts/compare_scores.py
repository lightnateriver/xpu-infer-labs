"""Pair retained score files and compute aggregate numeric drift, without a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import correlation, mean

SCORES=('motion_camera','motion_object','motion_dynamics')

def read_scores(path):
    raw=Path(path).read_text(encoding='utf-8')
    rows=json.loads(raw) if raw.lstrip().startswith('[') else [json.loads(s) for s in raw.splitlines() if s.strip()]
    records={}
    for row in rows:
        clip=row['clip_id']
        if not isinstance(clip,str) or not clip or clip in records:raise ValueError('Invalid or duplicate clip_id')
        for key in SCORES:
            value=row['scores'][key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Scores must be finite numbers')
        records[clip]=row
    if not records:raise ValueError('No score records')
    return records

def statistics(reference,candidate):
    error=[b-a for a,b in zip(reference,candidate)]
    absolute=sorted(map(abs,error));pos=(len(error)-1)*.95;i=int(pos)
    mse=mean(e*e for e in error)
    refvar=sum((v-mean(reference))**2 for v in reference)
    testvar=sum((v-mean(candidate))**2 for v in candidate)
    norm=sum(v*v for v in reference)
    return dict(count=len(error),mse=mse,rmse=math.sqrt(mse),mae=mean(absolute),
                p95_absolute_error=absolute[i]+(absolute[min(i+1,len(error)-1)]-absolute[i])*(pos-i),
                max_absolute_error=max(absolute),bias_test_minus_reference=mean(error),
                pearson_r=correlation(reference,candidate) if refvar and testvar else None,
                r2=1-sum(e*e for e in error)/refvar if refvar else None,
                relative_l2=math.sqrt(sum(e*e for e in error)/norm) if norm else None,
                exact_values=sum(a==b for a,b in zip(reference,candidate)))

def decisions(score):
    return dict(static=score['motion_camera']<.5 and score['motion_object']<.5 and score['motion_dynamics']<1,
                violent=score['motion_camera']>2.5 and score['motion_dynamics']>2.5)

def compare(reference,candidate,require_hashes=False):
    if set(reference)!=set(candidate):raise ValueError('Reference/candidate clip_id coverage differs')
    ids=list(reference)
    hashes_present=all('input_sha256' in reference[i] and 'input_sha256' in candidate[i] for i in ids)
    matches=sum(reference[i]['input_sha256']==candidate[i]['input_sha256'] for i in ids) if hashes_present else None
    if require_hashes and (not hashes_present or matches!=len(ids)):raise ValueError('Prepared input hashes differ or are missing')
    result=dict(clips=len(ids),pixel_hash_matches=matches,
                per_score={k:statistics([reference[i]['scores'][k] for i in ids],[candidate[i]['scores'][k] for i in ids]) for k in SCORES},
                global_metrics=statistics([reference[i]['scores'][k] for i in ids for k in SCORES],
                                          [candidate[i]['scores'][k] for i in ids for k in SCORES]),
                clips_all_scores_exact=sum(reference[i]['scores']==candidate[i]['scores'] for i in ids))
    result['threshold_changes']={k:sum(decisions(reference[i]['scores'])[k]!=decisions(candidate[i]['scores'])[k] for i in ids) for k in ['static','violent']}
    result['threshold_positive_counts']={k:dict(reference=sum(decisions(reference[i]['scores'])[k] for i in ids),candidate=sum(decisions(candidate[i]['scores'])[k] for i in ids)) for k in ['static','violent']}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-input-hashes',action='store_true')
    p.add_argument('--require-exact-scores',action='store_true')
    a=p.parse_args()
    result=compare(read_scores(a.reference),read_scores(a.candidate),a.require_input_hashes)
    result['source_sha256']={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('reference',a.reference),('candidate',a.candidate)]}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,allow_nan=False))
    if a.require_exact_scores and result['global_metrics']['mse']!=0:raise SystemExit('Score drift detected; metrics retained')

if __name__=='__main__':main()
