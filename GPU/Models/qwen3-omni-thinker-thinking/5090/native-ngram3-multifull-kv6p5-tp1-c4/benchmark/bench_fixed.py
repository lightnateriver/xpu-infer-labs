"""Sequential streaming caption benchmark. Preserve every response and timestamp."""
import base64
import csv
import copy
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

from _common import benchmark_setup, finish_audit, command
args,SESSION,headers,AUDIT=benchmark_setup('caption')
ROOT=args.run_dir.resolve()
DATA=args.data_dir.resolve()
RUN=ROOT/'runs'/SESSION['name']
PHASE=args.phase
OUT=RUN/PHASE
OUT.mkdir(parents=True,exist_ok=True)
CONTAINER=SESSION['container']
ENDPOINT=args.endpoint.rstrip('/')
manifest=json.loads((ROOT/'prepared/manifest.json').read_text(encoding='utf-8'))
assert manifest

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def data_url(path):
    mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()
telemetry=[]
stop=threading.Event()
def collect():
    while not stop.is_set():
        stamp=time.time()
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            cards=[[x.strip() for x in row] for row in csv.reader(raw.splitlines())]
            uuid_to_gpu={r[1]:int(r[0]) for r in cards}
            own={int(x.strip()) for x in command(['docker','top',CONTAINER,'-eo','pid']).splitlines()[1:] if x.strip().isdigit()}
            raw_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
            foreign=[{'gpu':uuid_to_gpu[r[0].strip()],'pid':int(r[1]),'memory_mib':r[2].strip()} for r in csv.reader(raw_processes.splitlines()) if len(r)==3 and uuid_to_gpu.get(r[0].strip()) in SESSION['gpus'] and int(r[1]) not in own]
            entry={'time':stamp,'gpus':[{'gpu':int(r[0]),'memory_mib':int(r[2]),'utilization_pct':int(r[3])} for r in cards],'foreign_gpu_processes':foreign}
            try:
                req=urllib.request.Request(ENDPOINT+'/metrics',headers=headers)
                with urllib.request.urlopen(req,timeout=2) as response:metrics=response.read().decode()
                selected_metrics={}
                for line in metrics.splitlines():
                    if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{','vllm:num_preemptions_total{')):
                        selected_metrics[line.split('{',1)[0]]=float(line.rsplit(' ',1)[-1])
                entry['engine_metrics']=selected_metrics
            except Exception as caught:entry['engine_metrics_error']=type(caught).__name__
            telemetry.append(entry)
        except Exception as error:telemetry.append({'time':stamp,'error':repr(error)})
        stop.wait(1)
thread=threading.Thread(target=collect,daemon=True)
def run(item,warmup=False,repeat_index=None,submitted_ns=None):
    artifact_id=item['sample_id'] if repeat_index is None else item['sample_id']+f'__repeat_{repeat_index:02d}'
    folder=OUT/('warmup' if warmup else artifact_id);folder.mkdir(exist_ok=True)
    started=time.perf_counter_ns();started_epoch=time.time();initial_started=started
    sample=DATA/item['sample_id']
    system=(ROOT/'prepared'/item['prepared_subdir']/'system-prompt.txt').read_text(encoding='utf-8')
    assert hashlib.sha256(system.encode()).hexdigest()==item['system_prompt_sha256']
    content=[]
    for reference in item['reference_images']:
        path=sample/reference['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['sha256']
        content.extend([{'type':'text','text':reference['picture_id']+' reference image:'},{'type':'image_url','image_url':{'url':data_url(path)}}])
    video=sample/'video.mp4'
    assert hashlib.sha256(video.read_bytes()).hexdigest()==item['video_sha256']
    content.extend([{'type':'text','text':'Target video with its original synchronized audio:'},{'type':'video_url','video_url':{'url':data_url(video)}},{'type':'text','text':'Generate the required output JSON for sample_id '+item['sample_id']+'.'}])
    media_namespace=SESSION['name']+':'+PHASE+':'+artifact_id
    for position,part in enumerate(content):
        if part['type'] in ['image_url','video_url']:
            part['uuid']=media_namespace+':'+str(position)
    body={'model':SESSION['model_alias'],'messages':[{'role':'system','content':system},{'role':'user','content':content}],
          'modalities':['text'],'temperature':0,'seed':0,'max_tokens':4096,'ignore_eos':True,'stream':True,
          'chat_template_kwargs':{'enable_thinking':True},
          'stream_options':{'include_usage':True},'return_token_ids':True,
          'mm_processor_kwargs':{'use_audio_in_video':True},'media_io_kwargs':{'video':{'fps':2,'num_frames':-1}}}
    messages_hash=hashlib.sha256(json.dumps(body['messages'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    semantic_messages=copy.deepcopy(body['messages'])
    for message in semantic_messages:
        if isinstance(message['content'],list):
            for part in message['content']:part.pop('uuid',None)
    semantic_hash=hashlib.sha256(json.dumps(semantic_messages,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    preflight_started=time.perf_counter_ns()
    token_path=ROOT/'capacity'/(item['sample_id']+'.json')
    if token_path.exists():
        counted=json.loads(token_path.read_text())
        assert counted.get('semantic_messages_sha256',counted['messages_sha256'])==semantic_hash
    else:
        token_body={k:body[k] for k in ['model','messages','chat_template_kwargs','mm_processor_kwargs','media_io_kwargs']}
        token_body['add_generation_prompt']=True
        token_req=urllib.request.Request(ENDPOINT+'/tokenize',data=json.dumps(token_body,ensure_ascii=False).encode(),headers=headers)
        with urllib.request.urlopen(token_req,timeout=1800) as response:token_result=json.load(response)
        counted={'sample_id':item['sample_id'],'input_tokens':token_result['count'],'messages_sha256':messages_hash,'semantic_messages_sha256':semantic_hash}
        save(token_path,counted)
    assert counted['input_tokens']+body['max_tokens']<=SESSION['context_length'],'Full input and output budget must fit'
    if PHASE=='preflight':
        row=dict(sample_id=item['sample_id'],input_tokens=counted['input_tokens'],max_output_tokens=body['max_tokens'],context_tokens=SESSION['context_length'],semantic_messages_sha256=semantic_hash)
        save(folder/'input-capacity.json',row);return row
    # Token-count preflight is excluded from sample preparation and HTTP timing.
    preflight_ns=time.perf_counter_ns()-preflight_started
    started+=preflight_ns;started_epoch+=preflight_ns/1e9
    payload=json.dumps(body,ensure_ascii=False).encode('utf-8')
    request_started=time.perf_counter_ns();request_epoch=time.time()
    first=last=first_content=done=None
    raw_thought_buffer='';raw_thought_ended=False
    events=[];raw=[];text=[];reasoning=[];token_ids=[];usage=None;finish=None;error=None;http_status=None
    try:
        req=urllib.request.Request(ENDPOINT+'/v1/chat/completions',data=payload,headers={**headers,'X-Probe-Request-Id':hashlib.sha256(artifact_id.encode()).hexdigest()[:16]})
        with urllib.request.urlopen(req,timeout=1800) as response:
            http_status=response.status
            for line in response:
                stamp=time.perf_counter_ns();raw.append(line)
                if not line.startswith(b'data:'):continue
                data=line[5:].strip()
                if data==b'[DONE]':done=stamp;break
                value=json.loads(data)
                events.append({'offset_seconds':(stamp-request_started)/1e9,'event':value})
                if value.get('error'):raise RuntimeError(json.dumps(value['error']))
                if value.get('usage'):usage=value['usage']
                for choice in value.get('choices',[]):
                    delta=choice.get('delta') or {}
                    chunk=delta.get('content') or ''
                    thought=delta.get('reasoning_content') or delta.get('reasoning') or ''
                    token_ids.extend(choice.get('token_ids') or delta.get('token_ids') or [])
                    if chunk or thought:
                        if first is None:first=stamp
                        last=stamp
                    if chunk:
                        if thought or reasoning:
                            if first_content is None:first_content=stamp
                        else:
                            raw_thought_buffer+=chunk
                            if '</think>' in raw_thought_buffer:
                                raw_thought_ended=True
                                if raw_thought_buffer.rsplit('</think>',1)[1].strip() and first_content is None:first_content=stamp
                        text.append(chunk)
                    if thought:reasoning.append(thought)
                    if choice.get('finish_reason'):finish=choice['finish_reason']
            assert done is not None,'SSE ended without [DONE]'
    except Exception as caught:error=repr(caught)
    ended=time.perf_counter_ns();ended_epoch=time.time()
    output=''.join(text)
    raw_output=output
    if '</think>' in output:
        thought,output=output.rsplit('</think>',1)
        reasoning.append(thought.replace('<think>','').strip())
        output=output.strip()
    (folder/'raw-content.txt').write_text(raw_output,encoding='utf-8')
    (folder/'reasoning.txt').write_text(''.join(reasoning),encoding='utf-8')
    save(folder/'output-token-ids.json',token_ids)
    (folder/'output.txt').write_text(output,encoding='utf-8')
    (folder/'stream.sse').write_bytes(b''.join(raw))
    save(folder/'events.json',events)
    parsed=None;schema_error=None
    try:
        parsed=json.loads(output)
        assert set(parsed)=={'sample_id','chain_mode','speaker_source_decisions','six_section'}
        assert parsed['sample_id']==item['sample_id']
        assert parsed['chain_mode']=='full_reference_six_section_en_v2'
        assert isinstance(parsed['speaker_source_decisions'],list)
        keys={'subject_definitions','summary','retention_analysis','detailed_description','overall_soundscape','non_diegetic_music'}
        assert set(parsed['six_section'])==keys and all(isinstance(v,str) for v in parsed['six_section'].values())
    except Exception as caught:schema_error=repr(caught)
    if parsed is not None:save(folder/'output.json',parsed)
    n=(usage or {}).get('completion_tokens')
    request_end=done or ended
    window=[t for t in telemetry if request_epoch<=t['time']<=ended_epoch]
    foreign=[t for t in window if t.get('foreign_gpu_processes')]
    ttft=(first-request_started)/1e9 if first else None
    request_e2e=(request_end-request_started)/1e9
    row={'sample_id':item['sample_id'],'request_id':artifact_id,'artifact_subdir':folder.name,'repeat_index':repeat_index,'warmup':warmup,'reference_count':item['reference_count'],'video_seconds':item['duration_seconds'],'shot_count':item['shot_count'],
         'started_epoch':started_epoch,'request_started_epoch':request_epoch,'ended_epoch':ended_epoch,
         'client_queue_seconds':(initial_started-submitted_ns)/1e9 if submitted_ns else 0,
         'dataset_e2e_seconds':(request_end-submitted_ns)/1e9 if submitted_ns else (request_end-started)/1e9,
         'prepare_seconds':(request_started-started)/1e9,'ttft_seconds':ttft,'first_content_seconds':(first_content-request_started)/1e9 if first_content else None,
         'request_e2e_seconds':request_e2e,'sample_e2e_seconds':(request_end-started)/1e9,
         'tpot_seconds':(request_e2e-ttft)/(n-1) if n and n>1 and ttft is not None else None,
         'tpot_first_to_last_text_seconds':(last-first)/1e9/(n-1) if n and n>1 and first else None,
         'stream_tail_seconds':(request_end-last)/1e9 if last else None,'text_chunk_count':sum(any((x.get('delta') or {}).get('content') for x in e['event'].get('choices',[])) for e in events),
         'usage':usage,'finish_reason':finish,'truncated':finish=='length','http_status':http_status,'error':error,'schema_valid':schema_error is None,'schema_error':schema_error,
         'foreign_gpu_samples':foreign,'gpu_isolation_valid':not foreign and bool(window) and not any('error' in t for t in window),
         'streaming_timing_valid':first is not None and n is not None and sum(any(any((x.get('delta') or {}).get(k) for k in ('content','reasoning_content','reasoning')) for x in e['event'].get('choices',[])) for e in events)>1,
         'input_payload_bytes':len(payload),'input_payload_sha256':hashlib.sha256(payload).hexdigest(),'system_prompt_sha256':item['system_prompt_sha256'],'reasoning_characters':len(''.join(reasoning)),
         'thinking_observed':bool(''.join(reasoning).strip()),
         'think_delimiter_token_ids_observed':any(t in [151667,151668] for t in token_ids),
         'returned_token_ids':len(token_ids),'messages_sha256':messages_hash,
         'semantic_messages_sha256':semantic_hash,'media_cache_namespace':media_namespace}
    row['preflight_input_tokens']=counted['input_tokens']
    row['prompt_tokens_match_preflight']=(usage or {}).get('prompt_tokens')==counted['input_tokens']
    save(folder/'metrics.json',row)
    save(folder/'input-description.json',{'sample':item,'request_parameters':{k:v for k,v in body.items() if k!='messages'},'runtime_context_file':str(ROOT/'prepared'/item['prepared_subdir']/'runtime-context.json'),'system_prompt_file':str(ROOT/'prepared'/item['prepared_subdir']/'system-prompt.txt'),'payload_sha256':row['input_payload_sha256']})
    return row
if __name__=='__main__':
    from concurrent.futures import ThreadPoolExecutor,as_completed
    import re
    def parsed_audit(row):
        p=OUT/row.get('artifact_subdir',row['sample_id'])/'output.txt'
        original=p.read_text(encoding='utf-8')
        stripped=re.sub(r'^```(?:json)?\s*|\s*```$','',original.strip(),flags=re.I).strip()
        result={'markdown_fence':original.strip().startswith('```')}
        try:
            v=json.loads(stripped)
            result['json_parsed']=True
            result['sample_id_matches']=v.get('sample_id')==row['sample_id']
            result['structure_valid']=set(v)=={'sample_id','chain_mode','speaker_source_decisions','six_section'} and v.get('chain_mode')=='full_reference_six_section_en_v2' and isinstance(v.get('speaker_source_decisions'),list) and isinstance(v.get('six_section'),dict) and set(v['six_section'])=={'subject_definitions','summary','retention_analysis','detailed_description','overall_soundscape','non_diegetic_music'} and all(isinstance(x,str) for x in v['six_section'].values())
            save(p.parent/'output-parsed.json',v)
        except Exception as e:result.update(json_parsed=False,sample_id_matches=False,structure_valid=False,error=repr(e))
        row['format_audit']=result
        save(p.parent/'metrics.json',row)
    thread.start()
    try:
        chosen=[x for x in manifest if x['sample_id'] in SESSION['sample_ids']]
        assert len(chosen)==len(SESSION['sample_ids'])
        largest=next(x for x in chosen if x['sample_id']==SESSION['warmup_sample_id'])
        if PHASE=='preflight':
            rows=[run(item) for item in chosen]
            save(OUT/'metrics.json',rows)
            save(OUT/'summary.json',dict(samples=len(rows),max_input_tokens=max(v['input_tokens'] for v in rows)))
        elif PHASE=='warmup':
            row=run(largest,True)
            parsed_audit(row)
            save(OUT/'metrics.json',[row])
            print('WARMUP',json.dumps(row),flush=True)
            assert row['error'] is None and not row['truncated'] and row['gpu_isolation_valid'] and row['streaming_timing_valid'] and row['prompt_tokens_match_preflight'] and row['thinking_observed'],'Warmup validation failed'
            assert row['format_audit']['structure_valid'] and row['format_audit']['sample_id_matches'],'Warmup output schema failed'
        else:
            concurrency=args.concurrency
            ids=SESSION['sample_ids']
            selected=[chosen[i] for i in [0,4,9,14]]
            request_count=len(selected)
            assert request_count>=len(selected) and request_count%len(selected)==0
            jobs=[(selected[i%len(selected)],None if request_count==len(selected) else i//len(selected)) for i in range(request_count)]
            start=time.perf_counter();start_epoch=time.time();rows=[]
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures=[pool.submit(run,item,False,repeat_index,time.perf_counter_ns()) for item,repeat_index in jobs]
                for future in as_completed(futures):
                    row=future.result();parsed_audit(row);rows.append(row)
                    save(OUT/'metrics.json',rows)
                    print('SAMPLE',row['sample_id'],json.dumps({k:row[k] for k in ['ttft_seconds','tpot_seconds','request_e2e_seconds','usage','error','truncated','gpu_isolation_valid','format_audit']}),flush=True)
            wall=time.perf_counter()-start
            summary={'samples':len(rows),'concurrency':concurrency,'gpus':SESSION['gpus'],'tensor_parallel_size':len(SESSION['gpus']),'wall_seconds':wall,'started_epoch':start_epoch,'ended_epoch':time.time(),'requests_per_second':len(rows)/wall,'output_tokens_per_second':sum((r['usage'] or {}).get('completion_tokens',0) for r in rows)/wall,'input_tokens_per_second':sum((r['usage'] or {}).get('prompt_tokens',0) for r in rows)/wall,'failed':sum(r['error'] is not None for r in rows),'truncated':sum(r['truncated'] for r in rows),'all_gpu_isolation_valid':all(r['gpu_isolation_valid'] for r in rows),'all_tokens_match':all(r['prompt_tokens_match_preflight'] for r in rows),'json_parsed':sum(r['format_audit']['json_parsed'] for r in rows),'structure_valid':sum(r['format_audit']['structure_valid'] for r in rows),'sample_id_matches':sum(r['format_audit']['sample_id_matches'] for r in rows),'output_tokens':sum((r['usage'] or {}).get('completion_tokens',0) for r in rows),'medians':{k:statistics.median([r[k] for r in rows if r[k] is not None]) if any(r[k] is not None for r in rows) else None for k in ['ttft_seconds','tpot_seconds','request_e2e_seconds','sample_e2e_seconds']}}
            summary['thinking_observed_samples']=sum(r['thinking_observed'] for r in rows)
            summary['unique_samples']=len({r['sample_id'] for r in rows})
            summary['measured_request_count']=len(rows)
            summary['requests_per_unique_sample']=request_count//len(selected)
            summary['reasoning_characters']=sum(r['reasoning_characters'] for r in rows)
            summary['requested_client_concurrency']=args.concurrency
            observed=[v['engine_metrics'] for v in telemetry if v.get('engine_metrics')]
            for field,metric in [('observed_server_running_peak','vllm:num_requests_running'),('observed_server_waiting_peak','vllm:num_requests_waiting'),('observed_kv_usage_peak','vllm:kv_cache_usage_perc')]:
                values=[v[metric] for v in observed if metric in v]
                summary[field]=max(values) if values else None
            preemptions=[v['vllm:num_preemptions_total'] for v in observed if 'vllm:num_preemptions_total' in v]
            summary['observed_preemptions_delta']=preemptions[-1]-preemptions[0] if preemptions else None
            summary['valid']=sum(not r['error'] and r['gpu_isolation_valid'] and r['streaming_timing_valid'] and r['prompt_tokens_match_preflight'] and r['thinking_observed'] and r['usage']['completion_tokens']==4096 and r['finish_reason']=='length' for r in rows)
            summary['valid_samples_per_minute']=summary['valid']/wall*60
            summary['output_tps_per_gpu']=summary['output_tokens_per_second']/len(SESSION['gpus'])
            summary['protocol']='fixed-length-efficiency-only-not-caption-quality';summary['fixed_output_tokens_per_request']=4096;save(OUT/'summary.json',summary);print('SUMMARY',json.dumps(summary),flush=True)
            assert summary['valid']==summary['samples'],'Some outputs are incomplete or invalid; keep artifacts but reject the run'
            assert summary['failed']==0 and summary['all_gpu_isolation_valid'] and summary['all_tokens_match'],'Measurements failed validation'
    finally:
        stop.set();thread.join(timeout=10);save(OUT/'gpu-telemetry.json',telemetry)

    finish_audit(args,SESSION,headers,AUDIT,OUT,0 if PHASE=='preflight' else 1 if PHASE=='warmup' else request_count)
