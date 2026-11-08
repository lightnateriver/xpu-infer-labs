"""Same heretic compatibility fixes; audit actual KV and freeze heldout-free scales."""
from caption_worker import CaptionBaselineWorker
import functools
import json
import os
from pathlib import Path
import torch

def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

class KVAuditWorker(CaptionBaselineWorker):
    def load_model(self,*args,**kwargs):
        expected_uuid=os.environ.get('KV_EXPECTED_GPU_UUID')
        if expected_uuid:
            assert torch.cuda.device_count()==1
            actual_uuid=str(torch.cuda.get_device_properties(0).uuid)
            assert actual_uuid==expected_uuid.removeprefix('GPU-'),(actual_uuid,expected_uuid)
            free,total=torch.cuda.mem_get_info(0)
            assert free>=28*1024**3,('Insufficient isolated GPU memory before model load',free,total)
            print('GPU_IDENTITY_PRELOAD_AUDIT',json.dumps({'uuid':actual_uuid,'free_bytes':free,'total_bytes':total}),flush=True)
        result=super().load_model(*args,**kwargs)
        self.audit_dir=Path(os.environ['KV_AUDIT_DIR']);self.audit_dir.mkdir(parents=True,exist_ok=True)
        self.calibrating=os.environ.get('KV_CALIBRATION')=='1'
        self.attention_layers={name:layer for name,layer in self.model_runner.model.named_modules()
            if 'language_model' in name and hasattr(layer,'kv_cache_dtype')}
        scales_path=os.environ.get('KV_SCALES_FILE')
        if scales_path:
            values=json.loads(Path(scales_path).read_text())
            assert values['heldout_used'] is False and set(values['layers'])==set(self.attention_layers)
            for name,layer in self.attention_layers.items():
                for kind in ['k','v']:
                    value=values['layers'][name][kind+'_scale']
                    assert 0<value<1000
                    with torch.no_grad():getattr(layer,'_'+kind+'_scale').fill_(value)
                    setattr(layer,'_'+kind+'_scale_float',value)
                    with torch.no_grad():getattr(layer,'_'+kind+'_scale_cpu').fill_(value)
        rows=[]
        for name,layer in self.attention_layers.items():
            rows.append({'name':name,'kv_dtype':layer.kv_cache_dtype,'impl':type(layer.impl).__name__,
                'k_scale':layer._k_scale.item(),'v_scale':layer._v_scale.item(),
                'q_scale':layer._q_scale.item(),'heads':layer.num_heads,'kv_heads':layer.num_kv_heads,
                'head_size':layer.head_size})
        expected=os.environ['KV_EXPECTED_DTYPE'];impl=os.environ['KV_EXPECTED_IMPL']
        assert len(rows)==48 and all(r['kv_dtype']==expected and r['impl']==impl for r in rows),rows
        experts=[{'name':n,'method':type(m.quant_method).__name__} for n,m in self.model_runner.model.named_modules()
            if 'experts' in n and hasattr(m,'quant_method')]
        assert len(experts)==48 and all('FP4' in r['method'].upper() for r in experts),experts
        save(self.audit_dir/'loaded-model-audit.json',{'attention':rows,'experts':experts,
            'calibration':self.calibrating,'scales_file':scales_path})
        print('FP8_KV_LOADED_MODEL_AUDIT',json.dumps({'attention_layers':len(rows),'expert_layers':len(experts),
            'kv_dtype':expected,'impl':impl,'calibration':self.calibrating}),flush=True)
        if self.calibrating:
            for name,layer in self.attention_layers.items():
                layer._kv_observed_amax=torch.zeros(2,device=layer._k_scale.device,dtype=torch.float32)
                original=layer.forward
                def wrap(fn,module):
                    @functools.wraps(fn)
                    def observed(query,key,value,*a,**kw):
                        if (self.audit_dir/'calibration-active').exists() and key is not None and value is not None:
                            maxima=torch.stack((key.detach().abs().amax().float(),value.detach().abs().amax().float()))
                            module._kv_observed_amax.copy_(torch.maximum(module._kv_observed_amax,maxima))
                        return fn(query,key,value,*a,**kw)
                    return observed
                layer.forward=wrap(original,layer)
            original_execute=self.model_runner.execute_model
            @functools.wraps(original_execute)
            def execute(*a,**kw):
                value=original_execute(*a,**kw)
                if (self.audit_dir/'calibration-active').exists():
                    maxima={n:m._kv_observed_amax.tolist() for n,m in self.attention_layers.items()}
                    save(self.audit_dir/'observed-kv-amax.json',maxima)
                return value
            self.model_runner.execute_model=execute
        return result

    def initialize_from_config(self,kv_cache_config):
        result=super().initialize_from_config(kv_cache_config);views=[]
        def collect(value,label):
            if isinstance(value,torch.Tensor):views.append({'name':label,'dtype':str(value.dtype),'shape':list(value.shape)})
            elif isinstance(value,dict):
                for k,v in value.items():collect(v,label+'.'+str(k))
            elif isinstance(value,(list,tuple)):
                for i,v in enumerate(value):collect(v,label+'.'+str(i))
        collect(getattr(self.model_runner,'kv_caches',[]),'kv_caches')
        expected='torch.bfloat16' if os.environ['KV_EXPECTED_DTYPE']=='bfloat16' else 'torch.float8_e4m3fn'
        # Some backends retain uint8 storage and expose typed views when consumed.
        allowed=[expected] if expected=='torch.bfloat16' else [expected,'torch.uint8']
        assert len(views)==48 and all(v['dtype'] in allowed for v in views),views[:5]
        save(self.audit_dir/'allocated-kv-cache-audit.json',{'expected_logical_dtype':expected,'views':views})
        routes=[]
        for group in self.model_runner.attn_groups:
            for attn_group in group:
                builder=attn_group.get_metadata_builder()
                routes.append({'builder':type(builder).__name__,'layers':attn_group.layer_names,
                    **{key:str(getattr(builder,key,None)) for key in ['q_data_type_prefill','q_data_type_decode',
                    'kv_cache_dtype','use_xqa','use_trtllm_decode_attention','flashinfer_trtllm_api_decode_kernel']}})
        save(self.audit_dir/'attention-routing-audit.json',routes)
        if os.environ['KV_EXPECTED_DTYPE']=='fp8_e4m3':
            fi=[r for r in routes if r['builder']=='FlashInferMetadataBuilder']
            assert fi and all(r['use_xqa']=='True' and r['q_data_type_prefill']==r['q_data_type_decode']=='torch.bfloat16' for r in fi),routes
        return result
