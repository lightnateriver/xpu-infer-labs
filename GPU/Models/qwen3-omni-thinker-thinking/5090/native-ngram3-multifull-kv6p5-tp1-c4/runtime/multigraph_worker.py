"""Exact-shape FULL graphs for XQA decode; process-local, no package edits."""
from collections import Counter,defaultdict
from dataclasses import dataclass
import functools,inspect,itertools,json,os,textwrap,time,types
from pathlib import Path
import torch
from native_worker import NativeAuditWorker
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor,get_forward_context,is_forward_context_available
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder,FlashInferTrtllmAPIDecode,FlashInferDecodeKernel


@dataclass(frozen=True)
class QueryShapeDescriptor(BatchDescriptor):
    query_shape:tuple[int,...]=()


class MultiShapeAuditWorker(NativeAuditWorker):
    def load_model(self,*args,**kwargs):
        result=super().load_model(*args,**kwargs)
        runner=self.model_runner;root=Path(os.environ['MULTIGRAPH_DIR'])
        assert self.vllm_config.parallel_config.tensor_parallel_size==1
        assert runner.uniform_decode_query_len==4 and not runner.lora_config
        state=dict(capture_shape=None,in_execute=False,captured=False,step=0,last_flush=0.,phase='startup')
        phases=defaultdict(Counter);shape_hits=defaultdict(Counter);audits=[];audit_count=Counter()
        def record(kind,shape=()):
            now=time.monotonic()
            if now-state['last_flush']>1:
                phase=(root/'phase.txt').read_text().strip() if (root/'phase.txt').exists() else 'startup'
                state['phase']=phase;state['last_flush']=now
                temp=root/'graph-runtime.json.tmp'
                temp.write_text(json.dumps(dict(phase_counters={k:dict(v) for k,v in phases.items()},
                    shape_replays={k:dict(v) for k,v in shape_hits.items()},audits=audits),indent=2))
                temp.replace(root/'graph-runtime.json')
            phases[state['phase']][kind]+=1
            if shape:shape_hits[state['phase']]['query='+','.join(map(str,shape))]+=1

        # The ragged causal mask is constant for an exact query tuple. Retain its
        # allocation across metadata builds and graph replays.
        mask_original=FlashInferMetadataBuilder._get_decode_mask
        @functools.wraps(mask_original)
        def mask(builder,q_len_per_req,ragged_q_lens,num_decodes,causal):
            if ragged_q_lens is None:return mask_original(builder,q_len_per_req,ragged_q_lens,num_decodes,causal)
            cache=getattr(builder,'_shape_graph_mask_cache',None)
            if cache is None:cache={};builder._shape_graph_mask_cache=cache
            key=(tuple(ragged_q_lens),q_len_per_req,bool(causal))
            if key not in cache:cache[key]=mask_original(builder,q_len_per_req,ragged_q_lens,num_decodes,causal)
            return cache[key]
        FlashInferMetadataBuilder._get_decode_mask=mask

        # Explicit dummy shape injection occurs only during startup capture.
        original_dummy=runner._dummy_run
        source=textwrap.dedent(inspect.getsource(original_dummy))
        needle='    assert sum(num_scheduled_tokens_list) == num_tokens'
        assert source.count(needle)==1
        source=source.replace(needle,"    if self._xqa_capture_shape is not None:\n        num_scheduled_tokens_list = list(self._xqa_capture_shape)\n        num_reqs = len(num_scheduled_tokens_list)\n        max_query_len = max(num_scheduled_tokens_list)\n"+needle)
        ns=dict(inspect.unwrap(original_dummy).__globals__);exec(compile(source,'<exact-shape-dummy-run>','exec'),ns)
        runner._dummy_run=types.MethodType(ns['_dummy_run'],runner);runner._xqa_capture_shape=None

        original_execute=runner.execute_model
        @functools.wraps(original_execute)
        def execute(*a,**kw):
            state['in_execute']=True;state['step']+=1
            try:return original_execute(*a,**kw)
            finally:state['in_execute']=False
        runner.execute_model=execute

        original_dispatch=runner._determine_batch_execution_and_padding
        signature=inspect.signature(original_dispatch)
        @functools.wraps(original_dispatch)
        def dispatch(*a,**kw):
            bound=signature.bind(*a,**kw);bound.apply_defaults();b=bound.arguments
            value=original_dispatch(*a,**kw);shape=tuple(int(v) for v in b['num_scheduled_tokens_np'])
            capturing=runner._xqa_capture_shape is not None
            eligible=capturing
            if state['in_execute'] and value[0]==CUDAGraphMode.PIECEWISE and not b['force_eager'] and not b['use_cascade_attn'] and b['num_encoder_reqs']==0:
                eligible=1<=len(shape)<=4 and set(shape)<={1,4}
                # A short final prefill chunk is not a decode merely because its
                # query length happens to equal one of the decode shapes.
                eligible=eligible and all(runner.input_batch.num_computed_tokens_cpu[i]>=runner.input_batch.num_prompt_tokens[i] for i in range(len(shape)))
            if eligible:
                assert sum(shape)==b['num_tokens'] and len(shape)==b['num_reqs']
                desc=QueryShapeDescriptor(num_tokens=sum(shape),num_reqs=len(shape),uniform=len(set(shape))==1,query_shape=shape)
                mode=CUDAGraphMode.NONE if b['force_eager'] else CUDAGraphMode.FULL
                value=(mode,desc,False,None,None)
            if state['in_execute']:
                record('dispatch_'+value[0].name)
                if isinstance(value[1],QueryShapeDescriptor):record('dispatch_shape_FULL')
            return value
        runner._determine_batch_execution_and_padding=dispatch

        original_capture=runner._capture_cudagraphs
        @functools.wraps(original_capture)
        def capture(*a,**kw):
            output=original_capture(*a,**kw)
            mode=kw.get('cudagraph_runtime_mode',a[1] if len(a)>1 else None)
            if mode==CUDAGraphMode.FULL and not state['captured']:
                state['captured']=True
                shapes=[s for n in range(1,5) for s in itertools.product([1,4],repeat=n) if not all(q==4 for q in s)]
                shapes.sort(key=lambda s:(-sum(s),-len(s),s))
                for shape in shapes:
                    runner._xqa_capture_shape=shape
                    desc=QueryShapeDescriptor(sum(shape),len(shape),len(set(shape))==1,query_shape=shape)
                    runner._warmup_and_capture(desc,CUDAGraphMode.FULL,num_warmups=1,allow_microbatching=False)
                    torch.cuda.synchronize()
                    print('XQA_SHAPE_CAPTURED',shape,flush=True)
                runner._xqa_capture_shape=None
                (root/'captured-shapes.json').write_text(json.dumps(dict(shapes=shapes,count=len(shapes)),indent=2))
            return output
        runner._capture_cudagraphs=capture

        # Count actual outer FULL replay calls. Optional correctness audit runs
        # the same live batch without graphs before any sampler/state advance.
        original_call=CUDAGraphWrapper.__call__
        def leaves(value):
            if isinstance(value,torch.Tensor):return [value]
            if isinstance(value,(tuple,list)):return [t for v in value for t in leaves(v)]
            if isinstance(value,dict):return [t for v in value.values() for t in leaves(v)]
            return []
        @functools.wraps(original_call)
        def call(wrapper,*a,**kw):
            if not is_forward_context_available():return original_call(wrapper,*a,**kw)
            ctx=get_forward_context();desc=ctx.batch_descriptor
            custom=isinstance(desc,QueryShapeDescriptor) and ctx.cudagraph_runtime_mode==CUDAGraphMode.FULL and wrapper.runtime_mode==CUDAGraphMode.FULL
            cached=custom and desc in wrapper.concrete_cudagraph_entries and wrapper.concrete_cudagraph_entries[desc].cudagraph is not None
            value=original_call(wrapper,*a,**kw)
            if cached and state['in_execute']:
                record('actual_shape_full_replay',desc.query_shape)
                if (root/'audit-enabled').exists() and audit_count[desc.query_shape]<2:
                    saved=[t.clone() for t in leaves(value)];old_mode=ctx.cudagraph_runtime_mode
                    ctx.cudagraph_runtime_mode=CUDAGraphMode.NONE
                    try:reference=original_call(wrapper,*a,**kw)
                    finally:ctx.cudagraph_runtime_mode=old_mode
                    reference=leaves(reference);torch.cuda.synchronize();assert len(saved)==len(reference)
                    rows=[dict(shape=list(x.shape),exact=torch.equal(x,y),max_abs=(x.float()-y.float()).abs().max().item()) for x,y in zip(saved,reference)]
                    audit=dict(step=state['step'],query_shape=desc.query_shape,tensors=rows,exact_all=all(r['exact'] for r in rows))
                    audits.append(audit);(root/'live-forward-audit.json').write_text(json.dumps(audits,indent=2))
                    assert audit['exact_all'],audit
                    audit_count[desc.query_shape]+=1
                    # Restore all capture-owned outputs/workspace after reference.
                    value=original_call(wrapper,*a,**kw)
            return value
        CUDAGraphWrapper.__call__=call
        return result

