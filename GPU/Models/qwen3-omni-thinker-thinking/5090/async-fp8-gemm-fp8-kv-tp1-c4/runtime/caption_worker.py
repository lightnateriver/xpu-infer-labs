"""Compatibility fix for image+video+original-audio DeepStack positions in v0.30."""
import hashlib
import inspect
import json
import textwrap
from vllm_omni.worker.gpu_ar_worker import GPUARWorker

class CaptionBaselineWorker(GPUARWorker):
    def load_model(self,*args,**kwargs):
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
        original_quant_method = CompressedTensorsConfig.get_quant_method
        def audited_quant_method(quant, layer, prefix):
            bad = r're:thinker\.language_model\.model\.layers\.'
            if bad in quant.ignore:
                quant.ignore = [rule for rule in quant.ignore if rule != bad]
                print("NVFP4_REMOVE_OVERBROAD_IGNORE", bad, flush=True)
            method = original_quant_method(quant, layer, prefix)
            if "expert" in prefix:
                print("NVFP4_QUANT_AUDIT", prefix, type(layer).__name__, type(method).__name__, "targets", list(quant.target_scheme_map), flush=True)
                if "Unquantized" in type(method).__name__:
                    raise RuntimeError("NVFP4 experts unexpectedly unquantized: " + prefix)
            return method
        CompressedTensorsConfig.get_quant_method = audited_quant_method
        config=self.vllm_config
        audit={}
        for section,fields in {
            'model_config':['dtype','quantization','max_model_len'],
            'parallel_config':['tensor_parallel_size'],
            'scheduler_config':['max_num_seqs','max_num_batched_tokens','enable_chunked_prefill','async_scheduling','scheduler_cls'],
            'cache_config':['cache_dtype','kv_cache_memory_bytes','enable_prefix_caching'],
            'compilation_config':['mode','cudagraph_mode','cudagraph_capture_sizes'],
            'kernel_config':['moe_backend','linear_backend'],
        }.items():
            obj=getattr(config,section,None)
            audit[section]={field:getattr(obj,field,None) for field in fields}
        print('CAPTION_EFFECTIVE_ENGINE_CONFIG',json.dumps(audit,default=str,sort_keys=True),flush=True)
        from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni_moe_thinker as module
        patched=0
        for cls in list(vars(module).values()):
            if not isinstance(cls,type) or cls.__module__!=module.__name__ or 'embed_input_ids' not in cls.__dict__:continue
            original=cls.__dict__['embed_input_ids']
            source=textwrap.dedent(inspect.getsource(original))
            old='is_vision = is_video.clone()'
            if old not in source:continue
            assert source.count(old)==1 and source.count('super().embed_input_ids(')==1
            fixed=source.replace(old,'is_vision = is_video | (is_mm_device & (input_ids == self.config.image_token_id))')
            # The standalone function needs an explicit equivalent of zero-arg super.
            fixed=fixed.replace('super().embed_input_ids(','super(_caption_mm_original_class, self).embed_input_ids(')
            namespace=dict(original.__globals__,_caption_mm_original_class=cls)
            exec(compile(fixed,'<caption-mm-deepstack-compat>','exec'),namespace)
            method=namespace['embed_input_ids'];method.__qualname__=original.__qualname__
            cls.embed_input_ids=method
            print('CAPTION_DEEPSTACK_IMAGE_VIDEO_MASK_FIX',cls.__name__,'original_sha256='+hashlib.sha256(source.encode()).hexdigest(),'patched_sha256='+hashlib.sha256(fixed.encode()).hexdigest(),flush=True)
            patched+=1
        assert patched==1,'Expected exactly one installed v0.30 DeepStack mask site'
        return super().load_model(*args,**kwargs)
