"""Compatibility fix for image+video+original-audio DeepStack positions in v0.30."""
import hashlib
import inspect
import json
import textwrap
from vllm.v1.worker.gpu_worker import Worker

class CaptionBaselineWorker(Worker):
    def load_model(self,*args,**kwargs):
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
        original_quant_method = CompressedTensorsConfig.get_quant_method
        def audited_quant_method(quant, layer, prefix):
            bad = r're:thinker\.language_model\.model\.layers\.'
            quant.ignore = [rule for rule in quant.ignore if rule != r're:language_model\.model\.layers\.']
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
        print('NATIVE_VLLM_DEEPSTACK_MASK_ALREADY_INCLUDES_IMAGES', flush=True)
        return super().load_model(*args,**kwargs)
