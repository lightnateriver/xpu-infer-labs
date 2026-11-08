import json
import hashlib
import inspect
import os
from pathlib import Path
import textwrap
from mixed_fp8_worker import MixedFp8AuditWorker


class NativeAuditWorker(MixedFp8AuditWorker):
    def load_model(self,*args,**kwargs):
        from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeThinkerForConditionalGeneration as cls
        from vllm.multimodal.utils import copy_mm_embedding_modality
        original=cls.embed_input_ids
        source=textwrap.dedent(inspect.getsource(original))
        old='multimodal_embeddings[index] = embeddings_main'
        assert source.count(old)==1
        fixed=source.replace(old,'multimodal_embeddings[index] = _copy_mm_embedding_modality(embeddings, embeddings_main)')
        fixed=fixed.replace('super().embed_input_ids(', 'super(_native_mm_cls, self).embed_input_ids(')
        namespace=dict(original.__globals__,_native_mm_cls=cls,_copy_mm_embedding_modality=copy_mm_embedding_modality)
        exec(compile(fixed,'<native-qwen3-omni-preserve-modality>','exec'),namespace)
        cls.embed_input_ids=namespace['embed_input_ids']
        patch=dict(purpose='Preserve existing modality metadata when splitting DeepStack visual embeddings',
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),patched_sha256=hashlib.sha256(fixed.encode()).hexdigest())
        (Path(os.environ['KV_AUDIT_DIR'])/'native-modality-compat.json').write_text(json.dumps(patch,indent=2)+'\n')
        print('NATIVE_DEEPSTACK_PRESERVE_MODALITY',json.dumps(patch),flush=True)
        result=super().load_model(*args,**kwargs)
        cls=type(self.model_runner.get_model())
        self.native_model_identity=dict(model_class=cls.__name__,model_module=cls.__module__)
        (self.audit_dir/'native-model-class-before-compile.json').write_text(json.dumps(self.native_model_identity,indent=2)+'\n')
        assert cls.__module__.startswith('vllm.model_executor.models.'),self.native_model_identity
        return result

    def initialize_from_config(self, config):
        result=super().initialize_from_config(config)
        v=self.vllm_config
        s=v.speculative_config
        audit=dict(**self.native_model_identity,
            compiled_wrapper_class=type(self.model_runner.model).__name__,
            compiled_wrapper_module=type(self.model_runner.model).__module__,
            runner_class=type(self.model_runner).__name__,
            async_scheduling=v.scheduler_config.async_scheduling,
            scheduler_cls=str(v.scheduler_config.scheduler_cls),
            speculative=None if s is None else dict(method=s.method,
                num_speculative_tokens=s.num_speculative_tokens,
                prompt_lookup_min=s.prompt_lookup_min,prompt_lookup_max=s.prompt_lookup_max),
            drafter_class=type(getattr(self.model_runner,'drafter',None)).__name__)
        (self.audit_dir/'native-runtime-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        assert audit['model_module'].startswith('vllm.model_executor.models.'),audit
        assert audit['async_scheduling'] is True
        if s is not None:assert audit['drafter_class']=='NgramProposerGPU'
        (self.audit_dir/'native-runtime-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print('NATIVE_RUNTIME_AUDIT',json.dumps(audit),flush=True)
        return result
