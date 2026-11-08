"""Keep the FP32 norm/deepstack boundaries observed in the actual GPU graph.

The reference is the frozen optimized GPU run's generated Inductor kernels,
not eager CUDA custom ops (which have different intermediate casts).
"""
def install_compiled_cuda_rounding(worker):
    import ast,copy,hashlib,torch
    from pathlib import Path
    import vllm.model_executor.models.qwen3_vl as module
    assert worker.model_runner.vllm_config.model_config.enforce_eager
    counts={'plain_norm':0,'residual_norm':0,'deepstack_add':0};worker._compiled_rounding_counts=counts
    def norm_forward(norm):
        def forward(x,residual=None):
            assert norm.weight.dtype==torch.bfloat16
            xf=x.float()
            if residual is not None:
                counts['residual_norm']+=1;xf=xf+residual.float();residual=xf.to(norm.weight.dtype)
            else:counts['plain_norm']+=1
            variance=(xf*xf).mean(dim=-1,keepdim=True)
            output=(xf*torch.rsqrt(variance+norm.variance_epsilon)*norm.weight.float()).to(norm.weight.dtype)
            return output if residual is None else (output,residual)
        return forward
    def deepstack_add(left,right):
        counts['deepstack_add']+=1
        return left.float()+right.float()
    selected=[]
    for name,norm in worker.model_runner.model.named_modules():
        if name.startswith('language_model.') and type(norm).__name__ in ['RMSNorm','AscendRMSNorm'] and '.self_attn.' not in name:
            assert getattr(norm,'_compiled_call_impl',None) is None
            assert getattr(norm,'variance_size_override',None) is None and not getattr(norm,'bias_loaded',False)
            selected.append({'name':name,'previous':str(norm._forward_method)});norm._forward_method=norm_forward(norm)
    assert len(selected)==57,len(selected)
    path=Path(module.__file__);source=path.read_text();tree=ast.parse(source)
    cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Qwen3LLMModel')
    function=copy.deepcopy(next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='forward'));function.decorator_list=[]
    changed=0
    for node in ast.walk(function):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Add):
            v=node.value
            if isinstance(v.left,ast.Name) and v.left.id=='hidden_states' and isinstance(v.right,ast.Subscript) and isinstance(v.right.value,ast.Name) and v.right.value.id=='deepstack_input_embeds':
                node.value=ast.Call(func=ast.Name(id='_diagnostic_fp32_deepstack_add',ctx=ast.Load()),args=[v.left,v.right],keywords=[]);changed+=1
    assert changed==1,changed
    namespace=dict(vars(module));namespace['_diagnostic_fp32_deepstack_add']=deepstack_add
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'compiled_cuda_fp32_deepstack_boundaries','exec'),namespace)
    module.Qwen3LLMModel.forward=namespace['forward']
    models=[m for m in worker.model_runner.model.modules() if type(m).__name__=='Qwen3LLMModel']
    assert len(models)==1 and getattr(models[0],'_compiled_call_impl',None) is None
    return {'norms':selected,'source_module':module.__name__,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'deepstack_add_sites_changed':changed,'weight_dtype':'bfloat16','internal_norm_and_deepstack_dtype':'float32','qk_fused_rope_and_swiglu':'unchanged'}


def compiled_cuda_rounding_count(worker):
    return dict(worker._compiled_rounding_counts)
