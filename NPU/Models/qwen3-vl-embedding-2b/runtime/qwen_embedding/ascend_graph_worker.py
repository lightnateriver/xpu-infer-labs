"""Install the frozen FP32 boundaries before any compile/profile/capture."""
import ast,copy,hashlib,os
from pathlib import Path
from vllm_ascend.worker.worker import NPUWorker

GRAPH_DISPATCH=dict(replay=0,capture=0,bypass=0)


def install_graph_dispatch_probe():
    import vllm_ascend.compilation.acl_graph as module
    original=module.ACLGraphWrapper.__call__
    def call(self,*args,**kwargs):
        ctx=module.get_forward_context()
        if ctx.cudagraph_runtime_mode!=self.runtime_mode:GRAPH_DISPATCH['bypass']+=1
        else:
            entry=self.concrete_aclgraph_entries.get(ctx.batch_descriptor)
            GRAPH_DISPATCH['replay' if entry is not None and entry.aclgraph is not None else 'capture']+=1
        return original(self,*args,**kwargs)
    module.ACLGraphWrapper.__call__=call


def install_deepstack_before_construction():
    import vllm.model_executor.models.qwen3_vl as module
    source=Path(module.__file__).read_text();tree=ast.parse(source)
    cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Qwen3LLMModel')
    fn=copy.deepcopy(next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='forward'));fn.decorator_list=[];changed=0
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Add):
            v=node.value
            if isinstance(v.left,ast.Name) and v.left.id=='hidden_states' and isinstance(v.right,ast.Subscript) and isinstance(v.right.value,ast.Name) and v.right.value.id=='deepstack_input_embeds':
                node.value=ast.BinOp(left=ast.Call(func=ast.Attribute(value=v.left,attr='float',ctx=ast.Load()),args=[],keywords=[]),op=ast.Add(),right=ast.Call(func=ast.Attribute(value=v.right,attr='float',ctx=ast.Load()),args=[],keywords=[]));changed+=1
    assert changed==1
    directory=Path(os.environ['QWEN_COMPILE_SOURCE_DIR']);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'graph_deepstack_forward.py';text=ast.unparse(ast.fix_missing_locations(fn))+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    namespace=dict(vars(module))
    exec(compile(text,str(path),'exec'),namespace);module.Qwen3LLMModel.forward=namespace['forward']
    return dict(sites=changed,generated_source=path.name,generated_sha256=hashlib.sha256(text.encode()).hexdigest())


def install_norms_before_compile(worker):
    import torch
    count=0
    def make(norm):
        def forward(x,residual=None):
            xf=x.float()
            if residual is not None:xf=xf+residual.float();residual=xf.to(norm.weight.dtype)
            variance=(xf*xf).mean(dim=-1,keepdim=True)
            output=(xf*torch.rsqrt(variance+norm.variance_epsilon)*norm.weight.float()).to(norm.weight.dtype)
            return output if residual is None else (output,residual)
        return forward
    for name,norm in worker.model_runner.model.named_modules():
        if name.startswith('language_model.') and type(norm).__name__ in ['RMSNorm','AscendRMSNorm'] and '.self_attn.' not in name:
            assert norm.weight.dtype==torch.bfloat16
            norm._forward_method=make(norm);norm._next_fp32_installed=True;count+=1
    assert count==57
    return dict(norms=count,internal_dtype='float32',gamma='original per-forward FP32 conversion',output_and_residual_dtype='bfloat16',python_counters=False)


class PrecisionGraphWorker(NPUWorker):
    def load_model(self):
        self._next_deepstack=install_deepstack_before_construction()
        super().load_model()
        self._next_norms=install_norms_before_compile(self)
        install_graph_dispatch_probe()


def inspect_graph(worker):
    import dataclasses,torch
    from vllm.compilation.counter import compilation_counter
    cfg=worker.vllm_config
    modules=[]
    for name,m in worker.model_runner.model.named_modules():
        if type(m).__name__=='Qwen3LLMModel':modules.append(dict(name=name,compiled=getattr(m,'compiled',None),do_not_compile=getattr(m,'do_not_compile',None),compiled_fn_type=str(type(getattr(m,'compiled_fn',None)))))
    return dict(worker_class=type(worker).__name__,deepstack=worker._next_deepstack,norms=worker._next_norms,compilation_counters=dataclasses.asdict(compilation_counter),
                cudagraph_mode=str(cfg.compilation_config.cudagraph_mode),compilation_mode=str(cfg.compilation_config.mode),enforce_eager=cfg.model_config.enforce_eager,
                capture_sizes=cfg.compilation_config.cudagraph_capture_sizes,language_modules=modules,torch_threads=torch.get_num_threads(),runtime_dispatch=dict(GRAPH_DISPATCH))
