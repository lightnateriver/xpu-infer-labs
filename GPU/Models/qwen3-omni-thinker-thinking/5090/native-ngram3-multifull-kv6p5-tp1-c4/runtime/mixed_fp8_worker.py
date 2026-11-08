"""Audit selective FP8 linears while preserving the original model worker."""
import json
import os
from pathlib import Path
import re
import torch
from kv_audit_worker import KVAuditWorker


class MixedFp8AuditWorker(KVAuditWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        selected = []
        routers = []
        heads = []
        unintended = []
        for name, module in self.model_runner.model.named_modules():
            weight = getattr(module, 'weight', None)
            if not isinstance(weight, torch.Tensor):
                continue
            method = getattr(module, 'quant_method', None)
            row = dict(name=name, dtype=str(weight.dtype), shape=list(weight.shape),
                method=type(method).__name__)
            if re.fullmatch(r'language_model\.model\.layers\.\d+\.self_attn\.(qkv_proj|o_proj)', name):
                scale = getattr(module, 'weight_scale', None)
                kernel = getattr(method, 'fp8_linear', None)
                row.update(kernel=type(kernel).__name__,
                    activation_quant_key=str(getattr(method, 'activation_quant_key', None)),
                    output_dtype=str(getattr(method, 'out_dtype', None)),
                    weight_bytes=weight.numel()*weight.element_size(),
                    scale_shape=list(scale.shape) if isinstance(scale, torch.Tensor) else None,
                    scales_finite_positive=bool(torch.isfinite(scale).all() and (scale>0).all()) if isinstance(scale, torch.Tensor) else False)
                assert weight.dtype == torch.float8_e4m3fn, row
                assert row['method'] == 'Fp8PerTensorOnlineLinearMethod', row
                assert row['kernel'] == 'CutlassFP8ScaledMMLinearKernel', row
                assert row['scales_finite_positive'] and row['output_dtype'] == 'torch.bfloat16', row
                selected.append(row)
            elif re.fullmatch(r'language_model\.model\.layers\.\d+\.mlp\.gate', name):
                assert weight.dtype == torch.bfloat16, row
                routers.append(row)
            elif name.endswith('lm_head'):
                assert weight.dtype == torch.bfloat16, row
                heads.append(row)
            elif weight.dtype == torch.float8_e4m3fn:
                unintended.append(row)
        assert len(selected) == 96 and len(routers) == 48 and heads, (len(selected),len(routers),heads)
        assert not unintended, unintended
        audit = dict(scheme='FP8 E4M3 W8A8; static tensorwise weights, dynamic activations, BF16 output',
            selected_layers=len(selected),qkv_layers=sum(x['name'].endswith('qkv_proj') for x in selected),
            o_proj_layers=sum(x['name'].endswith('o_proj') for x in selected),
            selected_weight_bytes=sum(x['weight_bytes'] for x in selected),
            unchanged_expert_layers=48,unchanged_router_layers=len(routers),
            unchanged_lm_heads=heads,selected=selected,routers=routers,
            no_unintended_fp8_weight_layers=True)
        out=Path(os.environ['KV_AUDIT_DIR'])/'mixed-fp8-loaded-audit.json'
        out.write_text(json.dumps(audit,indent=2)+'\n')
        print('MIXED_FP8_LOADED_AUDIT',json.dumps({k:v for k,v in audit.items() if k not in ['selected','routers','unchanged_lm_heads']}),flush=True)
        return result
