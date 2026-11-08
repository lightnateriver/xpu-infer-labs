"""Rebuild the standalone public report from anonymous numeric evidence."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT/'evidence/results.json').read_text(encoding='utf-8'))
esc = lambda s:html.escape(str(s))
tr = lambda row:'<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>'


def main():
    original = DATA['pooled']
    repeat = DATA['repeat_context']['pooled']
    bf16 = DATA['bf16_kv_context']['pooled']
    cases = [('FP8 KV · 本目录记录', original['valid_samples_per_minute'], original['output_tps_per_gpu']),
             ('FP8 KV · 同配置复测', repeat['complete_items_per_minute'], repeat['output_tps_per_gpu']),
             ('BF16 KV · 固定缓存预算', bf16['complete_items_per_minute'], bf16['output_tps_per_gpu'])]
    metrics = [('协议完整输出', '30/30'),
               ('条/分钟', f"{original['valid_samples_per_minute']:.6f}"),
               ('输出 token/s/卡 · 含 Thinking', f"{original['output_tps_per_gpu']:.6f}"),
               ('正式 wall 合计 / s', f"{original['wall_seconds']:.6f}"),
               ('输出 tokens · 含 Thinking', original['output_tokens']),
               ('输入 tokens · 两轮合计', original['input_tokens']),
               ('抢占重算次数 · 两轮合计', original['observed_preemptions_delta'])]
    for title, key, scale in [('TTFT P50 / s', 'ttft_seconds', 1),
                              ('TPOT P50 / ms', 'tpot_seconds', 1000),
                              ('请求 E2E P50 / s', 'request_e2e_seconds', 1),
                              ('样本 E2E P50 / s · 含准备', 'sample_e2e_seconds', 1),
                              ('整批提交至样本完成 P50 / s · 含客户端排队', 'dataset_e2e_seconds', 1)]:
        metrics.append((title, f"{original['medians'][key]*scale:.6f}"))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RTX 5090 · Qwen3-Omni Thinking · 3.024 条/分钟</title><style>
    body{font-family:Inter,"Microsoft YaHei",sans-serif;max-width:1200px;margin:auto;padding:32px 22px;background:#f7f8fa;color:#282d36;line-height:1.7}h1{font-size:31px}h2{border-left:4px solid #c7000b;padding-left:12px;margin-top:32px}.eyebrow{color:#c7000b;font-weight:bold}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card,section{background:white;padding:20px;border:1px solid #e2e6ea;border-radius:12px;margin:14px 0}.card b{display:block;font-size:35px;color:#c7000b}.notice{background:#fff3e8;border-left:4px solid #d68b3a;padding:18px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e5e7eb}th{background:#fff0f0}.scroll{overflow:auto}.barrow{display:grid;grid-template-columns:170px 1fr 80px;gap:10px;align-items:center;margin:18px 0;font-size:12px}.track{height:25px;background:#f0f2f4}.bar{height:100%;background:#c7000b}code{background:#f0f2f4;padding:2px 4px}li{margin:5px 0}.muted{color:#6b7280}@media(max-width:800px){.grid{grid-template-columns:1fr}.barrow{grid-template-columns:125px 1fr 65px}body{padding:18px 10px}}
    </style><header><div class="eyebrow">RTX 5090 32 GiB × 1 · TP1 / C4 · 2026-10-20 实测</div><h1>Qwen3-Omni Thinking：异步调度 + FP8 GEMM + FP8 KV</h1><p>公开脱敏版 · 15 个冻结视频 Caption 样本 × 2 轮 · Thinking 计入全部生成耗时</p></header>'''
    page += f'<div class="grid"><div class="card">完整任务吞吐<b>{original["valid_samples_per_minute"]:.3f} 条/分钟</b>30/30 协议完整，合并两轮总数 / 总 wall</div><div class="card">单卡输出吞吐<b>{original["output_tps_per_gpu"]:.2f} token/s</b>包含 Thinking 与最终 Caption</div></div>'
    page += '<div class="notice"><b>3.024 是历史观测值：</b>同配置另一次两轮复测为 2.815 条/分钟。温度 0、seed 0 不保证动态并发组批下逐 token 相同，输出长度变化会影响条/分钟。此报告不把不同输出长度下的比率解释为单个算子的加速收益，也不宣称 Caption 精度无损。</div>'
    page += '<h2>1. 结果与复测背景</h2><div class="grid">'
    for title, position in [('完整任务吞吐 · 条/分钟', 1), ('输出吞吐 · token/s/卡', 2)]:
        maximum = max(c[position] for c in cases)*1.05
        page += '<section><h3>'+title+'</h3>'
        for row in cases:
            page += f'<div class="barrow"><span>{esc(row[0])}</span><div class="track"><div class="bar" style="width:{row[position]/maximum*100:.3f}%"></div></div><b>{row[position]:.3f}</b></div>'
        page += '</section>'
    page += '</div><section><table><tr><th>本目录历史方案指标</th><th>数值</th></tr>'+''.join(tr(r) for r in metrics)+'</table></section>'
    page += '<section><h3>本目录方案逐轮数据</h3><div class="scroll"><table><tr>'+''.join('<th>'+x+'</th>' for x in ['轮次', '完整输出', 'wall / s', '条/分钟', 'token/s/卡', '输出 tokens', '抢占'])+'</tr>'
    for i, r in enumerate(DATA['rounds'], 1):
        page += tr([i, r['valid'], f"{r['wall_seconds']:.6f}", f"{r['valid_samples_per_minute']:.6f}", f"{r['output_tps_per_gpu']:.6f}", r['output_tokens'], r['observed_preemptions_delta']])
    page += '</table></div><p class="muted">附图中的 BF16 KV 是后续相同 6.25 GiB 缓存预算实验；该路径与本目录部署配置不同，仅作背景。该实验观测运行请求峰值降到 2、排队和抢占增加。三组各有 30 个完整正式输出，但生成长度不同。</p></section>'
    page += '<h2>2. 卡型与软件环境</h2><section><p>RTX 5090 32 GiB 单卡，SM120；Linux x86_64，Python 3.10，CUDA Toolkit 13.0 / nvcc 13.0.48，GCC 11.4.0，原生 Python venv。仓库不提供或假定私有 Docker 镜像。</p><table><tr><th>组件</th><th>已记录版本</th></tr>'
    versions = json.loads((ROOT/'environment/versions.json').read_text())
    page += ''.join(tr([key, value]) for key, value in versions['packages'].items())+'</table><p>版本号未附可验证的包构建 Git commit；它们是环境记录，不是完整依赖锁文件或二进制一致性证明。</p></section>'
    page += '<h2>3. 优化项与实际配置</h2><section><table><tr><th>项目</th><th>配置与作用范围</th></tr>'
    optimizations = [('选择性 FP8 GEMM', '48 QKV + 48 O，共 96 层；E4M3 W8A8；静态 tensorwise 权重 scale、动态激活量化，BF16 输出；实际 CutlassFP8ScaledMMLinearKernel。'),
        ('保持的原精度', '48 层 MoE 专家保持模型原有 NVFP4 CUTLASS；router、lm_head、视觉与音频编码器不纳入该 FP8 替换。'),
        ('异步调度', 'async_scheduling=true；实际 OmniARAsyncScheduler。与 async_chunk 音频流式功能不同，本方案 async_chunk=false。'),
        ('CUDA Graph 与编译', 'enforce_eager=false，mode=3，FULL_AND_PIECEWISE，capture sizes=[1,2,4]，custom_ops=[+rotary_embedding]。'),
        ('长输入 prefill', 'enable_chunked_prefill=true，max_num_batched_tokens=4096，上下文 65536。'),
        ('并发容量', 'TP1，max_num_seqs=4，客户端并发 4。'),
        ('FP8 KV', '48 层 E4M3 KV，固定 6,710,886,400 bytes（6.25 GiB），采用随包保存的原校准 K/V scales。'),
        ('注意力路径', 'FlashInfer / XQA decode，Q 保持 BF16；disable_flashinfer_q_quantization=true。'),
        ('缓存边界', 'prefix caching 关闭，多模态处理器缓存 0；每个请求使用独立媒体缓存命名空间。')]
    page += ''.join(tr(r) for r in optimizations)+'</table><p>以上是组合配置说明。未在本目录做每一项独立的同卡消融，不能将组合吞吐归因给某一项，也不能叠加各项理论收益。</p></section>'
    page += '<h2>4. 性能输入与测试方法</h2><section><p>15 个冻结样本分为 3 组，每组 5 条；每条为约 12–15 秒原视频及同步音轨、目录内全部 3–5 张参考图、完整 system prompt 与预先冻结的 shot_plan/runtime context。视频以 2 fps 采样，use_audio_in_video=true。每轮输入 tokens 合计 419,460，两轮 838,920。公开证据用匿名 sample 编号、哈希和数值保留逐条可核查性；原媒体、业务 prompt 和原始 Caption 不在公开仓库中。</p><p>每轮完整预热 1 条后，以 C4 测试 15 条，两轮共 30 次正式请求。temperature=0、seed=0、enable_thinking=true、max_tokens=16384。预热与启动/编译排除在正式吞吐之外。客户端进行输入文件 SHA256、预检 token 数、Thinking、JSON 结构、流完整性、GPU 归属与 API 请求计数检查；错误或截断不会列为完整输出。</p><ul><li>条/分钟 = 完整正式请求数 / 正式批次 wall × 60。两轮合并使用总数 / 总 wall。</li><li>输出 token/s/卡 = 全部 completion tokens / 总 wall / GPU 数，包含 Thinking 与 Caption，不含输入 tokens。</li><li>TTFT：HTTP 发出到首个文本 token，包含 Thinking。</li><li>TPOT：(请求 E2E − TTFT) / (completion tokens − 1)，包含首 token 后的调度、抢占重算和流结束影响。</li><li>请求 E2E：HTTP 开始到最终 SSE 完成，包括服务端排队、prefill、Thinking、Caption。样本 E2E 另含客户端媒体准备；批次 wall 再含客户端排队。</li><li>P50 从全部 30 个请求重算；两轮均要求 GPU 选卡无其他服务进程、无额外生成请求。</li></ul></section>'
    page += '<h2>5. 匿名逐样本性能</h2><section><div class="scroll"><table><tr>'+''.join('<th>'+s+'</th>' for s in ['轮次', '匿名样本', '参考图', '输入 tokens', '输出 tokens', 'TTFT / s', 'TPOT / ms', '请求 E2E / s'])+'</tr>'
    for r in DATA['samples']:
        page += tr([r['round'], r['sample_id'], r['reference_count'], r['input_tokens'], r['output_tokens'], f"{r['ttft_seconds']:.3f}", f"{r['tpot_seconds']*1000:.3f}", f"{r['request_e2e_seconds']:.3f}"])
    page += '</table></div></section>'
    page += '<h2>6. 质量与复现边界</h2><section><p>协议完整和 JSON 结构有效不等于视频语义正确。本方案没有已经验收的人工 Caption 精度结论，不宣称精度无损。实验使用非官方 Thinking-heretic-NVFP4 衍生权重，不能当作官方 BF16 Thinking 模型的等价精度结果。</p><p>复现 3.024 需要同一冻结媒体、完整 system prompt、shot plan、模型指纹及相容环境；换数据会改变输入、输出长度和性能。新增启动/测试封装做了路径参数化整理，发布过程中未重新跑 GPU 测试；原始测量来自保存的历史运行。核心 worker 和流式请求/计时代码保留原实现，封装适配记录在 evidence/client-provenance.json。</p><p>公开仓库没有机器 IP、登录账号、私有路径、GPU UUID、PID、API key、SSH key、GitHub token、私有镜像地址或原始业务数据。此 HTML 完全独立，无外部脚本、字体或其他 HTML 依赖。</p></section></html>'
    target = ROOT/'report/report.html'
    target.write_text(page, encoding='utf-8', newline='\n')
    print('Wrote report/report.html')


if __name__ == '__main__':
    main()
