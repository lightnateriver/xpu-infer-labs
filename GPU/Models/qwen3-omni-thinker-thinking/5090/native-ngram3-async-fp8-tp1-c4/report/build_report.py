"""Render a standalone report from anonymous recorded numeric evidence."""
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'evidence/results.json').read_text(encoding='utf-8'))
LABELS={'control':'原生无 n-gram','ngram3':'N3/K3 · 选用','ngram5':'N5/K5 · 未保留'}
esc=lambda value:html.escape(str(value))


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def bars(title,field,factor=1):
    values=[]
    for k in LABELS:
        p=DATA['groups'][k]['pooled']
        v=(p['medians'][field] if field in p['medians'] else p[field])*factor
        values.append((k,v))
    largest=max(v for _,v in values)*1.05
    control=values[0][1]
    result='<section><h3>'+esc(title)+'</h3><p class="muted">括号为相对原生无 n-gram 的观测变化。</p>'
    for key,value in values:
        result+=f'<div class="barrow"><span>{esc(LABELS[key])}</span><div class="track"><div class="bar {key}" style="width:{value/largest*100:.4f}%"></div></div><strong>{value:.3f}<small>{(value/control-1)*100:+.2f}%</small></strong></div>'
    return result+'</section>'


def main():
    n=DATA['groups']['ngram3']['pooled']
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3-Omni · RTX 5090 · Native N3/K3 实践报告</title><style>
    :root{--red:#c7000b;--ink:#252c36;--muted:#626b78}*{box-sizing:border-box}body{margin:auto;max-width:1320px;padding:32px 24px;background:#f6f7f9;color:var(--ink);font:15px/1.75 "Microsoft YaHei",Arial,sans-serif}header{border-top:6px solid var(--red);padding:26px;background:white;border-radius:12px}h1{font-size:32px;margin:8px 0}h2{margin:34px 0 16px;border-left:5px solid var(--red);padding-left:12px;font-size:23px}h3{margin-top:0}.eyebrow{color:var(--red);font-weight:700}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,section{padding:22px;border:1px solid #e1e5ea;background:white;border-radius:12px;margin:14px 0}.card strong{display:block;font-size:30px;color:var(--red);white-space:nowrap}.muted,small{color:var(--muted)}.notice{background:#fff2ed;border-left:4px solid var(--red);padding:18px 22px;margin:20px 0}.barrow{display:grid;grid-template-columns:150px 1fr 110px;gap:12px;align-items:center;margin:22px 0}.track{height:28px;background:#f0f2f5;border-radius:3px;overflow:hidden}.bar{height:100%;background:var(--red)}.bar.control{background:#8792a4}.bar.ngram5{background:#e8969d}.barrow strong{text-align:right}.barrow small{display:block;font-size:11px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid #e6e9ed;text-align:left;vertical-align:top}th{background:#fff0f1;white-space:nowrap}code,pre{background:#f0f2f5;border-radius:4px}code{padding:2px 5px}pre{padding:18px;overflow:auto;font-size:12px}li{margin:8px 0}.badge{background:#fff0f1;color:var(--red);padding:5px 10px;border-radius:20px;font-size:12px}footer{padding:25px 0;color:var(--muted);font-size:12px}@media(max-width:900px){.cards,.grid{grid-template-columns:1fr 1fr}.card strong{font-size:23px}.barrow{grid-template-columns:115px 1fr 85px;font-size:12px}body{padding:16px 10px}}@media(max-width:600px){.cards,.grid{grid-template-columns:1fr}h1{font-size:25px}}@media print{body{background:white;padding:0}.scroll{overflow:visible}section{break-inside:avoid}.cards{grid-template-columns:repeat(4,1fr)}}
    </style><header><div class="eyebrow">GPU INFER LABS · RTX 5090 32 GiB × 1 · TP1 / C4 · 2026-10-21</div><h1>Qwen3-Omni Thinking：原生 vLLM N3/K3 实践报告</h1><p>异步调度 + 选择性 FP8 GEMM + FP8 KV + GPU n-gram；公开脱敏独立版</p><span class="badge">Thinking 全程计入</span> <span class="badge">15 个冻结输入 × 两轮 / 条件</span> <span class="badge">三条件合计 90 次正式请求</span></header>'''
    page+='<div class="cards">'+''.join('<div class="card">'+label+'<strong>'+value+'</strong><small>'+note+'</small></div>' for label,value,note in [
        ('N3 完整任务吞吐',f"{n['valid_samples_per_minute']:.3f} 条/min",'总完成请求 / 总 wall'),
        ('N3 每卡输出吞吐',f"{n['output_tokens_per_second']:.2f} tok/s",'Thinking + Caption，TP1'),
        ('N3 TPOT P50',f"{n['medians']['tpot_seconds']*1000:.2f} ms",'相对原生对照 −13.38%'),
        ('N3 正式性能门槛','30 / 30','不是人工语义正确率')])+'</div>'
    page+='''<div class="notice"><b>结论：</b>保留 N3/K3 作为当前实验配置。相对原生无 n-gram，条/分钟 +2.27%，token TPS +10.72%，但输出 tokens 同时 +8.26%，两轮之间存在波动，<b>未证明稳定生产吞吐收益或 Caption 语义精度无损</b>。N5 的条/分钟仅比 N3 +0.85%，TPOT 反而慢 7.27%，未保留。不能把 token TPS 增长直接等价为任务吞吐或质量提高。</div>'''
    page+='<h2>1. 吞吐、每卡效率与 E2E</h2><div class="grid">'+bars('完整任务吞吐 · 条/分钟 ↑','valid_samples_per_minute')+bars('输出 token/s/卡 ↑ · 包含 Thinking','output_tokens_per_second')+bars('TPOT P50 · ms ↓','tpot_seconds',1000)+bars('样本 E2E P50 · s ↓','sample_e2e_seconds')+'</div>'
    rows=[]
    for key in LABELS:
        p=DATA['groups'][key]['pooled'];m=p['medians']
        rows.append([LABELS[key],f"{p['valid']}/30",f"{p['wall_seconds']:.6f}",p['output_tokens'],f"{p['valid_samples_per_minute']:.6f}",f"{p['output_tokens_per_second']:.6f}",f"{m['ttft_seconds']:.3f}",f"{m['tpot_seconds']*1000:.3f}",f"{m['request_e2e_seconds']:.3f}",f"{m['sample_e2e_seconds']:.3f}",int(p['preemptions'])])
    page+='<section>'+table(['配置','完整','Wall/s','输出 tokens','条/分钟','TPS/卡','TTFT P50/s','TPOT P50/ms','请求 E2E P50/s','样本 E2E P50/s','抢占'],rows)+'</section>'
    page+='<h2>2. 逐轮数据与收益可信度</h2><section>'
    rows=[]
    for key in LABELS:
        for i,r in enumerate(DATA['groups'][key]['rounds'],1):
            rows.append([LABELS[key],i,r['valid'],f"{r['wall_seconds']:.6f}",r['output_tokens'],f"{r['valid_samples_per_minute']:.6f}",f"{r['output_tps_per_gpu']:.6f}",f"{r['medians']['tpot_seconds']*1000:.3f}",int(r['observed_preemptions_delta'])])
    page+=table(['条件','轮次','完整','Wall/s','输出 tokens','条/分钟','TPS/卡','TPOT P50/ms','抢占'],rows)
    page+='''<ul><li>对照 R1/R2：3.239 / 2.954 条/分钟；N3：3.082 / 3.243。N3 不是每轮都更快，两轮合并的 2.27% 优势小于观测波动。</li><li>无 n-gram graph capture sizes=[1,2,4]，N3=[1,2,4,8,16]。比较包含捕获容量变化，不是只切换 n-gram 标志的严格消融。</li><li>N5 capture sizes=[1,2,4,6,12,24]、max=24，也改变了图捕获容量；N5 同日稍后测试，未交错测量；共享主机 CPU/存储等时间变化仍可能影响结果。N3/N5 的 8 个关键 vLLM 源码 SHA 一致。</li><li>temperature=0、seed=0 仍未使动态组批下输出逐 token 相同。任务长度变化会影响条/分钟、TPS、E2E，当前不进行统计显著性或稳定胜出声明。</li></ul></section>'''
    page+='<h2>3. 配置、优化作用范围与实际启用证据</h2><section>'+table(['项目','配置 / 实际证据'],[
        ['运行时','native vLLM 0.30.0；Qwen3OmniMoeThinkerForConditionalGeneration，GPUModelRunner，async=true；不是 OmniAR worker。'],
        ['插件与 runner','VLLM_PLUGINS 设为空字符串；VLLM_USE_V2_MODEL_RUNNER=0，避免已安装 Omni 插件注册覆盖原生模型。'],
        ['GPU n-gram','method=ngram_gpu，N=3、K=3；实际 NgramProposerGPU。无外部 drafter 模型，无所谓 n-gram FP8 权重。'],
        ['选择性 FP8 GEMM','48 QKV + 48 O = 96 层，E4M3 W8A8，静态 tensorwise 权重 scale、动态激活、BF16 输出；实际 Fp8PerTensorOnlineLinearMethod / CutlassFP8ScaledMMLinearKernel。'],
        ['保留精度','48 层 MoE experts 仍 NVFP4；router、lm_head、多模态编码器不进入本次 FP8 替换。'],
        ['FP8 KV','6,710,886,400 bytes（6.25 GiB）；48 层 FP8 E4M3 typed KV views，原冻结 scales，heldout_used=false。'],
        ['Attention','FlashInfer XQA；use_trtllm_attention=true；disable_flashinfer_q_quantization=true，prefill/decode Q 均 BF16。'],
        ['编译 + CUDA Graph','enforce_eager=false，mode=3，FULL_AND_PIECEWISE，custom_ops=[+rotary_embedding]；N3 capture=[1,2,4,8,16]，实际记录 3 个 FULL decode bucket。'],
        ['调度与上下文','TP1 / C4，max_num_seqs=4，max_model_len=65536；chunked prefill=true，max_num_batched_tokens=4096。'],
        ['缓存边界','prefix caching=false，MM processor cache=0，每请求不同媒体 UUID；未借用相同媒体 cache 命中作为加速。'],
        ['多模态兼容修复','原生 DeepStack 视觉 embedding 切分后复制既有 modality metadata；不改变 tensor 数值；仅进程内修改，定位不匹配即失败，保存前后源码 SHA。']])+'<p>组合配置中的 Graph/FP8/异步等没有在这组 native 数据上逐项独立消融，不能分别赋予吞吐提升，更不能相加。柱状图展示实际整套条件的变化。</p></section>'
    page+='<h2>4. 性能数据集与请求协议</h2><section>'+table(['项目','冻结协议'],[
        ['性能集','15 个独立视频，12.04–15.04 秒，总时长约 207.2633 秒；每条使用目录内全部 3–5 张参考图，共 64 张。'],
        ['输入','完整长 system prompt + 冻结 shot_plan/runtime context + 参考图 + 视频及原同步音轨；视频采样 2 fps，use_audio_in_video=true。'],
        ['镜头切分','先前自动生成并冻结；未经人工确认切分准确性。比较时不重新生成镜头表。'],
        ['规模','每轮输入 419,460 tokens；每条件两轮 838,920 input tokens；上下文 65,536，单条预热最大输入 37,321 tokens。'],
        ['生成','temperature=0，seed=0，enable_thinking=true，max_tokens=16384，text modality；return_token_ids=true，SSE usage。'],
        ['预热/正式','每轮完整预热最大输入一条，再并发 4 测 15 条；两轮/条件。启动、编译及预热排除正式 wall。'],
        ['输入一致性','对应匿名样本的 system/semantic messages SHA、媒体 SHA 和输入 token 预检对齐。新业务输入可以运行，但不能声称复现同一数值。'],
        ['资源隔离','启动前目标卡无其他计算服务；正式阶段每秒 GPU 归属采样，API 计数增量恰好 15，服务未重启，无额外生成请求。']])+'</section>'
    page+='''<h2>5. 计时定义：Thinking 包含在哪里</h2><section><ul><li><b>完整任务吞吐</b> = 完整正式请求总数 / 正式整批 wall ×60；两轮按总数/总 wall 合并。</li><li><b>每卡 TPS</b> = Thinking + Caption completion tokens / 总 wall / GPU 数。TP1 下就是单卡总输出速率，不是单请求逐 token 解码速度。</li><li><b>TTFT</b>：HTTP 发出至第一个 Thinking 或正文 token。</li><li><b>TPOT</b>：(请求 E2E − TTFT)/(completion tokens −1)。草拟验证可能一次返回多个 token，所以它是平均 token 时间，不能当作相邻 SSE chunk 的间隔。</li><li><b>请求 E2E</b>：HTTP 开始至最终 SSE，包括服务排队、prefill、Thinking、Caption；<b>样本 E2E</b>另含客户端媒体准备，不含线程池排队；<b>整批 wall</b>含全部客户端排队。</li><li>P50 对全部 30 条逐请求值计算，不是两轮 P50 再平均。失败/截断保留证据，不纳入“完整任务”计数；本次三条件各 30/30 满足性能门槛。</li></ul></section>'''
    page+='<h2>6. 接受率与 N5 未保留原因</h2><section>'
    rows=[]
    for key in ['ngram3','ngram5']:
        a=DATA['acceptance'][key]['pooled']
        rows.append([LABELS[key],int(a['drafts']),int(a['proposed']),int(a['accepted']),f"{a['acceptance_rate']*100:.4f}%",f"{a['accepted_per_draft']:.6f}"])
    page+=table(['条件','草拟轮次','提议 tokens','接受 tokens','accepted/proposed','每轮接受 tokens'],rows)
    page+='''<p>N3 26.42% vs N5 15.86% 的分母不同。更有可比性的每轮接受 tokens 约 0.792 / 0.793，几乎不变；N5 为每轮多投 2 token 付出验证成本，却未增加每轮实际接受长度。实测 TPOT 从 11.48 ms 变为 12.32 ms。此处没有新 GPU counter profiling，不能仅凭该表断言硬件 DRAM 带宽或单个算子是瓶颈。</p></section>'''
    page+='<h2>7. Caption 质量与精度边界</h2><section>'
    rows=[]
    for key in LABELS:
        c=DATA['groups'][key]['structural_contract']
        rows.append([LABELS[key],'30/30','30/30',f"{c['ordered_shot_ids']}/30",f"{c['exact_shot_headers']}/30",f"{c['subjects_closed']}/30",c['caption_characters']])
    page+=table(['配置','Thinking 可观察','基础 JSON/结构','镜头 ID 顺序','精确镜头标题/时间','主体引用闭合','Caption 字符总计'],rows)
    page+='''<div class="notice"><b>这些是格式/结构合同检查，不是 Caption 语义正确率。</b>15 条性能集没有人工逐帧/逐音轨语义真值，没有盲评、独立听音或标注一致性分析；不能给出“精度 100%”或“量化/投机无损”。</div><p>原生无 n-gram 与 N3 的文本长度和具体内容不同；N3 有 1/30 主体引用闭合检查失败、5/30 精确镜头标题/时间检查失败。性能门槛的 JSON 合法与任务结构完整不包含这些更严格的质量合同。既往人工抽查也存在遗漏/幻觉风险，不能用已有音乐分类/H800 的历史准确率替代本次视频 Caption 精度。</p><p>要确定质量影响，需要同输入盲评：视频事实、参考图身份、动作/时间、声音/台词、跨镜头一致性分别标注；评估 Thinking 与最终答案的截断/拒答情况，并保留配对输出。公开包不发布业务 Caption 原文，只保留匿名数值与格式判定。</p></section>'''
    page+='<h2>8. 实践部署与复测方案</h2><section><pre>'+esc('''export MODEL_DIR=/path/to/model
export DATA_DIR=/path/to/private/dataset
export PREPARED_DIR=/path/to/private/frozen-prepared
export CUDA_HOME=/usr/local/cuda-13.0
export CXX=/usr/bin/g++-11
export VLLM_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python3 scripts/check_environment.py
bash scripts/start.sh --gpu 0 --port 8000 --variant ngram3
bash scripts/benchmark.sh --run-dir "$PWD/.local/n3-run" --rounds 2
python3 scripts/stop.py
# 再用相同输入/环境/卡执行 --variant control，保存独立 run-dir
# --variant ngram5 仅作可选对照；默认仍为 N3''')+'</pre><p>新包装脚本检查模型指纹、目标卡空闲、工具链，保存 receipt，启动后核对实际 native runner、FP8 kernel、KV typed view 和 proposer。只停止 receipt 的自有会话；不停止其他用户作业。媒体/prompt、原始 Thinking/Caption、SSE、token IDs、逐条指标写入忽略的 .local，不能直接公开提交。</p><p>包装脚本在整理发布时新增并做静态审查，未再次运行 GPU benchmark；实测核心 Worker、请求/流解析和计时客户端文本保持一致，SHA 在 publication-provenance 中记录。</p></section>'
    page+='<h2>9. 软件环境及可复现边界</h2><section>'+table(['组件','记录值'],[['GPU','RTX 5090 32 GiB，SM120；1 卡 TP1'],['平台','Linux x86_64 / Python 3.10'],['工具链','CUDA Toolkit 13.0 / nvcc 13.0.48 / GCC 11.4.0 / ffmpeg / ffprobe']]+list(json.loads((ROOT/'environment/versions.json').read_text())['packages'].items()))
    page+='''<p>模型为非官方 Thinking-heretic NVFP4 衍生权重；与官方 BF16 Thinking 不等价。模型 fingerprint 沿用已有 10 个文件 SHA，N3/N5 没有新做完整 checkpoint SHA；新启动包装脚本会重新校验指纹。原环境有继承的兼容修改，未保存完整包构建 commit / 私有镜像；版本号不足以证明二进制一致。公开提供 8 个关键已安装源码 SHA 及进程内补丁，不承诺仅 pip 安装相同版本即可复现全部结果。</p></section>'''
    page+='<h2>10. 全部 90 条匿名正式请求指标</h2><section>'
    rows=[]
    for r in DATA['samples']:
        rows.append([LABELS[r['variant']],r['round'],r['sample_id'],r['reference_count'],r['input_tokens'],r['output_tokens'],f"{r['ttft_seconds']:.3f}",f"{r['tpot_seconds']*1000:.3f}",f"{r['request_e2e_seconds']:.3f}",f"{r['sample_e2e_seconds']:.3f}",'是' if r['thinking_observed'] else '否'])
    page+=table(['配置','轮次','匿名样本','图数','输入 tokens','输出 tokens','TTFT/s','TPOT/ms','请求 E2E/s','样本 E2E/s','Thinking'],rows)+'</section>'
    page+='''<footer>公开脱敏版：无主机 IP、账号、私钥、凭证、GPU UUID、原数据来源、业务媒体、system prompt 或 Caption 原文。无外部依赖，可独立离线打开。数值来自已记录实测，不是本次发布重新测试结果。</footer></html>'''
    (ROOT/'report/report.html').write_text(page,encoding='utf-8',newline='\n')
    print(json.dumps(dict(report_bytes=len(page.encode('utf-8')),anonymous_rows=len(DATA['samples']))))


if __name__=='__main__':
    main()
