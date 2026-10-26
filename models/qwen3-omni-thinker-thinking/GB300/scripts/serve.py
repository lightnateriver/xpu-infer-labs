"""Portable launcher for the historical C56 configuration; no GPU takeover."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.local/service.json'

def command(args):
    return subprocess.check_output(args, text=True).strip()

def arguments(model, image, gpu, port, cache, name):
    config = json.loads((ROOT / 'configs/deploy.json').read_text())
    args = ['docker', 'run', '-d', '--name', name, '--label', 'gpu-infer-labs.gb300=' + name,
            '--gpus', 'device=' + str(gpu), '--network', 'host', '--ipc', 'host',
            '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
            '-v', str(model) + ':/model:ro', '-v', str(cache) + ':/root/.cache']
    for filename, target in [('qwen3_omni_moe_thinker.py', 'model_executor/models'),
                             ('cuda_graph.py', 'compilation'), ('codex_graph_audit.py', 'compilation')]:
        args += ['-v', str(ROOT / 'runtime' / filename) +
                 ':/usr/local/lib/python3.12/dist-packages/vllm/' + target + '/' + filename + ':ro']
    for env in ['VLLM_API_KEY', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                'VLLM_NO_USAGE_STATS=1', 'DO_NOT_TRACK=1', 'OMP_NUM_THREADS=1', 'TOKENIZERS_PARALLELISM=false']:
        args += ['-e', env]
    args += ['--entrypoint', 'vllm', image, 'serve', '/model', '--host', '127.0.0.1',
             '--port', str(port), '--served-model-name', 'omni-thinking', '--tensor-parallel-size', '1',
             '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.35', '--max-model-len', '65536',
             '--max-num-seqs', '56', '--max-num-batched-tokens', '65536', '--enable-chunked-prefill',
             '--kv-cache-memory-bytes', str(128 * 1024**3), '--kv-cache-dtype', 'fp8',
             '--no-enable-prefix-caching', '--mm-processor-cache-gb', '0',
             '--limit-mm-per-prompt', json.dumps(dict(image=5, video=1, audio=1)),
             '--reasoning-parser', 'qwen3', '--generation-config', 'vllm', '--seed', '0',
             '--trust-request-mm-kwargs', '--no-async-scheduling', '--cudagraph-metrics',
             '--compilation-config', json.dumps(config['compilation']),
             '--quantization', 'nvfp4_per_token', '--quantization-config',
             json.dumps(dict(ignore=config['quantization_ignore']))]
    return args

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--stop', action='store_true')
    a = p.parse_args()
    if a.stop:
        state = json.loads(STATE.read_text())
        info = json.loads(command(['docker', 'inspect', state['id']]))[0]
        assert info['Config']['Labels'].get('gpu-infer-labs.gb300') == state['name']
        command(['docker', 'stop', '-t', '30', state['id']])
        command(['docker', 'rm', state['id']])
        STATE.unlink()
        return
    assert not STATE.exists(), 'Existing receipt: inspect or stop your previous service first'
    assert os.environ.get('VLLM_API_KEY'), 'Set a private VLLM_API_KEY'
    model = Path(os.environ['MODEL_DIR']).resolve()
    assert (model / 'config.json').is_file()
    image = os.environ.get('VLLM_IMAGE', json.loads((ROOT / 'configs/deploy.json').read_text())['image_id'])
    # Compare stock files before overlaying the pinned compatibility/audit patches.
    for filename, target, receipt in [
        ('qwen3_omni_moe_thinker.py', 'model_executor/models', 'compatibility-receipt.json'),
        ('cuda_graph.py', 'compilation', 'cuda-graph-audit-receipt.json')]:
        r = json.loads((ROOT / 'runtime' / receipt).read_text())
        assert hashlib.sha256((ROOT / 'runtime' / filename).read_bytes()).hexdigest() == r['patched_sha256']
        path = '/usr/local/lib/python3.12/dist-packages/vllm/' + target + '/' + filename
        source = command(['docker', 'run', '--rm', '--entrypoint', 'sha256sum', image, path]).split()[0]
        assert source == r['source_sha256'], 'Image source differs: ' + filename
    card = command(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits']).split(',')
    processes = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    assert not any(line.split(',')[0].strip() == card[0].strip() for line in processes.splitlines()), 'GPU occupied'
    assert int(card[1]) > 196 * 1024, 'Insufficient free GPU memory'
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1', a.port))
    cache = ROOT / '.local/cache'
    cache.mkdir(parents=True, exist_ok=True)
    name = 'gpu-infer-gb300-' + uuid.uuid4().hex[:10]
    container = command(arguments(model, image, a.gpu, a.port, cache, name))
    STATE.write_text(json.dumps(dict(id=container, name=name, port=a.port, gpu=a.gpu)))
    print('Container:', name, '\nInspect startup: docker logs -f ' + name)
    print('Wait for readiness, then verify dtype and Graph audits before measuring.')

if __name__ == '__main__':
    main()
