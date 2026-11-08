"""Launch the native n-gram / selective-FP8 / FP8-KV / async TP1 service locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from check_environment import check
from ops import ROOT, api, endpoint_open, require_empty_gpu, save, stop_owned, ticks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=os.environ.get('MODEL_DIR'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--state-dir', type=Path, default=ROOT/'.local/service')
    parser.add_argument('--variant', choices=['c4', 'c3', 'budget90', 'budget95', 'kv6p5', 'kv6p75', 'chunk2048', 'memory'], default='kv6p5')
    args = parser.parse_args()
    assert os.name == 'posix' and Path('/proc').exists(), 'Run on the Linux GPU host'
    assert args.model_dir and args.model_dir.is_dir(), 'Provide --model-dir or MODEL_DIR'
    key = os.environ.get('VLLM_API_KEY')
    assert key and key != 'replace-with-your-local-api-key', 'Set your private local VLLM_API_KEY'
    environment = check()
    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    assert (cuda/'bin/nvcc').is_file() and (cuda/'include').is_dir(), 'CUDA_HOME must contain the CUDA 13 toolkit'
    cxx = os.environ.get('CXX', '/usr/bin/g++-11')
    assert shutil.which(cxx), 'Provide the recorded GCC 11 CXX toolchain'
    nvcc_version = subprocess.check_output([str(cuda/'bin/nvcc'), '--version'], text=True)
    cxx_version = subprocess.check_output([cxx, '-dumpfullversion'], text=True).strip()
    assert 'release 13.0,' in nvcc_version, 'Recorded toolkit is CUDA 13.0'
    assert cxx_version.startswith('11.'), 'Recorded C++ toolchain is GCC 11'
    environment.update(nvcc_version_output=nvcc_version, host_cxx_version=cxx_version)
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    state.chmod(0o700)
    receipt_path = state/'receipt.json'
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if Path(f'/proc/{previous["pid"]}').exists() and ticks(previous['pid']) == previous['start_ticks']:
            raise RuntimeError('Recorded service is still alive; use stop.py explicitly')
    endpoint = f'http://127.0.0.1:{args.port}'
    assert not endpoint_open(endpoint), 'Selected port is already serving'
    require_empty_gpu(args.gpu)
    fingerprints = json.loads((ROOT/'configs/model-fingerprint.json').read_text())
    hashes = []
    for row in fingerprints['files']:
        digest = hashlib.sha256()
        with (args.model_dir/row['file']).open('rb') as stream:
            for block in iter(lambda:stream.read(16*1024*1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        assert actual == row['sha256'], f"Model fingerprint mismatch: {row['file']}"
        hashes.append(dict(file=row['file'], sha256=actual))
    deployment = state/('deployment-'+str(time.time_ns()))
    deployment.mkdir(exist_ok=True)
    variant = json.loads((ROOT/'configs/experiment-plan.json').read_text())['variants'][args.variant]
    config_name = variant['config']
    config = json.loads((ROOT/'configs'/config_name).read_text())
    config.update(model=str(args.model_dir.resolve()), host='127.0.0.1', port=args.port)
    save(deployment/'deploy.json', config)
    for _ in range(3):
        gpu = require_empty_gpu(args.gpu)
        time.sleep(2)
    cache = state/'cache'/hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    graphs = deployment/'graphs'
    graphs.mkdir(exist_ok=True)
    (graphs/'phase.txt').write_text('startup')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(ROOT/'runtime'),
        VLLM_PLUGINS='', VLLM_USE_V2_MODEL_RUNNER='0', MULTIGRAPH_DIR=str(graphs),
        KV_MEMORY_PROBE='1' if args.variant=='memory' else '0', KV_AUDIT_DIR=str(deployment), KV_EXPECTED_DTYPE='fp8_e4m3', KV_EXPECTED_IMPL='FlashInferImpl',
        KV_EXPECTED_GPU_UUID=gpu['uuid'], KV_SCALES_FILE=str(ROOT/'configs/kv-scales.json'),
        VLLM_USE_FLASHINFER_SAMPLER='0', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VLLM_NO_USAGE_STATS='1', PYTORCH_ALLOC_CONF='expandable_segments:True', CUDA_HOME=str(cuda),
        CXX=cxx, NVCC_CCBIN=cxx, CPATH=str(cuda/'include')+':'+str(cuda/'include/cccl'),
        PATH=str(Path(sys.executable).parent)+':'+str(cuda/'bin')+':'+env['PATH'],
        XDG_CACHE_HOME=str(cache), TRITON_CACHE_DIR=str(cache/'triton'),
        FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'), VLLM_CACHE_ROOT=str(cache/'vllm'))
    alias = 'qwen3-omni-thinking'
    config['served_model_name'] = [alias]
    save(deployment/'deploy.json', config)
    cmd = [sys.executable, '-u', '-m', 'vllm.entrypoints.openai.api_server']
    for name, value in config.items():
        flag = '--'+name.replace('_', '-')
        if isinstance(value, bool):
            cmd.append(flag if value else '--no-'+name.replace('_', '-'))
        elif name == 'served_model_name':
            cmd.extend([flag, *value])
        elif isinstance(value, (dict, list)):
            cmd.extend([flag, json.dumps(value, separators=(',', ':'))])
        else:
            cmd.extend([flag, str(value)])
    save(deployment/'command.json', cmd)
    log = state/'server.log'
    with log.open('w') as stream:
        proc = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    receipt = dict(runtime='native', container='qwen3-omni-native-ngram-tp1', variant=args.variant, pid=proc.pid,
        start_ticks=ticks(proc.pid), uid=os.getuid(), epoch=time.time(), gpu=args.gpu,
        gpu_uuid=gpu['uuid'], alias=alias, endpoint=endpoint, log=str(log), deploy_dir=str(deployment))
    save(receipt_path, receipt)
    save(state/'environment.json', environment)
    save(state/'model-verification.json', hashes)
    print(json.dumps(dict(phase='starting', pid=proc.pid, endpoint=endpoint, log=str(log))), flush=True)
    try:
        deadline = time.monotonic()+1800
        while True:
            assert proc.poll() is None, f'Service exited {proc.returncode}; inspect {log}'
            try:
                if any(row['id'] == alias for row in json.loads(api(endpoint, '/v1/models', 3))['data']):
                    break
            except Exception:
                pass
            assert time.monotonic() < deadline, f'Startup timed out; inspect {log}'
            time.sleep(5)
        mixed = json.loads((deployment/'mixed-fp8-loaded-audit.json').read_text())
        kv = json.loads((deployment/'allocated-kv-cache-audit.json').read_text())
        native = json.loads((deployment/'native-runtime-audit.json').read_text())
        assert mixed['selected_layers'] == 96 and native['async_scheduling']
        assert native['runner_class'] == 'GPUModelRunner'
        assert native['model_module'] == 'vllm.model_executor.models.qwen3_omni_moe_thinker'
        assert native['speculative'] == config.get('speculative_config')
        assert native['drafter_class'] == 'NgramProposerGPU'
        captured = json.loads((graphs/'captured-shapes.json').read_text())
        assert captured['count'] == 26
        assert kv['expected_logical_dtype'] == 'torch.float8_e4m3fn' and len(kv['views']) == 48
        print(json.dumps(dict(phase='ready', endpoint=endpoint, gpu=args.gpu,
                              selected_fp8_linears=96, fp8_kv_layers=48, receipt=str(receipt_path))), flush=True)
    except BaseException:
        if Path(f'/proc/{proc.pid}').exists():
            stop_owned(receipt)
        raise


if __name__ == '__main__':
    main()
