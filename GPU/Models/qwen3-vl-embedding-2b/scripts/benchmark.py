"""Frozen-manifest video embedding benchmark. Outputs stay private by default."""
import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def initialize_worker(parameters, rows, counter, barrier):
    from qwen_embedding import cpu_decode as cd
    cd._initialize(parameters)
    with counter.get_lock():
        slot = counter.value
        counter.value += 1
    r = rows[slot % len(rows)]
    cd._prepare_tar(r['path'], r['offset'], r['size'], r['sha256'], r['height'], r['width'])
    barrier.wait(timeout=240)


def ready(_):
    return os.getpid()


def warm_pool(ex, parameters, rows):
    from qwen_embedding.cpu_decode import CPUDecodePool
    n = parameters['decode_processes']
    ctx = mp.get_context('spawn')
    counter = ctx.Value('i', 0)
    barrier = ctx.Barrier(n)
    pool = object.__new__(CPUDecodePool)
    pool.executor = ProcessPoolExecutor(max_workers=n, mp_context=ctx,
        initializer=initialize_worker, initargs=(parameters, rows, counter, barrier))
    ex._decode_pool = pool
    list(pool.executor.map(ready, range(n)))
    assert counter.value == n
    return {'workers': counter.value, 'method': 'real clip per worker and process barrier'}


def core_state(worker):
    import torch
    return {'torch_threads': torch.get_num_threads(), 'dtype': str(next(worker.model_runner.model.parameters()).dtype)}


def units(x):
    import numpy as np
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def quality(x, reference):
    import numpy as np
    q, g = units(np.load(reference['query'])), units(np.load(reference['gallery']))
    u = units(x)
    cosine = np.sum(u * q, axis=1)
    mask = ~np.eye(len(x), dtype=bool)
    scores, before = u @ g.T, q @ g.T
    result = {'same_input_cosine_min': float(cosine.min()), 'vectors_below_099': int((cosine < .99).sum()),
        'fixed_gallery_crossings': int((mask & ((scores >= .99) != (before >= .99))).sum()),
        'max_pair_score_delta': float(np.abs(scores - before)[mask].max())}
    if 'stable' in reference:
        stable = np.load(reference['stable'])
        old = units(stable) @ g.T
        result.update(exact_vs_stable_npu=bool(np.array_equal(x, stable)),
            max_abs_vs_stable_npu=float(np.abs(x-stable).max()),
            decision_changes_vs_stable_npu=int((mask & ((scores >= .99) != (old >= .99))).sum()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--device', choices=['cuda:0', 'npu:0'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--references', type=Path, help='Optional PRIVATE reference-map JSON')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--model-sha256', default='c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1')
    a = p.parse_args()
    if a.repeats < 1:
        raise ValueError('repeats must be positive')
    # EngineCore and decode workers must inherit OMP=1. Client MM changes later.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    if a.device == 'npu:0':
        os.environ.setdefault('VLLM_PLUGINS', 'ascend,ascend_model')
        os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
        # Only local, trusted callable RPC is used; do not expose it as a service.
        os.environ['QWEN_COMPILE_SOURCE_DIR'] = str(a.output.resolve() / 'compile-source')
    import numpy as np
    import torch
    from qwen_embedding.video import QwenVLLMExtractor
    parameters = json.loads(a.config.read_text())
    manifest = json.loads(a.manifest.read_text())
    refs = json.loads(a.references.read_text()) if a.references else {}
    rows_all = manifest['warmup'] + [r for rows in manifest['cohorts'].values() for r in rows]
    ids = [r['sample_id'] for r in rows_all]
    if len(ids) != len(set(ids)):
        raise ValueError('Warmup/cohort sample IDs must be disjoint')
    for r in rows_all:
        if not Path(r['path']).is_absolute():
            r['path'] = str((a.manifest.resolve().parent / r['path']).resolve())
        if r['offset'] < 0 or r['size'] <= 0:
            raise ValueError('Invalid payload range')
    if digest(a.model / 'model.safetensors') != a.model_sha256:
        raise ValueError('Checkpoint checksum mismatch')
    a.output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ['torch', 'torchvision', 'vllm', 'vllm-ascend', 'torch-npu', 'transformers', 'qwen-vl-utils', 'av', 'numpy']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    status = {'state': 'initializing', 'parameters': parameters, 'versions': versions,
        'model_sha256': a.model_sha256, 'cohorts': [], 'repeats': a.repeats,
        'timing': 'resident model; reread+SHA256+decode+resize+MM+inference+L2+result materialization; excludes startup/warmup',
        'code_sha256': {str(f.relative_to(Path(__file__).resolve().parents[1])): digest(f)
            for folder in ['runtime', 'scripts'] for f in (Path(__file__).resolve().parents[1]/folder).rglob('*')
            if f.suffix in ['.py', '.cpp']}}
    def save():
        (a.output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    save()
    ex = None
    capture = {'enabled': False, 'records': {}}
    try:
        begun = time.perf_counter()
        ex = QwenVLLMExtractor(dict(parameters, decode_processes=0), a.model, a.device)
        status['startup_s'] = time.perf_counter()-begun
        if a.device == 'npu:0':
            if parameters.get('enforce_eager', True):
                from qwen_embedding.compiled_cuda_rounding import install_compiled_cuda_rounding
                status['precision_patch'] = ex._loop.run_until_complete(ex.model.collective_rpc(install_compiled_cuda_rounding))
            else:
                from qwen_embedding.ascend_graph_worker import inspect_graph
                status['graph_initial'] = ex._loop.run_until_complete(ex.model.collective_rpc(inspect_graph))
            status['core_before'] = ex._loop.run_until_complete(ex.model.collective_rpc(core_state))
            assert all(r['torch_threads'] == 1 for r in status['core_before'])
            renderer = ex.model.renderer
            assert renderer._mm_executor._max_workers == 1
            submit = renderer._mm_executor.submit
            def mm_submit(fn, *args, **kwargs):
                def task():
                    if torch.get_num_threads() != 8:
                        torch.set_num_threads(8)
                    return fn(*args, **kwargs)
                return submit(task)
            renderer._mm_executor.submit = mm_submit
        status['state'] = 'warming_workers'
        save()
        status['pool_warmup'] = warm_pool(ex, parameters, manifest['warmup'])
        def run(rows):
            def prepare(index):
                r = rows[index]
                v = ex.prepare_tar(r['path'], r['offset'], r['size'], r['sha256'], r['height'], r['width'])
                if v.metadata['total_num_frames'] != r['num_frames']:
                    raise ValueError('Frame count differs from frozen manifest')
                if capture['enabled']:
                    h = hashlib.sha256(v.pixels.contiguous().numpy().tobytes())
                    h.update(json.dumps(v.metadata, sort_keys=True).encode())
                    capture['records'][r['sample_id']] = h.hexdigest()
                return v
            vectors = [None]*len(rows)
            contracts = [None]*len(rows)
            begun = time.perf_counter()
            for index, info, vector, _ in ex.encode_rows(rows, prepare,
                    workers=parameters['decode_workers'], batch_size=parameters['batch_clips']):
                vectors[index] = vector
                contracts[index] = {'sample_id': rows[index]['sample_id'], 'frames': info.metadata['frames_indices'], 'shape': info.shape}
            # Mixed submission order is fixed in the released profiles.
            assert parameters['submission_order'] == 'mixed'
            for c, tokens in zip(contracts, ex.last_prompt_tokens, strict=True):
                c['prompt_tokens_sha256'] = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
            x = np.asarray(vectors, dtype=np.float32)
            elapsed = time.perf_counter()-begun
            if x.shape != (len(rows), 2048) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1,atol=2e-6):
                raise ValueError('Invalid vectors')
            return x, contracts, elapsed
        run(manifest['warmup'])
        if a.device == 'npu:0':
            os.environ['OMP_NUM_THREADS'] = '8'  # Existing workers/core remain at one thread.
        for name, rows in manifest['cohorts'].items():
            case = {'name': name, 'count': len(rows), 'rounds': []}
            status['cohorts'].append(case)
            status.update(state='warming_cohort', active_cohort=name)
            save()
            capture.update(enabled=True, records={})
            old_threads = torch.get_num_threads()
            if a.device == 'npu:0':
                torch.set_num_threads(8)
            try:
                _, contract, _ = run(rows)
            finally:
                torch.set_num_threads(old_threads)
                capture['enabled'] = False
            (a.output/f'{name}-boundary.json').write_text(json.dumps(capture['records'],indent=2))
            (a.output/f'{name}-contract.json').write_text(json.dumps(contract,indent=2))
            if name in refs and 'contract' in refs[name]:
                expected = json.loads(Path(refs[name]['contract']).read_text())
                strip = lambda seq: [{k:v for k,v in c.items() if k!='sample_id'} for c in seq]
                if strip(contract) != strip(expected):
                    raise ValueError('Prompt/frame/shape contract differs from frozen reference')
                case['frozen_contract_equal'] = True
            first = None
            status['state'] = 'measuring'
            for rep in range(a.repeats):
                x, current, elapsed = run(rows)
                assert current == contract
                np.save(a.output/f'{name}-{rep}.npy', x)
                rec = {'repeat': rep, 'pipeline_s': elapsed, 'clips_s': len(rows)/elapsed,
                    'byte_equal_vs_first': first is None or bool(np.array_equal(x,first))}
                if first is None:
                    first = x
                if name in refs:
                    rec.update(quality(x, refs[name]))
                case['rounds'].append(rec)
                save()
                print(json.dumps({'cohort': name, **rec}), flush=True)
            case['median_s'] = statistics.median(r['pipeline_s'] for r in case['rounds'])
            case['clips_s'] = len(rows)/case['median_s']
        if a.device == 'npu:0':
            status['core_after'] = ex._loop.run_until_complete(ex.model.collective_rpc(core_state))
            assert status['core_before'] == status['core_after']
            if not parameters.get('enforce_eager', True):
                status['graph_final'] = ex._loop.run_until_complete(ex.model.collective_rpc(inspect_graph))
        status.update(state='complete', finished_at=time.time())
        save()
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        save()
        raise
    finally:
        if ex is not None:
            ex.close()


if __name__ == '__main__':
    main()
