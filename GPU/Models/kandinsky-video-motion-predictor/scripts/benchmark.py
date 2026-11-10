"""Bounded CPU preparation concurrency; unchanged single-model, single-clip GPU path."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import defaultdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time

WORKER_SCORER = None

def rss(pid):
    result = {}
    try:
        fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
        result.update(rss_kib=int(fields['VmRSS'].split()[0]), hwm_kib=int(fields['VmHWM'].split()[0]))
        fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines() if ':' in line)
        result['pss_kib'] = int(fields['Pss'].split()[0])
    except (OSError, KeyError):
        pass
    return result

def init_worker(counter, ready, groups, all_cpus, pinned, handoff, warm_video):
    global WORKER_SCORER
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with counter.get_lock():
        rank = counter.value
        counter.value += 1
    os.sched_setaffinity(0, groups[rank] if pinned else all_cpus)
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(Path(handoff)))
    from motion_only import MotionScorer
    WORKER_SCORER = MotionScorer(load_model=False, max_segments=16)
    WORKER_SCORER.prep({'clip_path': warm_video})
    ready.put(dict(pid=os.getpid(), rank=rank, affinity=sorted(os.sched_getaffinity(0)),
                   torch_threads=torch.get_num_threads(), cuda_initialized=torch.cuda.is_initialized(), memory=rss(os.getpid())))

def probe():
    return os.getpid()

def prepare(item):
    cpu = time.process_time()
    start = time.perf_counter()
    tensor = WORKER_SCORER.prep({'clip_path': item['path']})
    seconds = time.perf_counter() - start
    array = tensor.numpy()
    return dict(index=item['index'], array=array, prepare_s=seconds, cpu_s=time.process_time()-cpu,
                input_sha256=hashlib.sha256(array.tobytes()).hexdigest(), worker_pid=os.getpid(), memory=rss(os.getpid()))

def cpu_times():
    rows = {}
    for line in Path('/proc/stat').read_text().splitlines():
        parts = line.split()
        if parts and parts[0].startswith('cpu') and parts[0][3:].isdigit():
            values = list(map(int, parts[1:9]))
            rows[int(parts[0][3:])] = (sum(values), values[3]+values[4])
    return rows

def choose_groups(allowed):
    before = cpu_times()
    time.sleep(1)
    after = cpu_times()
    busy = {cpu:1-(after[cpu][1]-before[cpu][1])/max(1, after[cpu][0]-before[cpu][0]) for cpu in allowed}
    topology = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET,NODE'], text=True)
    cores = defaultdict(list)
    for line in topology.splitlines():
        if line.startswith('#'):
            continue
        cpu, core, socket_id, node = map(int, line.split(','))
        if cpu in allowed:
            cores[(socket_id, core, node)].append(cpu)
    nodes = defaultdict(list)
    for (_, _, node), siblings in cores.items():
        nodes[node].append((sum(busy[c] for c in siblings)/len(siblings), min(siblings)))
    groups = []
    for node, entries in nodes.items():
        entries.sort()
        for start in range(0, len(entries)-3, 4):
            chunk = entries[start:start+4]
            groups.append(dict(node=node, cpus=[c for _, c in chunk], sampled_busy=sum(b for b, _ in chunk)/4))
    groups.sort(key=lambda g:g['sampled_busy'])
    assert len(groups) >= 9
    return groups[:9], topology

def stats(values):
    return dict(count=len(values), sum=sum(values), mean=statistics.mean(values),
                median=statistics.median(values), min=min(values), max=max(values)) if values else None

def decision(score):
    return (score['motion_camera'] < .5 and score['motion_object'] < .5 and score['motion_dynamics'] < 1,
            score['motion_camera'] > 2.5 and score['motion_dynamics'] > 2.5)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--fixture-dir', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    def save(path, obj):
        path.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    physical_gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    for _ in range(3):
        row = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True).splitlines()[physical_gpu]
        fields = row.split(',')
        assert int(fields[1]) < 32 and int(fields[2]) == 0, row
        time.sleep(1)
    allowed = sorted(os.sched_getaffinity(0))
    groups, topology = choose_groups(allowed)
    # Parent receives its own four physical cores, consistent across all configurations.
    os.sched_setaffinity(0, groups[0]['cpus'])
    worker_groups = [g['cpus'] for g in groups[1:]]
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(a.code))
    from motion_only import MotionScorer
    items = [dict(index=i, **json.loads(line)) for i, line in enumerate((a.dataset/'clips.jsonl').read_text().splitlines())]
    assert len(items) == 100 and len({r['clip_id'] for r in items}) == 100
    baseline = {r['clip_id']:r for r in map(json.loads, (a.baseline/'cases.jsonl').read_text().splitlines())}
    assert len(baseline) == 100
    # Per source, choose its shortest and longest duration, covering all ten sources.
    sources = defaultdict(list)
    for item in items:
        sources[item['clip_id'].rsplit('/', 1)[0]].append(item)
    selected = []
    for source in sorted(sources):
        ordered = sorted(sources[source], key=lambda r:(r['duration_s'], r['clip_id']))
        selected.extend([ordered[0], ordered[-1]])
    selected = sorted(selected, key=lambda r:r['index'])
    assert len(selected) == 20 and len({r['clip_id'] for r in selected}) == 20
    warm_video = min(items, key=lambda r:r['duration_s'])['path']
    save(a.output/'environment.json', dict(host=socket.gethostname(), torch=torch.__version__, cuda=torch.version.cuda,
         physical_gpu=physical_gpu, cpu_groups=groups, original_allowed_cpus=allowed,
         parent_affinity=sorted(os.sched_getaffinity(0)), selected=[r['clip_id'] for r in selected],
         worker_threads=4, decoder_threads=2, queue_limit='2*workers', gpu_batch_size=1,
         numa_note='Distinct physical cores, per-worker group within one NUMA node. GPU reports no NUMA locality. No memory binding.',
         topology=topology, cold_start_note='Model once; each pool warmup/startup separately timed and excluded from request loop. File cache uncontrolled.'))
    start = time.perf_counter()
    scorer = MotionScorer(device='cuda:0', max_segments=16)
    torch.cuda.synchronize()
    model_load_s = time.perf_counter()-start
    model_identity = id(scorer.model)
    fixed = torch.from_numpy(np.load(a.fixture_dir/'input.npy', allow_pickle=False))
    reference = json.loads((a.fixture_dir/'tensor-result.json').read_text())['scores']
    for _ in range(4):
        assert scorer.infer_batch([fixed])[0] == reference
    ctx = mp.get_context('spawn')
    summaries = []

    def run_config(workers, pinned, data, name):
        out = a.output/name
        out.mkdir()
        ready = ctx.Queue()
        counter = ctx.Value('i', 0)
        init_start = time.perf_counter()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker,
                                   initargs=(counter, ready, worker_groups, allowed, pinned, str(a.code), warm_video))
        probes = [pool.submit(probe) for _ in range(workers)]
        infos = [ready.get(timeout=240) for _ in range(workers)]
        for future in probes:
            future.result()
        pool_ready_s = time.perf_counter()-init_start
        assert all(not r['cuda_initialized'] for r in infos)
        save(out/'workers.json', infos)
        samples = []
        stop = threading.Event()
        pids = [os.getpid()] + [r['pid'] for r in infos]
        def monitor():
            with (out/'memory.jsonl').open('w') as f:
                while not stop.is_set():
                    per_pid = {pid:rss(pid) for pid in pids}
                    row = dict(time=time.time(), pids=per_pid,
                               total_rss_kib=sum(v.get('rss_kib',0) for v in per_pid.values()),
                               total_pss_kib=sum(v.get('pss_kib',0) for v in per_pid.values()))
                    samples.append(row)
                    f.write(json.dumps(row)+'\n'); f.flush()
                    stop.wait(1)
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        pending, cursor, records = {}, 0, []
        waiting_s, infer_s = 0.0, 0.0
        max_pending = 0
        def submit():
            nonlocal cursor, max_pending
            while cursor < len(data) and len(pending) < 2*workers:
                item = data[cursor]
                pending[pool.submit(prepare, item)] = item
                cursor += 1
            max_pending = max(max_pending, len(pending))
        try:
            submit()
            with (out/'cases.jsonl').open('w') as case_log:
                while pending:
                    before_wait = time.perf_counter()
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    waiting_s += time.perf_counter()-before_wait
                    for future in done:
                        item = pending.pop(future)
                        result = future.result()
                        array = result.pop('array')
                        old = baseline[item['clip_id']]
                        input_exact = result['input_sha256'] == old['input_sha256']
                        assert input_exact, f"Input pixel mismatch: {item['clip_id']}"
                        tensor = torch.from_numpy(array)
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        score = scorer.infer_batch([tensor])[0]
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-t
                        infer_s += elapsed
                        assert id(scorer.model) == model_identity
                        assert all(np.isfinite(v) for v in score.values())
                        result.update(clip_id=item['clip_id'], shape=list(tensor.shape), scores=score,
                                      infer_s=elapsed, input_exact=input_exact,
                                      absolute_error={k:abs(score[k]-old['scores'][k]) for k in score},
                                      threshold_change=decision(score)!=decision(old['scores']))
                        records.append(result)
                        case_log.write(json.dumps(result)+'\n'); case_log.flush()
                        save(a.output/'progress.json', dict(config=name, completed=len(records), total=len(data),
                                                           elapsed_s=time.perf_counter()-start))
                        submit()
            wall = time.perf_counter()-start
            final_allocated = torch.cuda.memory_allocated()
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
        finally:
            stop.set(); monitor_thread.join()
            pool.shutdown(wait=True, cancel_futures=True)
            ready.close(); ready.join_thread()
        summary = dict(name=name, workers=workers, pinned=pinned, count=len(records), pool_ready_s=pool_ready_s,
                       request_wall_s=wall, clips_per_s=len(records)/wall,
                       prep_wall_sum_s=sum(r['prepare_s'] for r in records),
                       prep_cpu_sum_s=sum(r['cpu_s'] for r in records),
                       gpu_consumer_wait_s=waiting_s, inference_sum_s=infer_s,
                       peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                       final_allocated_bytes=final_allocated,
                       sampled_peak_pss_kib=max(r['total_pss_kib'] for r in samples),
                       sampled_peak_rss_kib=max(r['total_rss_kib'] for r in samples),
                       input_hash_matches=sum(r['input_exact'] for r in records),
                       max_absolute_error={k:max(r['absolute_error'][k] for r in records) for k in reference},
                       threshold_changes=sum(r['threshold_change'] for r in records),
                       max_inflight=max_pending, worker_pids=[r['pid'] for r in infos],
                       workers_exited=all(not Path(f"/proc/{r['pid']}").exists() for r in infos))
        save(out/'summary.json', summary)
        summaries.append(summary)
        save(a.output/'summaries.json', summaries)
        print(json.dumps(summary), flush=True)
        return summary

    for workers, pinned in [(1,False), (2,True), (4,False), (4,True), (8,True)]:
        run_config(workers, pinned, selected, f"screen-{workers}-{'pin' if pinned else 'free'}")
    best = min(summaries, key=lambda r:r['request_wall_s'])
    full = run_config(best['workers'], best['pinned'], items, 'full100-best')
    save(a.output/'complete.json', dict(model_initializations=1, model_load_s=model_load_s, fixed_reference_exact=True,
         best_screen=best['name'], full=full, all_configs=len(summaries), cpu_parent_affinity=sorted(os.sched_getaffinity(0))))
    save(a.output/'progress.json', dict(phase='complete', full_count=full['count'], best=best['name']))

if __name__ == '__main__':
    main()
