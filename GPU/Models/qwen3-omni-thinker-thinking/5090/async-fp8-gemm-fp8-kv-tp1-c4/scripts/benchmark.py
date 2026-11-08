"""Run the original streaming benchmark with portable local input paths."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from ops import ROOT, owned, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=os.environ.get('DATA_DIR'))
    parser.add_argument('--prepared-dir', type=Path, default=os.environ.get('PREPARED_DIR'))
    parser.add_argument('--receipt', type=Path, default=ROOT/'.local/service/receipt.json')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--concurrency', type=int, choices=[4], default=4)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    assert args.data_dir and args.data_dir.is_dir(), 'Provide DATA_DIR or --data-dir'
    assert args.prepared_dir and (args.prepared_dir/'manifest.json').is_file(), 'Provide frozen PREPARED_DIR'
    assert args.rounds >= 1 and args.concurrency >= 1
    assert os.environ.get('VLLM_API_KEY'), 'Provide the local API key used to start this service'
    receipt = json.loads(args.receipt.read_text())
    owned(receipt)
    run = args.run_dir or ROOT/'.local'/('benchmark-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    run = run.resolve()
    assert not run.exists(), 'Run directory already exists; choose a fresh directory'
    run.mkdir(parents=True, mode=0o700)
    (run/'prepared').symlink_to(args.prepared_dir.resolve(), target_is_directory=True)
    (run/'capacity').mkdir()
    save(run/'benchmark-plan.json', dict(rounds=args.rounds, concurrency=args.concurrency,
        warmup_per_round=1, thinking=True, protocol='original frozen Caption request/SSE/timing logic',
        historical_comparison_requires_identical_inputs=True))
    env = os.environ.copy()
    env['BENCH_NATIVE_RECEIPT'] = str(args.receipt.resolve())
    base = [sys.executable, '-u', str(ROOT/'benchmark/bench_caption.py'), '--env-file', os.devnull,
            '--run-dir', str(run), '--data-dir', str(args.data_dir.resolve()),
            '--container', receipt['container'], '--endpoint', receipt['endpoint'], '--alias', receipt['alias'],
            '--gpus', str(receipt['gpu']), '--concurrency', str(args.concurrency)]
    with (run/'preflight.log').open('w') as stream:
        subprocess.run(base+['--name', 'input-preflight', '--phase', 'preflight'],
                       env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    for i in range(1, args.rounds+1):
        name = f'fp8-c4-r{i}'
        for phase in ['warmup', 'measured']:
            print(json.dumps(dict(round=i, phase=phase, run_dir=str(run))), flush=True)
            with (run/(name+'-'+phase+'.log')).open('w') as stream:
                subprocess.run(base+['--name', name, '--phase', phase], env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/summarize.py'), '--run-dir', str(run),
                    '--rounds', str(args.rounds)], check=True)


if __name__ == '__main__':
    main()
