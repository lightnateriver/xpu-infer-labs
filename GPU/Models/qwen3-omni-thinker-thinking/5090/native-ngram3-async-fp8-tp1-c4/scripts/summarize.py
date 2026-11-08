"""Aggregate complete rounds using total requests/tokens over total batch wall."""
import argparse
import json
from pathlib import Path
import statistics
from ops import save

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--variant', choices=['control', 'ngram3', 'ngram5'], default='ngram3')
    args = parser.parse_args()
    rounds, rows = [], []
    for i in range(1, args.rounds+1):
        folder = args.run_dir/f'runs/native-{args.variant}-c4-r{i}/measured'
        summary = json.loads((folder/'summary.json').read_text())
        audit = json.loads((folder/'api-counter-audit.json').read_text())
        assert summary['valid'] == summary['samples'] == summary['thinking_observed_samples']
        assert summary['failed'] == summary['truncated'] == 0
        assert audit['actual'] == audit['expected'] == summary['samples']
        assert audit['no_other_generation_requests'] and audit['same_container_start']
        rounds.append(summary)
        rows.extend(json.loads((folder/'metrics.json').read_text()))
    wall = sum(r['wall_seconds'] for r in rounds)
    tokens = sum(r['output_tokens'] for r in rounds)
    result = dict(formal_requests=len(rows), protocol_complete=len(rows), wall_seconds=wall,
        complete_items_per_minute=len(rows)/wall*60, output_tokens=tokens,
        output_tokens_per_second_per_gpu=tokens/wall, thinking_included=True,
        preemptions=sum(r['observed_preemptions_delta'] for r in rounds),
        medians={key:statistics.median(r[key] for r in rows) for key in
                 ['ttft_seconds', 'tpot_seconds', 'request_e2e_seconds', 'sample_e2e_seconds', 'dataset_e2e_seconds']},
        caveat='Not directly comparable to recorded throughput unless input/media/prompt hashes match.')
    save(args.run_dir/'pooled-results.json', result)
    print(json.dumps(result, indent=2))
