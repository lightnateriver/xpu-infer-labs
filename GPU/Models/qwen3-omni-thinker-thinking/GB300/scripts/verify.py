"""Verify public evidence arithmetic and exact runtime files, without GPU execution."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def main():
    s = json.loads((ROOT / 'evidence/summary.json').read_text())
    rows = json.loads((ROOT / 'evidence/requests.json').read_text())
    assert len(rows) == 56 and len({r['sample'] for r in rows}) == 15
    assert all(r['finish_reason'] == 'stop' and not r['truncated'] and r['schema_valid'] for r in rows)
    assert all(r['thinking_observed'] and r['gpu_isolation_valid'] for r in rows)
    assert all(r['preflight_input_tokens'] + r['effective_max_output_tokens'] <= 65536 for r in rows)
    assert all(r['usage']['completion_tokens'] <= r['effective_max_output_tokens'] for r in rows)
    assert abs(statistics.median(r['request_e2e_seconds'] for r in rows) - s['e2e_seconds']) < 1e-6
    assert abs(sum(r['usage']['completion_tokens'] for r in rows)/s['batch_wall_seconds'] - s['output_tokens_per_second']) < 1e-6
    assert abs(56*60/s['batch_wall_seconds'] - s['samples_per_minute']) < 1e-9
    for filename, receipt in [('qwen3_omni_moe_thinker.py', 'compatibility-receipt.json'), ('cuda_graph.py', 'cuda-graph-audit-receipt.json')]:
        r = json.loads((ROOT / 'runtime' / receipt).read_text())
        assert hashlib.sha256((ROOT / 'runtime' / filename).read_bytes()).hexdigest() == r['patched_sha256']
    receipt = json.loads((ROOT / 'runtime/compatibility-receipt.json').read_text())
    assert hashlib.sha256((ROOT / 'runtime/codex_graph_audit.py').read_bytes()).hexdigest() == receipt['graph_audit_helper_sha256']
    print('PASS: 56 complete requests; budgets, throughput, latency and pinned runtime hashes')

if __name__ == '__main__':
    main()
