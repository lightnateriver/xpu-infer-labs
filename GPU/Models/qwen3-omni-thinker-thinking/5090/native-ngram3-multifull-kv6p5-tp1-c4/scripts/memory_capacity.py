"""Idle-only diagnostic commands and conservative KV capacity accounting."""
import json,time
from pathlib import Path
from ops import api,owned,save,gpu_processes


def memory_command(receipt,identity,command):
    assert receipt['variant']=='memory','Use the separate diagnostic worker'
    members=owned(receipt)
    assert all(v['pid'] in members for v in gpu_processes(receipt['gpu_uuid']))
    metrics=api(receipt['endpoint'],'/metrics')
    gauges=[float(v.rsplit(' ',1)[1]) for v in metrics.splitlines()
        if v.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
    assert gauges and all(v==0 for v in gauges),'Requests must be idle'
    graphs=Path(receipt['deploy_dir'])/'graphs';result=graphs/('memory-result-'+identity+'.json')
    assert not result.exists(),'Do not reuse a diagnostic command ID'
    save(graphs/'memory-command.json',dict(id=identity,command=command))
    deadline=time.monotonic()+90
    while not result.exists():
        assert time.monotonic()<deadline;time.sleep(1)
    return json.loads(result.read_text())


def capacity(graphs):
    records=json.loads((graphs/'memory-probe.json').read_text())
    relevant=[v for v in records if v.get('reset_id')=='diagnostic-start'];assert relevant
    endpoint=json.loads((graphs/'memory-result-diagnostic-end.json').read_text())['after']
    known=endpoint.get('pool_ids_available',False)
    slack=sum(max(0,v['total']-v['allocated']) for k,v in endpoint.get('pools',{}).items() if k!='0,0') if known else 0
    peak=max(v['peak_allocated_bytes' if known else 'peak_reserved_bytes'] for v in relevant)
    external=max(v['non_torch_bytes'] for v in relevant);total=min(v['cuda_total_bytes'] for v in relevant)
    margin=512*2**20;extra=max(0,total-peak-external-slack-margin)
    return dict(total_bytes=total,peak_allocated_or_reserved_bytes=peak,peak_non_torch_bytes=external,
        private_pool_slack_bytes=slack,pool_ids_available=known,safety_margin_bytes=margin,
        conservative_extra_capacity_bytes=extra,eligible_kv_gib=[g for g in [6.5,6.75] if (g-6.25)*2**30<=extra],
        caveat='Eligibility is workload-specific; still validate service startup, warmup and full natural requests.')
