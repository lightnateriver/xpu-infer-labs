"""Local native-process ownership and HTTP helpers; no remote access."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def owned(receipt):
    pid = receipt['pid']
    assert Path(f'/proc/{pid}').stat().st_uid == receipt['uid'] == os.getuid(), 'UID changed'
    assert ticks(pid) == receipt['start_ticks'], 'PID was reused'
    assert os.getsid(pid) == os.getpgid(pid) == pid, 'Unexpected process session'
    members = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid == os.getuid() and os.getsid(int(p.name)) == pid:
                members.append(int(p.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return members


def gpu_info(index):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    row = next(r.split(',') for r in output.splitlines() if int(r.split(',')[0]) == index)
    return dict(index=index, uuid=row[1].strip(), name=row[2].strip(), utilization=int(row[3]))


def gpu_processes(uuid):
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                      '--format=csv,noheader,nounits'], text=True)
    return [dict(uuid=v[0].strip(), pid=int(v[1]), memory_mib=v[2].strip())
            for line in output.splitlines() if len(v := line.split(',')) == 3 and v[0].strip() == uuid]


def require_empty_gpu(index):
    info = gpu_info(index)
    assert '5090' in info['name'], f"Expected RTX 5090, found {info['name']}"
    assert not gpu_processes(info['uuid']), 'Selected GPU already has a compute process; nothing stopped'
    assert info['utilization'] <= 3, 'Selected GPU is busy'
    return info


def api(endpoint, path, timeout=10):
    key = os.environ.get('VLLM_API_KEY')
    assert key and key != 'replace-with-your-local-api-key', 'Set a private VLLM_API_KEY in the environment'
    req = urllib.request.Request(endpoint+path, headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode()


def endpoint_open(endpoint):
    from urllib.parse import urlsplit
    address = urlsplit(endpoint)
    assert address.hostname == '127.0.0.1'
    with socket.socket() as sock:
        sock.settimeout(2)
        return sock.connect_ex((address.hostname, address.port)) == 0


def stop_owned(receipt, timeout=90):
    members = owned(receipt)
    identities = {pid:ticks(pid) for pid in members}
    for pid in sorted(members, reverse=True):
        try:
            assert Path(f'/proc/{pid}').stat().st_uid == os.getuid()
            assert ticks(pid) == identities[pid]
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic()+timeout
    while any(p['pid'] in members for p in gpu_processes(receipt['gpu_uuid'])) or endpoint_open(receipt['endpoint']):
        assert time.monotonic() < deadline, 'Owned service did not fully stop; inspect receipt/log before retrying'
        time.sleep(2)
