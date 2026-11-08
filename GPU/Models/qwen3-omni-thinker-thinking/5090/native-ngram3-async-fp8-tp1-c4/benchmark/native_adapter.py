"""Audit one native process session using the original client's container hooks.

This is explicitly a native process adapter, not evidence of a Docker container.
Request generation, stream parsing, timing, validity and API counter checks stay unchanged.
"""
import json
import os
from pathlib import Path


def command_native(args, receipt_path):
    assert args[0] == 'docker'
    receipt = json.loads(Path(receipt_path).read_text())
    assert args[2] == receipt['container'], 'Only the receipt-owned native service is authorized'
    pid = receipt['pid']
    proc = Path(f'/proc/{pid}')
    assert proc.stat().st_uid == receipt['uid'] == os.getuid()
    ticks = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    assert ticks == receipt['start_ticks'], 'Process PID changed'
    operation = args[1]
    if operation == 'inspect':
        return json.dumps([{'State': {'Running': True, 'StartedAt': str(receipt['epoch'])},
                           'Config': {'Env': []}, 'Runtime': 'native process'}])
    if operation == 'logs':
        return Path(receipt['log']).read_text()
    if operation == 'top':
        own = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid == receipt['uid'] and os.getsid(int(entry.name)) == pid:
                    own.append(int(entry.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        assert pid in own
        return 'PID\n' + '\n'.join(map(str, sorted(own)))
    raise ValueError('Unsupported native process operation: ' + operation)
