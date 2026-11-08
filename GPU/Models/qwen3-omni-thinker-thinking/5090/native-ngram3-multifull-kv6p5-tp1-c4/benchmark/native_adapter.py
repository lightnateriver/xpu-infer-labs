"""Owned descendant audit for Nsight's separate application session."""
import json
import os
from pathlib import Path


def command_native(args, receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    assert args[0] == 'docker' and args[2] == receipt['container']
    root = receipt['pid']
    p = Path(f'/proc/{root}')
    assert p.stat().st_uid == receipt['uid'] == os.getuid()
    assert p.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] == receipt['start_ticks']
    if args[1] == 'inspect':
        return json.dumps([{'State': {'Running': True, 'StartedAt': str(receipt['epoch'])}, 'Config': {'Env': []}, 'Runtime': 'native under Nsight'}])
    if args[1] == 'logs':
        return Path(receipt['log']).read_text()
    if args[1] == 'top':
        parents = {}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():
                continue
            try:
                if p.stat().st_uid == receipt['uid']:
                    parents[int(p.name)] = int(p.joinpath('stat').read_text().rsplit(')', 1)[1].split()[1])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        own = {root}
        while True:
            more = {pid for pid, parent in parents.items() if parent in own}
            if more <= own:
                break
            own.update(more)
        return 'PID\n' + '\n'.join(map(str, sorted(own)))
    raise ValueError('Unsupported operation')
