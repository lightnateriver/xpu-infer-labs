"""Python 3.10 protocol-class import compatibility for the text-only API.

Python 3.11 changed dataclass inherited-slot handling. Disable slot generation
only while importing Omni's protocol/duplex classes under Python 3.10.
Model, attention, scheduling, graph and worker code are untouched.
"""
import dataclasses
import sys

original = dataclasses.dataclass
patched_classes = []


def compatible(cls=None, **kwargs):
    def decorate(candidate):
        name = candidate.__module__
        if name.startswith('vllm_omni.protocol.') or name == 'vllm_omni.engine.duplex.commands':
            kwargs['slots'] = False
            patched_classes.append(name + '.' + candidate.__name__)
        return original(candidate, **kwargs)
    return decorate if cls is None else decorate(cls)


if sys.version_info < (3, 11):
    dataclasses.dataclass = compatible
    try:
        import vllm_omni.engine.duplex.commands
    finally:
        dataclasses.dataclass = original
    print('PY310_PROTOCOL_IMPORT_COMPAT', len(patched_classes), flush=True)

from vllm.entrypoints.cli.main import main
if __name__ == '__main__':
    main()
