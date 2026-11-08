"""CPU-only process workers for the unchanged decord/torchvision preparation.

Use spawn even when the parent already owns CUDA. Only resized NumPy arrays
cross the return queue; workers never construct a vLLM engine or CUDA context.
"""
from __future__ import annotations

import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

_extractor = None


def read_tar_payload(path, offset, size, expected_sha256):
    with open(path, "rb") as handle:
        handle.seek(offset)
        payload = handle.read(size)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"TAR payload checksum mismatch: {path}@{offset}")
    return payload


def _initialize(parameters):
    global _extractor
    import torch
    from .video import QwenVLLMExtractor

    torch.set_num_threads(parameters.get("cpu_threads", 1))
    torch.set_num_interop_threads(1)
    _extractor = object.__new__(QwenVLLMExtractor)
    _extractor.parameters = parameters
    _extractor.torch = torch
    _extractor._decode_pool = None


def _ready(value):
    return value


def _prepare(path, height, width, start_s, end_s):
    video = _extractor.prepare(path, height, width, start_s, end_s)
    return video.pixels.numpy(), video.metadata, video.duration


def _prepare_tar(path, offset, size, checksum, height, width):
    return _prepare(read_tar_payload(path, offset, size, checksum), height, width, None, None)


class CPUDecodePool:
    def __init__(self, parameters):
        workers = parameters["decode_processes"]
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError("decode_processes must be a positive integer")
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(dict(parameters),),
        )
        try:
            # Trigger CPU worker startup before submitting real videos.
            list(self.executor.map(_ready, range(workers)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _result(future):
        import torch
        from .video import PreparedVideo

        pixels, metadata, duration = future.result()
        return PreparedVideo(torch.from_numpy(pixels), metadata, duration)

    def prepare(self, path, height, width, start_s, end_s):
        return self._result(self.executor.submit(_prepare, path, height, width, start_s, end_s))

    def prepare_tar(self, path, offset, size, checksum, height, width):
        return self._result(self.executor.submit(_prepare_tar, path, offset, size, checksum, height, width))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
