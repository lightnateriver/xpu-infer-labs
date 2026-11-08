"""Whole-video Qwen embeddings, with official sampling and vLLM pooling.

The official embedding_vllm.ipynb supplies the prompt and pooling contract.
Frame selection uses the official qwen-vl-utils rules; real frame
indices and source FPS travel with the video into vLLM. No window averaging.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .types import Extraction

INSTRUCTION = "Represent the user's input."


def resized_shape(height, width, short_side=320):
    if short_side <= 0 or short_side % 32:
        raise ValueError("Qwen short side must be a positive multiple of 32")
    if min(height, width) <= 0:
        raise ValueError("Invalid video dimensions")
    scale = short_side / min(height, width)
    return tuple(max(32, round(size * scale / 32) * 32) for size in (height, width))



def input_shape(height, width, parameters):
    """Select a size policy without changing frame selection or interpolation."""
    mode = parameters.get("resize_mode", "short_side")
    if mode == "short_side":
        return resized_shape(height, width, parameters["decode_short_side"])
    if mode == "official":
        from qwen_vl_utils.vision_process import smart_resize, VIDEO_MIN_TOKEN_NUM, VIDEO_MAX_TOKEN_NUM
        return smart_resize(height, width, factor=32,
                            min_pixels=VIDEO_MIN_TOKEN_NUM * 32 * 32,
                            max_pixels=VIDEO_MAX_TOKEN_NUM * 32 * 32)
    raise ValueError(f"Unknown resize_mode: {mode}")


def bucket_key(height, width, duration, short_side=320):
    """Exact aligned resolution, and power-of-two duration ranges in seconds."""
    return (*resized_shape(height, width, short_side), 2 ** math.ceil(math.log2(max(1, duration))))


@dataclass
class VideoInfo:
    metadata: dict
    duration: float
    shape: list[int]


@dataclass
class PreparedVideo:
    pixels: object  # TCHW RGB, uint8 or equivalent float32 values in [0, 255].
    metadata: dict
    duration: float

    @property
    def shape(self):
        return list(self.pixels.shape)

    def info(self):
        return VideoInfo(self.metadata, self.duration, self.shape)


def isolate_device(device):
    """Select one physical GPU before any stage initializes CUDA."""
    import os

    import torch

    if not device.startswith("cuda:"):
        raise ValueError("Qwen vLLM requires a CUDA device")
    gpu = int(device.split(":")[1])
    if torch.cuda.is_initialized():
        if gpu != 0 or torch.cuda.device_count() != 1:
            raise ValueError("Initialize Qwen in a fresh process with one visible GPU")
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[gpu] if visible else str(gpu)
    return "cuda:0"


class QwenVLLMExtractor:
    def __init__(self, parameters, model_path, device):
        import os
        from importlib.metadata import version

        import torch

        # Newer Transformers changed video marker expansion. Do not silently
        # produce a different embedding space after a dependency upgrade.
        for package, expected in {
            "vllm": "0.19.0",
            "transformers": "4.57.6",
            "qwen-vl-utils": "0.0.14",
        }.items():
            if version(package) != expected:
                raise RuntimeError(f"Use requirements/qwen-vllm.txt: {package} must be {expected}")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        isolate_device(device)
        from vllm import LLM

        if parameters.get("cpu_threads") is not None:
            torch.set_num_threads(parameters["cpu_threads"])
        self.parameters = parameters
        self.torch = torch
        self._decode_pool = None
        os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
        from qwen_vl_utils import vision_process

        vision_process.FORCE_QWENVL_VIDEO_READER = "decord"
        vision_process.get_video_reader_backend.cache_clear()
        engine_options = dict(
            model=str(model_path),
            runner="pooling",
            dtype=parameters["dtype"],
            trust_remote_code=True,
            max_model_len=parameters["max_model_len"],
            gpu_memory_utilization=parameters["gpu_memory_utilization"],
            max_num_seqs=parameters.get("max_num_seqs", parameters["batch_clips"]),
            max_num_batched_tokens=parameters.get("max_num_batched_tokens", 16384),
            limit_mm_per_prompt={"image": 0, "video": 1},
            mm_processor_kwargs={"do_resize": False, "do_sample_frames": False},
            mm_processor_cache_gb=0,
            enable_prefix_caching=False,
            renderer_num_workers=parameters.get("processor_workers", 1),
            seed=0,
        )
        self._loop = None
        self._request_ids = itertools.count()
        if parameters.get("engine_mode", "sync") == "async":
            from vllm import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            self._loop = asyncio.new_event_loop()

            async def initialize():
                return AsyncLLM.from_engine_args(
                    AsyncEngineArgs(**engine_options, disable_log_stats=True)
                )

            self.model = self._loop.run_until_complete(initialize())
        else:
            self.model = LLM(**engine_options)

        self.prompt = self.model.get_tokenizer().apply_chat_template(
            [
                {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
                {"role": "user", "content": [{"type": "video", "video": "placeholder"}]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if parameters.get("decode_processes", 0):
            from .cpu_decode import CPUDecodePool
            try:
                self._decode_pool = CPUDecodePool(parameters)
            except BaseException:
                self.close()
                raise

    def prepare(self, path, height=None, width=None, start_s=None, end_s=None):
        """Use the official frame-selection math with one bounded decoder.

        qwen-vl-utils opens another reader after our former metadata probe.
        Keeping one reader also lets each decode worker use one FFmpeg thread.
        The resize operation and sampled indices are identical to fetch_video.
        """
        pool = getattr(self, "_decode_pool", None)
        if pool is not None:
            return pool.prepare(path, height, width, start_s, end_s)
        import io
        from .pyav_reader import VideoReader
        from qwen_vl_utils.vision_process import calculate_video_frame_range, smart_nframes
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize

        reader = VideoReader(io.BytesIO(path) if isinstance(path, bytes) else str(path),
                                    num_threads=self.parameters.get("decode_threads", 1))
        count, fps = len(reader), float(reader.get_avg_fps())
        if not np.isfinite(fps) or fps <= 0 or count < 1:
            raise ValueError("Invalid in-memory video" if isinstance(path, bytes) else f"Invalid video: {path}")
        item = dict(
            video="in-memory-video" if isinstance(path, bytes) else str(path),
            fps=self.parameters["sample_fps"],
            max_frames=self.parameters["max_frames"],
        )
        if start_s is not None:
            item["video_start"] = start_s
        if end_s is not None:
            item["video_end"] = end_s
        start, end, selected_count = calculate_video_frame_range(item, count, fps)
        if count == 1:
            indices = [0, 0]  # Same documented single-frame temporal-patch exception.
        else:
            sampled = smart_nframes(item, total_frames=selected_count, video_fps=fps)
            indices = self.torch.linspace(start, end, sampled).round().long().tolist()
        pixels = self.torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
        if height is None or width is None:
            height, width = pixels.shape[-2:]
        height, width = input_shape(height, width, self.parameters)
        pixels = resize(pixels, [height, width], InterpolationMode.BICUBIC, antialias=True)
        if self.parameters.get("frame_transport", "float32") == "float32":
            pixels = pixels.float()
        metadata = dict(
            fps=fps,
            frames_indices=indices,
            total_num_frames=selected_count,
            video_backend="pyav-single-frame" if count == 1 else "pyav",
            do_sample_frames=False,
        )
        return PreparedVideo(pixels, metadata, count / fps)

    def prepare_tar(self, path, offset, size, checksum, height=None, width=None):
        """Read and validate a TAR member in the CPU worker when enabled."""
        pool = getattr(self, "_decode_pool", None)
        if pool is not None:
            return pool.prepare_tar(str(path), offset, size, checksum, height, width)
        from .cpu_decode import read_tar_payload
        return self.prepare(read_tar_payload(path, offset, size, checksum), height, width)

    def _input(self, video):
        return dict(
            prompt=self.prompt,
            multi_modal_data={
                "video": [(video.pixels.permute(0, 2, 3, 1).numpy(), dict(video.metadata))]
            },
        )

    @staticmethod
    def _normalize(vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != 2048 or not np.isfinite(vectors).all():
            raise ValueError("Invalid Qwen embedding")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms < 1e-8).any():
            raise ValueError("Zero Qwen embedding")
        return vectors / norms

    async def _request(self, video):
        from vllm import PoolingParams

        parsed = (await self.model.renderer.render_cmpl_async([self._input(video)]))[0]
        del video
        request_id = str(next(self._request_ids))
        async for output in self.model.encode(parsed, PoolingParams(task="embed"), request_id):
            if output.finished:
                return output.outputs.data.float().cpu().numpy(), list(output.prompt_token_ids)
        raise RuntimeError("vLLM returned no final embedding")

    @staticmethod
    async def _gather(coroutines):
        tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def encode_prepared(self, videos):
        if self._loop is None:
            outputs = self.model.embed([self._input(video) for video in videos], use_tqdm=False)
            self.last_prompt_tokens = [list(out.prompt_token_ids) for out in outputs]
            return self._normalize([out.outputs.embedding for out in outputs])

        async def run():
            semaphore = asyncio.Semaphore(self.parameters.get("request_concurrency", 64))

            async def one(video):
                async with semaphore:
                    return await self._request(video)

            return await self._gather(one(video) for video in videos)

        results = self._loop.run_until_complete(run())
        self.last_prompt_tokens = [tokens for _, tokens in results]
        return self._normalize([vector for vector, _ in results])

    def encode_rows(self, rows, prepare, workers=4, batch_size=None):
        """Bounded decode → preprocess → GPU pipeline, preserving row identity.

        A processing window bounds queued metadata/output vectors; a semaphore
        bounds decoded video buffers and active requests. The next video can
        decode while earlier requests run in the vLLM engine core.
        """
        if self._loop is None:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for indices in bucket_batches(
                    rows,
                    batch_size or self.parameters["batch_clips"],
                    self.parameters["decode_short_side"],
                ):
                    begun = time.perf_counter()
                    videos = list(pool.map(prepare, indices))
                    vectors = self.encode_prepared(videos)
                    infos = [video.info() for video in videos]
                    elapsed = (time.perf_counter() - begun) / len(indices)
                    del videos
                    for index, info, vector in zip(indices, infos, vectors, strict=True):
                        yield index, info, vector, elapsed
            return

        async def window(indices):
            semaphore = asyncio.Semaphore(self.parameters.get("request_concurrency", 64))
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=workers) as pool:

                async def one(index):
                    async with semaphore:
                        begun = time.perf_counter()
                        video = await loop.run_in_executor(pool, prepare, index)
                        info = video.info()
                        request = self._request(video)
                        del video
                        vector, tokens = await request
                        return index, info, vector, tokens, time.perf_counter() - begun

                return await self._gather(one(index) for index in indices)

        window_size = self.parameters.get("processing_window", 512)
        for start in range(0, len(rows), window_size):
            indices = list(range(start, min(start + window_size, len(rows))))
            if self.parameters.get("submission_order", "ascending") != "mixed":
                indices.sort(
                    key=lambda index: estimated_tokens(rows[index], self.parameters),
                    reverse=self.parameters.get("submission_order") == "descending",
                )
            begun = time.perf_counter()
            results = self._loop.run_until_complete(window(indices))
            vectors = self._normalize([item[2] for item in results])
            elapsed = time.perf_counter() - begun
            latency_sum = sum(item[4] for item in results)
            self.last_prompt_tokens = [item[3] for item in results]
            for item, vector in zip(results, vectors, strict=True):
                index, info, _, _, latency = item
                # Amortized share, not a claim of isolated single-clip latency.
                yield index, info, vector, elapsed * latency / latency_sum

    def close(self):
        pool = getattr(self, "_decode_pool", None)
        if pool is not None:
            self._decode_pool = None
            pool.close()
        if self._loop is not None and not self._loop.is_closed():
            self.model.shutdown()

            async def drain():
                tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            self._loop.run_until_complete(drain())
            self._loop.close()

    def extract(self, sample):
        video = self.prepare(
            sample["video_path"], start_s=sample.get("start_s"), end_s=sample.get("end_s")
        )
        vector = self.encode_prepared([video])[0]
        return self.result(sample, video, vector)

    @staticmethod
    def result(sample, video, vector):
        start = sample.get("start_s") or 0
        end = sample.get("end_s") or video.duration
        return Extraction(
            arrays={
                "desc": vector[None],
                "mean": vector,
                "span": np.asarray([[start, end]], np.float32),
                "valid": np.ones(1, bool),
            },
            scores={"n_clips": 1, "n_frames": len(video.metadata["frames_indices"])},
            extra={
                "workflow": "official-whole-video-vllm",
                "input_shape": video.shape,
                "sampled_frames": video.metadata["frames_indices"],
                "source_fps": video.metadata["fps"],
            },
        )

    def extract_many(self, samples):
        """Bounded cross-video batches for ordinary manifests and benchmarks."""

        import av

        rows = []
        for sample in samples:
            with av.open(sample["video_path"]) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                duration = sample.get("duration_s") or float(stream.duration * stream.time_base)
                rows.append(
                    dict(
                        height=stream.height,
                        width=stream.width,
                        fps=fps,
                        num_frames=max(1, round(duration * fps)),
                    )
                )

        def prepare(index):
            sample = samples[index]
            return self.prepare(
                sample["video_path"],
                rows[index]["height"],
                rows[index]["width"],
                sample.get("start_s"),
                sample.get("end_s"),
            )

        for index, info, vector, elapsed in self.encode_rows(
            rows, prepare, workers=self.parameters.get("decode_workers", 4)
        ):
            yield samples[index], self.result(samples[index], info, vector), elapsed


def estimated_tokens(row, parameters):
    """Schedule by sampled work, not raw duration after the 64-frame cap."""
    from qwen_vl_utils.vision_process import smart_nframes

    n = row["num_frames"]
    sampled = (
        2
        if n == 1
        else smart_nframes(
            {"fps": parameters["sample_fps"], "max_frames": parameters["max_frames"]},
            total_frames=n,
            video_fps=row["fps"],
        )
    )
    h, w = input_shape(row["height"], row["width"], parameters)
    return sampled // 2 * (h // 32) * (w // 32) + sampled // 2 * 16 + 64


def bucket_batches(rows, batch_size, short_side=320):
    """Return original row indices, preserving output identity after reordering."""
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        key = bucket_key(row["height"], row["width"], row["num_frames"] / row["fps"], short_side)
        buckets[key].append(index)
    for key in sorted(buckets):
        indices = buckets[key]
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size]
