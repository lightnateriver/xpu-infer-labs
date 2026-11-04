#!/usr/bin/env python3
"""Submit one synchronous MiniMax-H3 request and verify the resulting MP4."""
import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    metrics = args.output.with_suffix(".json")
    if args.output.suffix.lower() != ".mp4" or args.output.exists() or metrics.exists():
        parser.error("choose an unused .mp4 output path")
    if min(args.width, args.height, args.steps) <= 0 or args.width % 32 or args.height % 32:
        parser.error("steps must be positive and both dimensions must be positive multiples of 32")
    if not 4 <= args.duration <= 15:
        parser.error("duration must be in [4, 15] seconds")
    image_data = args.image.read_bytes() if args.image else None
    seed = args.seed if args.seed is not None else (42 if args.image else 0)
    endpoint = args.url.rstrip("/")
    with urlopen(f"{endpoint}/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    fields = {
        "model": model,
        "prompt": args.prompt_file.read_text(encoding="utf-8").rstrip("\n"),
        "width": str(args.width), "height": str(args.height), "fps": "24",
        "num_inference_steps": str(args.steps), "flow_shift": "12", "seed": str(seed),
        "extra_params": json.dumps({
            "task": "fl2va" if args.image else "t2va",
            "duration": args.duration, "audio_flow_shift": 3.0,
        }),
    }
    if args.image:
        fields.update(guidance_scale="1", quality="lossless")
    else:
        fields["aspect_ratio"] = "16:9"
    boundary = uuid.uuid4().hex
    parts = [
        (f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n').encode()
        for key, value in fields.items()
    ]
    if image_data is not None:
        mime = mimetypes.guess_type(args.image.name)[0]
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            parser.error("use a PNG, JPEG or WebP reference image")
        suffix = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[mime]
        parts.extend([
            (f'--{boundary}\r\nContent-Disposition: form-data; name="input_reference"; filename="reference.{suffix}"\r\n'
             f"Content-Type: {mime}\r\n\r\n").encode(),
            image_data, b"\r\n",
        ])
    parts.append(f"--{boundary}--\r\n".encode())
    request = Request(
        f"{endpoint}/v1/videos/sync", data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    start = time.monotonic()
    with urlopen(request, timeout=args.timeout) as response:
        content = response.read()
        peak = response.headers.get("x-peak-memory-mb")
    elapsed = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    media = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(args.output)]
    ))
    video = next(s for s in media["streams"] if s["codec_type"] == "video")
    audio = next(s for s in media["streams"] if s["codec_type"] == "audio")
    frames = round(args.duration * 24)
    frames += (5 - frames) % 17
    shape = (video["width"], video["height"], video["avg_frame_rate"], int(video["nb_frames"]))
    if shape != (args.width, args.height, "24/1", frames):
        raise RuntimeError(f"Unexpected video shape: {shape}")
    if float(audio["duration"]) < frames / 24 - 0.1:
        raise RuntimeError("Audio stream is shorter than expected")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(args.output), "-f", "null", "-"],
        check=True,
    )
    result = {
        "e2e_seconds": elapsed, "steps": args.steps, "requested_seconds": args.duration,
        "width": shape[0], "height": shape[1], "frames": shape[3], "fps": 24,
        "audio_seconds": float(audio["duration"]), "seed": seed,
        "peak_reserved_mib": float(peak) if peak else None,
        "mp4_sha256": hashlib.sha256(content).hexdigest(),
    }
    metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
