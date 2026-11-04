"""Benchmark one round of 15 OpenAI-compatible request bodies."""
import argparse
import concurrent.futures
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path


def request(url, body):
    start = time.perf_counter()
    first = None
    finish = None
    usage = {}
    chunks = []
    payload = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first is None:
                        first = time.perf_counter() - start
                    chunks.append(content)
                finish = choice.get("finish_reason") or finish
    raw = "".join(chunks)
    try:
        json.loads(raw)
        direct_json = True
    except ValueError:
        direct_json = False
    return dict(e2e=time.perf_counter() - start, ttft=first, finish_reason=finish,
                usage=usage, direct_json=direct_json, raw_output=raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18096/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bodies = [json.loads(line) for line in args.requests.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(bodies) != 15 or len({json.dumps(body, sort_keys=True) for body in bodies}) != 15:
        parser.error("Provide 15 distinct request bodies")
    warmup = request(args.url, random.Random(args.seed).choice(bodies))
    if warmup["finish_reason"] != "stop":
        raise RuntimeError("Warmup did not finish normally")
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda body: request(args.url, body), bodies))
    wall = time.perf_counter() - start
    completed = sum(row["finish_reason"] == "stop" for row in rows)
    summary = dict(concurrency=args.concurrency, batch_wall_seconds=wall, completed=completed,
                   samples_per_minute=60 * completed / wall,
                   request_e2e_p50=statistics.median(row["e2e"] for row in rows),
                   direct_json=sum(row["direct_json"] for row in rows), results=rows)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(summary, out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
