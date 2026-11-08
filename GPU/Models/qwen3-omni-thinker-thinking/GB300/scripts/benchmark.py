"""Public portable client. Private prompts/media and business schema are caller supplied."""
import argparse
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
import statistics
import time
import urllib.request
import uuid

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--endpoint', default='http://127.0.0.1:8000')
    p.add_argument('--validator', type=Path)
    a = p.parse_args()
    data = json.loads(a.input.read_text())
    assert len(data['samples']) == 15 and len(data['schedule']) == 56
    assert set(data['schedule']) == set(range(15))
    assert all(data['schedule'].count(i) in (3, 4) for i in range(15))
    assert data['warmup']['messages'] not in [s['messages'] for s in data['samples']]
    a.out.mkdir(parents=True, exist_ok=False)
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['VLLM_API_KEY']}
    validator = None
    if a.validator:
        spec = importlib.util.spec_from_file_location('business_schema', a.validator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        validator = module.validate

    def body(item):
        messages = copy.deepcopy(item['messages'])
        for message in messages:
            if not isinstance(message['content'], list):
                continue
            for part in message['content']:
                kind = part.get('type')
                if kind not in ('image_url', 'video_url'):
                    continue
                url = part[kind]['url']
                if not url.startswith('data:'):
                    assert '://' not in url, 'Use frozen local media, not remote URLs'
                    path = Path(url)
                    part[kind]['url'] = 'data:' + (mimetypes.guess_type(path.name)[0] or 'application/octet-stream') + ';base64,' + base64.b64encode(path.read_bytes()).decode()
                part['uuid'] = uuid.uuid4().hex
        return dict(model='omni-thinking', messages=messages, chat_template_kwargs=dict(enable_thinking=True),
                    mm_processor_kwargs=dict(use_audio_in_video=True), media_io_kwargs=dict(video=dict(fps=2, num_frames=-1)))

    def tokenize(item):
        payload = body(item)
        payload['add_generation_prompt'] = True
        request = urllib.request.Request(a.endpoint + '/tokenize', data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=1800) as response:
            count = json.load(response)['count']
        if 'input_tokens' in item:
            assert count == item['input_tokens']
        assert 0 < count < 65536
        return count

    # Exclude tokenization/media preflight from formal timing, as in historical policy.
    counts = [tokenize(item) for item in data['samples']]
    warmup_count = tokenize(data['warmup'])

    def run(index, item, count):
        started = time.perf_counter()
        folder = a.out / str(index)
        folder.mkdir()
        text, thought, usage, finish, done, first = '', '', {}, None, False, None
        error = None
        request_started = None
        try:
            payload = body(item)
            budget = min(40960, 65536 - count)
            payload.update(modalities=['text'], temperature=0, seed=0, max_tokens=budget,
                           stream=True, stream_options=dict(include_usage=True), return_token_ids=True)
            serialized = json.dumps(payload).encode()
            (folder / 'input-audit.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(serialized).hexdigest(), input_tokens=count, effective_max_output_tokens=budget)))
            req = urllib.request.Request(a.endpoint + '/v1/chat/completions', data=serialized, headers=headers)
            request_started = time.perf_counter()
            with urllib.request.urlopen(req, timeout=3600) as response, (folder / 'stream.sse').open('wb') as raw:
                for line in response:
                    raw.write(line)
                    if not line.startswith(b'data:'):
                        continue
                    chunk = line[5:].strip()
                    if chunk == b'[DONE]':
                        done = True
                        break
                    event = json.loads(chunk)
                    if event.get('usage'):
                        usage = event['usage']
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        content = delta.get('content') or ''
                        reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
                        if (content or reasoning) and first is None:
                            first = time.perf_counter() - request_started
                        text += content
                        thought += reasoning
                        finish = choice.get('finish_reason') or finish
            http_e2e = time.perf_counter() - request_started
            if '</think>' in text:
                prefix, text = text.split('</think>', 1)
                thought += prefix.removeprefix('<think>')
            parsed = json.loads(text)
            schema = bool(validator(text)) if validator else isinstance(parsed, dict)
            valid = bool(done and finish == 'stop' and thought.strip() and schema
                         and usage.get('prompt_tokens') == count
                         and usage.get('completion_tokens', 0) <= budget)
        except Exception as caught:
            error = type(caught).__name__ + ': ' + str(caught)
            http_e2e = time.perf_counter() - request_started if request_started else None
            valid = False
        (folder / 'output.txt').write_text(text, encoding='utf-8')
        (folder / 'thinking.txt').write_text(thought, encoding='utf-8')
        result = dict(index=index, valid=valid, finish_reason=finish, usage=usage, error=error,
                      ttft_seconds=first, request_e2e_seconds=http_e2e,
                      sample_e2e_seconds=time.perf_counter() - started,
                      validator='caller schema' if validator else 'JSON object only')
        (folder / 'metrics.json').write_text(json.dumps(result, indent=2))
        result['sample_e2e_seconds'] = time.perf_counter() - started
        return result

    warmup = run('warmup', data['warmup'], warmup_count)
    assert warmup['valid'], 'Warmup failed; inspect preserved artifacts'
    batch_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=56) as pool:
        futures = [pool.submit(run, j, data['samples'][i], counts[i]) for j, i in enumerate(data['schedule'])]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - batch_started
    valid = sum(r['valid'] for r in results)
    summary = dict(requests=56, valid=valid, batch_wall_seconds=wall, complete_samples_per_minute=60*valid/wall,
                   median_request_e2e_seconds=statistics.median(r['request_e2e_seconds'] for r in results if r['valid']) if valid else None,
                   output_tokens_per_second=sum(r['usage'].get('completion_tokens', 0) for r in results)/wall,
                   telemetry_qualified=False, note='Independently audit GPU isolation/preemptions/dtypes/Graph before qualifying this run')
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
