#!/usr/bin/env python3
"""Sequential, durable server A/B benchmark. Uses only the standard library."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import urllib.request

BASE = Path('/home/user/llm')
BUILDS = {'old': BASE / 'llama-cpp-turboquant-current-reasoning/build-vulkan-gfx1100/bin/llama-server',
          'new': BASE / 'llama-cpp-turboquant-tqp-v0.4.0-vulkan/build-vulkan-gfx1100/bin/llama-server'}
MODEL = BASE / 'models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf'
MODES = {'none': 'none', 'mtp': 'draft-mtp', 'combined': 'draft-mtp,ngram-map-k'}
URL = 'http://127.0.0.1:8194'
PROMPTS = {
    'prose': 'Explain how a filesystem journal recovers from a power failure. Discuss ordering, checksums, and limits. Write a detailed technical explanation.',
    'code': 'Write a Python standard-library implementation of an LRU cache with get and put, capacity validation, and a small runnable example. Explain invariants and edge cases.',
}


def api(path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(URL + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def request(prompt, n=256, temperature=0, slot=0):
    payload = {'prompt': prompt, 'n_predict': n, 'temperature': temperature,
               'top_p': .95, 'top_k': 20, 'min_p': 0, 'seed': 42,
               'ignore_eos': temperature == 0, 'cache_prompt': False,
               'stream': True, 'return_tokens': True, 'id_slot': slot}
    start = time.monotonic()
    first = None
    content = ''
    tokens = []
    final = None
    req = urllib.request.Request(URL + '/completion', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                continue
            event = json.loads(raw)
            if 'error' in event:
                raise RuntimeError(event['error'])
            content += event.get('content', '')
            tokens.extend(event.get('tokens', []))
            if first is None and (event.get('content') or event.get('tokens')):
                first = time.monotonic() - start
            if event.get('stop'):
                final = event
    if final is None or 'timings' not in final:
        raise RuntimeError('Missing final timing event')
    if final.get('truncated') or final['timings'].get('cache_n', 0) > 0:
        raise RuntimeError('Truncation or unexpected prompt cache reuse')
    timing = final['timings']
    if temperature == 0 and abs(timing['predicted_n'] - n) > 3:
        raise RuntimeError(f'Unexpected generated length: {timing}')
    return {'elapsed_s': time.monotonic() - start, 'ttft_s': first,
            'timings': timing, 'final': final, 'content': content,
            'output_sha256': hashlib.sha256(content.encode()).hexdigest(), 'tokens': tokens}


def stop_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def start_server(build, mode, out, label):
    args = [str(BUILDS[build]), '--model', str(MODEL), '--mmproj', str(MODEL.parent / 'mmproj-Q8_0.gguf'),
            '--ctx-size', '147456', '--parallel', '2', '--kv-unified', '--n-gpu-layers', '99',
            '--cache-type-k', 'q8_0', '--cache-type-v', 'turbo4',
            '--cache-type-k-draft', 'q8_0', '--cache-type-v-draft', 'turbo4',
            '--batch-size', '4096', '--ubatch-size', '1024', '--threads', '10',
            '--flash-attn', 'on', '--fit', 'off', '--jinja', '--cont-batching',
            '--no-context-shift', '--cache-ram', '0', '--no-cache-prompt',
            '--spec-type', MODES[mode], '--spec-draft-n-max', '3', '--spec-draft-p-min', '.60',
            '--reasoning', 'on', '--reasoning-effort', 'medium', '--reasoning-budget', '8192',
            '--reasoning-format', 'deepseek', '--image-min-tokens', '1024',
            '--host', '127.0.0.1', '--port', '8194', '--timeout', '900', '--metrics']
    env = dict(os.environ, RADV_PERFTEST='nogttspill')
    env.pop('GGML_VK_DISABLE_MMVQ', None)
    (out / (label + '-args.json')).write_text(json.dumps(args, indent=2))
    log = out / (label + '-server.log')
    with log.open('w') as handle:
        proc = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT, env=env)
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f'Server exited {proc.returncode}: {log}')
            try:
                if api('/health', timeout=2).get('status') == 'ok':
                    props = api('/props')
                    (out / (label + '-props.json')).write_text(json.dumps(props, indent=2))
                    return proc, log
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise TimeoutError(f'Server readiness timeout: {log}')
    except BaseException:
        stop_server(proc)
        raise


def chat_prompt(text):
    return api('/apply-template', {'messages': [{'role': 'user', 'content': text}],
                                   'reasoning_effort': 'medium',
                                   'add_generation_prompt': True})['prompt']


def prepare_prompt(kind, depth, out):
    path = out / f'prompt-{kind}-{depth}.json'
    if path.exists():
        value = json.loads(path.read_text())
        if value['template'] != chat_prompt(PROMPTS[kind]):
            raise RuntimeError('Builds render different chat templates')
        return value
    prompt = chat_prompt(PROMPTS[kind])
    ids = api('/tokenize', {'content': prompt, 'add_special': True})['tokens']
    if depth:
        # Fixed non-repetitive document makes prompt lengths exactly reproducible.
        paragraphs = '\n'.join(f'Record {i}: value {i * 7919 % 100003}, checksum {hashlib.sha256(str(i).encode()).hexdigest()[:16]}.' for i in range(depth))
        text = 'Reference records. Use them only as background.\n' + paragraphs + '\n' + PROMPTS[kind]
        full = chat_prompt(text)
        full_ids = api('/tokenize', {'content': full, 'add_special': True})['tokens']
        ids = full_ids[:128] + full_ids[-(depth - 128):]
        assert len(ids) == depth
    value = {'tokens': ids, 'template': prompt, 'length': len(ids), 'kind': kind, 'depth': depth}
    path.write_text(json.dumps(value))
    return value


def summary(out):
    groups = {}
    for line in (out / 'results.jsonl').read_text().splitlines():
        r = json.loads(line)
        key = (r['stage'], r['build'], r['mode'], r['kind'])
        groups.setdefault(key, []).append(r)
    rows = []
    for key, values in groups.items():
        speeds = [v['timings']['predicted_per_second'] for v in values]
        draft = sum(v['timings'].get('draft_n', 0) for v in values)
        accepted = sum(v['timings'].get('draft_n_accepted', 0) for v in values)
        rows.append(dict(zip(('stage', 'build', 'mode', 'kind'), key), n=len(values),
                         tg_median=statistics.median(speeds), tg_min=min(speeds), tg_max=max(speeds),
                         ttft_median=statistics.median(v['ttft_s'] for v in values),
                         elapsed_median=statistics.median(v['elapsed_s'] for v in values),
                         draft_n=draft, accepted=accepted, acceptance=accepted / draft if draft else None,
                         unique_outputs=len({v['output_sha256'] for v in values})))
    (out / 'summary.json').write_text(json.dumps(rows, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['short', '8k', '32k', 'realistic', 'concurrent'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    opts = parser.parse_args()
    out = opts.out
    out.mkdir(parents=True, exist_ok=True)
    stage = opts.stage
    modes = ['none', 'mtp', 'combined'] if stage == 'short' else (['combined'] if stage in ('realistic', 'concurrent') else ['none', 'combined'])
    depth = {'short': 0, '8k': 8192, '32k': 32768, 'realistic': 0, 'concurrent': 8192}[stage]
    reps = 5 if stage == 'short' else 3
    for mode in modes:
        order = ['old', 'new', 'old'] if stage == 'short' else (['new', 'old'] if mode == 'combined' else ['old', 'new'])
        for block, build in enumerate(order):
            label = f'{stage}-{mode}-{build}-{block}'
            print(f'START {label}', flush=True)
            proc, log = start_server(build, mode, out, label)
            try:
                warm = prepare_prompt('prose', 0, out)
                request(warm['tokens'], n=32)
                kinds = ['prose', 'code'] if stage == 'short' else ['prose']
                count = (2 if block == 0 else 3) if stage == 'short' and build == 'old' else reps
                for kind in kinds:
                    p = prepare_prompt(kind, depth, out)
                    for rep in range(count):
                        if stage == 'concurrent':
                            begin = time.monotonic()
                            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                                runs = list(pool.map(lambda slot: request(p['tokens'], slot=slot), [0, 1]))
                            wall = time.monotonic() - begin
                        else:
                            runs = [request(p['tokens'], n=512 if stage == 'realistic' else 256,
                                            temperature=1 if stage == 'realistic' else 0)]
                            wall = runs[0]['elapsed_s']
                        for slot, result in enumerate(runs):
                            result.update(stage=stage, mode=mode, build=build, kind=kind,
                                          block=block, rep=rep, slot=slot, prompt_length=p['length'], batch_wall_s=wall)
                            with (out / 'results.jsonl').open('a') as f:
                                f.write(json.dumps(result) + '\n')
                                f.flush()
                                os.fsync(f.fileno())
                            t = result['timings']
                            print(f'RESULT {label} {kind} {rep} slot={slot} pp={t["prompt_n"]} tg={t["predicted_per_second"]:.2f} draft={t.get("draft_n", 0)} accepted={t.get("draft_n_accepted", 0)}', flush=True)
                        summary(out)
                if mode == 'mtp':
                    evidence = log.read_text(errors='replace')
                    if 'creating MTP draft context' not in evidence:
                        raise RuntimeError('MTP context creation missing from log')
                    if not any(r['draft_n'] > 0 for r in summary(out) if r['stage'] == stage and r['build'] == build and r['mode'] == mode):
                        raise RuntimeError('MTP-only generated no draft tokens')
                (out / (label + '-metrics.txt')).write_text(urllib.request.urlopen(URL + '/metrics', timeout=5).read().decode())
            finally:
                stop_server(proc)
            print(f'DONE {label}', flush=True)
    print(f'COMPLETE {stage}', flush=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
