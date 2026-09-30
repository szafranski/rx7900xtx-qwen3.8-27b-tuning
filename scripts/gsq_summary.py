"""Read-only summary and consistency check of the September GSQ exports."""
import hashlib
import json
from pathlib import Path
import statistics

from build_manifest import records

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
GIB = 1024 ** 3


def rows(name):
    return [json.loads(line) for line in (DATA / f'gsq-2026-09-{name}.jsonl').read_text().splitlines()]


def main():
    expected = {'perf': 24, 'mtp': 4, 'profile': 4, 'full-context': 2}
    for name, count in expected.items():
        assert records(DATA / f'gsq-2026-09-{name}.jsonl') == count
    config = json.loads((DATA / 'gsq-2026-09-config.json').read_text())
    assert len(config['runs']) == 9
    for entry in config['provenance']:
        assert hashlib.sha256((ROOT / entry['export']).read_bytes()).hexdigest() == entry['export_sha256']
    perf = [r for r in rows('perf') if r['phase'] == 'perf']
    for kind in ('prose', 'code'):
        for depth in (8192, 26000):
            for model in ('q4', 'gsq'):
                group = [r for r in perf if (r['kind'], r['depth'], r['model']) == (kind, depth, model)]
                assert len(group) == 3
                assert all(r['timings']['predicted_n'] == 256 and r['timings']['cache_n'] == 0
                           and not r['final'].get('truncated') for r in group)
                print(kind, depth, model, 'input', group[0]['timings']['prompt_n'],
                      'PP', round(statistics.median(r['timings']['prompt_per_second'] for r in group), 2),
                      'decode', round(statistics.median(r['timings']['predicted_per_second'] for r in group), 2),
                      'wall', round(statistics.median(r['elapsed_s'] for r in group), 2))
    full = rows('full-context')
    for r in full:
        if r['phase'] != 'full_context':
            continue
        response = r['response']
        timing = response['timings']
        assert response['choices'][0]['finish_reason'] == 'stop'
        assert timing['cache_n'] == 0 and r['retrieval_pass'] and r['has_reasoning']
        assert r['prompt_tokens'] == timing['prompt_n'] == response['usage']['prompt_tokens']
        samples = next(m['samples'] for m in full if m['phase'] == 'memory' and m['ctx'] == r['ctx'])
        print('context', r['ctx'], 'input', r['prompt_tokens'], 'PP', round(timing['prompt_per_second'], 2),
              'decode', round(timing['predicted_per_second'], 2), 'output', timing['predicted_n'],
              'peak VRAM GiB', round(max(s['vram_used'] for s in samples) / GIB, 2))
    print('SEPTEMBER DATA CHECK OK')


if __name__ == '__main__':
    main()
