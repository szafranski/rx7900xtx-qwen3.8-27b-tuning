"""Fill GSQ production and native contexts using the existing test helpers."""
import hashlib
import functools
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys

import gsq_test_models as test


def make_text(count):
    records = [f'Record {i}: tracking_code={hashlib.sha256(str(i).encode()).hexdigest()[:16]}.' for i in range(count)]
    return ('Read these reference records.\n' + '\n'.join(records)
            + '\nReturn the tracking_code of Record 317. Answer only the code.')


def main(out):
    test.old.api = functools.partial(test.old.api, timeout=3600)
    out.mkdir(parents=True, exist_ok=False)
    for port in (8086, 8194):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} busy')
    active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'llama-launcher.service']).returncode == 0
    if active:
        subprocess.run(['systemctl', '--user', 'stop', 'llama-launcher.service'], check=True)
    try:
        for ctx in (147456, 262144):
            label = f'gsq-{ctx}'
            print('START ' + label, flush=True)
            proc, done, watcher, samples = test.start('gsq', ctx, 2, out, label)
            try:
                props = test.old.api('/props')
                assert props['default_generation_settings']['n_ctx'] == ctx
                target = ctx - 8192
                count = target // 20
                for _ in range(6):
                    text = make_text(count)
                    rendered = test.render(text)
                    tokens = test.old.api('/tokenize', {'content': rendered, 'add_special': True})['tokens']
                    if target - 128 <= len(tokens) <= target:
                        break
                    count = max(318, int(count * (target - 64) / len(tokens)))
                if not target - 256 <= len(tokens) <= target:
                    raise RuntimeError(f'Prompt calibration failed: {len(tokens)} vs {target}')
                (out / f'{label}-prompt.txt').write_text(rendered)
                test.append(out, {'phase': 'prompt', 'ctx': ctx, 'tokens': len(tokens), 'memory': test.memory()})
                print(f'PREFILL {label} input={len(tokens)} reserve={ctx - len(tokens)}', flush=True)
                # A real medium chat, allowing the entire remaining reasoning/output budget.
                response = test.chat([{'role': 'user', 'content': text}], max_tokens=8192, temperature=0, id_slot=0)
                choice = response['choices'][0]
                usage = response['usage']
                assert usage['prompt_tokens'] >= len(tokens) - 8, usage
                assert choice['finish_reason'] == 'stop', choice['finish_reason']
                expected = hashlib.sha256(b'317').hexdigest()[:16]
                message = choice['message']
                result = {'phase': 'full_context', 'ctx': ctx, 'prompt_tokens': len(tokens),
                          'response': response, 'retrieval_pass': message.get('content', '').strip() == expected,
                          'has_reasoning': bool(message.get('reasoning_content')), 'memory': test.memory()}
                test.append(out, result)
                print('RESULT ' + json.dumps({k: v for k, v in result.items() if k != 'response'}), flush=True)
                print('USAGE ' + json.dumps(usage), flush=True)
            finally:
                done.set()
                test.old.stop_server(proc)
                watcher.join(timeout=3)
                test.append(out, {'phase': 'memory', 'ctx': ctx, 'samples': samples})
            print('DONE ' + label, flush=True)
    finally:
        if active:
            subprocess.run(['systemctl', '--user', 'start', 'llama-launcher.service'], check=True)
        print('LAUNCHER RESTORED', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        assert 'Record 317: tracking_code=' + hashlib.sha256(b'317').hexdigest()[:16] in make_text(318)
        assert len(make_text(640)) > len(make_text(318))
        print('SELF CHECK OK')
    else:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        main(Path(sys.argv[1]))
