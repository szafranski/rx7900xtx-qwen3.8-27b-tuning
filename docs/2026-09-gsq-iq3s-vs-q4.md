# September: GSQ IQ3_S vs Q4

Measurements on the same RX 7900 XTX and Ryzen 5 5600, on September 29-30,
2026 UTC. These are a separate series, not replacements for the August results.
GSQ saved about 4.77 GiB of VRAM but did not generally improve throughput.
It also completed an actual 253887-token input with medium reasoning and MTP.

## Build and conditions

TheTom `llama-cpp-turboquant`, tqp-v0.4.0, commit
`bcb85fc3ae85efa0f5f392c6c880dfc524923860`. The full hash was recorded in
the pre-test and deployment notes. The retained server logs independently
report `build 1 (bcb85fc)`, GNU 16.2.1, Vulkan RADV NAVI31. Do not interpret
the self-reported build number 1 as the older August build.

| File | Size in bytes | Provenance |
|---|---:|---|
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17559178144 | Existing unsloth Q4 file; September SHA-256 not recorded |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | 12120016960 | ISTA-DASLab revision `d562806dbafae37109975e970aae91b43e73b440` |
| `mmproj-Q8_0.gguf` | 629247008 | Same existing projector for both variants; September SHA-256 not recorded |

GSQ's SHA-256 was verified after download:
`58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f`.
Q4/projector sizes are the recorded inventory, not new file-integrity checks.

Both models offloaded 66/66 layers. Shared settings include K `q8_0`, V
`turbo4` for model and draft, unified KV, batch 4096, ubatch 1024, threads 10,
flash attention on, fit off, projector loaded, and no context shift or prompt
cache reuse. Main comparisons use `draft-mtp,ngram-map-k`, draft maximum3,
minimum probability0.60, reasoning on, effort medium, and reasoning budget8192.
The scripts set `RADV_PERFTEST=nogttspill` and remove `GGML_VK_DISABLE_MMVQ`.
Other inherited environment variables were not inventoried.

Exact arguments and `/props` snapshots for nine starts are in
[`gsq-2026-09-config.json`](../data/gsq-2026-09-config.json). Complete sanitised
server logs are in [`prompts/gsq-2026-09/logs/`](../prompts/gsq-2026-09/logs/).
Contemporaneous kernel, Mesa, vBIOS readback, CMake cache and GPU power/voltage/
clock profile readback were not retained with these measurements. Do not fill
those gaps with August pins or the machine's present state.

## Throughput A/B

Source: [`gsq-2026-09-perf.jsonl`](../data/gsq-2026-09-perf.jsonl), 24 measured
requests. Context 32768, parallel 1, one client, greedy sampling, seed 42,
256 generated tokens, `ignore_eos=false`. Medium is represented by the rendered
chat prompt, sent as token IDs to `/completion`. Both variants were checked for
identical rendered text and tokenization. Every measured request has cache_n0
and 256 output tokens. Warmup requests are excluded.

Medians of three requests per cell:

| Prompt | Actual input | Q4 PP tok/s | GSQ PP tok/s | Q4 decode tok/s | GSQ decode tok/s | Q4 wall s | GSQ wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prose, about 8K | 7997 | 679.9 | 666.8 | 58.87 | 63.14 | 16.11 | 16.04 |
| Code, about 8K | 8001 | 691.5 | 665.0 | 67.41 | 65.24 | 15.38 | 15.95 |
| Prose, about 26K | 25641 | 585.4 | 569.2 | 58.28 | 53.23 | 48.20 | 49.85 |
| Code, about 26K | 25645 | 585.4 | 569.7 | 61.47 | 57.96 | 48.06 | 49.43 |

Order was Q4 two repeats per cell, GSQ three, Q4 one. This was not a fully
interleaved experiment or a statistical test. GSQ's 8K-prose decode samples
include one low result around 35.2 tok/s after startup and two around 63.1-63.4.
At 26K prose, its samples were about 53.14, 53.23, 60.85 tok/s. All remain in the
raw data. Q4 was more stable in this small sample.

There is no general speedup here. GSQ's 26K decode medians are about 6-9% lower;
that is a descriptive difference, not an established population effect.
The 58-67 tok/s values must not be compared directly to the August 47.6 tok/s
headline: build, workload, context allocation and sampling differ.

## Memory

The source scripts read GPU total/used VRAM from DRM sysfs and available RAM/
used swap from `/proc/meminfo`. A watchdog samples every nominal second from
startup through teardown, without sample timestamps. It terminates the server
below 3 GiB available RAM or 0.5 GiB free VRAM, or after 1 GiB swap growth when
available RAM is below 5 GiB. This is not a GPU-hang or temperature guard.

At 32K, after-load VRAM increase relative to each server's own pre-start baseline
was 18.56 GiB for Q4 and 13.79 GiB for GSQ, saving 4.77 GiB. This is a baseline
delta, not total VRAM.

[`gsq-2026-09-profile.jsonl`](../data/gsq-2026-09-profile.jsonl) has four short
measured requests at ctx 147456, parallel 2. Total sampled peak VRAM after short
requests was 23.04 GiB for Q4 and 18.26 GiB for GSQ, a 4.78 GiB difference.
This test allocated a large context but did not fill it. It does not establish
Q4 capacity for a real 147K input, nor simultaneous long-request capacity.

## Actual long input on GSQ

Source: [`gsq-2026-09-full-context.jsonl`](../data/gsq-2026-09-full-context.jsonl).
One `/v1/chat/completions` request per context, explicit medium, temperature 0,
max_tokens 8192 including reasoning, parallel 2 but only slot 0 active. Unified
KV is shared; parallel 2 is not evidence of two independent full windows.

| Configured context | Actual input | Remaining space | Cold PP tok/s | Prefill time | Decode tok/s | Generated tokens | Total peak VRAM GiB |
|---|---:|---:|---:|---|---:|---:|---:|
| 147456 | 139174 | 8282 | 276.39 | 8 min 23.55 s | 25.36 | 72 | 18.33 |
| 262144 | 253887 | 8257 | 183.30 | 23 min 5.11 s | 18.11 | 80 | 21.70 |

Both returned HTTP 200, finish_reason stop, cache_n 0 and log truncated 0.
Tokenization, usage and timings agree on input length. Both contained a
reasoning trace and returned the correct code for Record 317 near the beginning
of the document, `8d1ede4f889e0ed6`. This is one retrieval check, not a quality
benchmark or evidence that arbitrary tasks work at 254K.

The draft-mtp-specific log counters show 42 and 52 accepted tokens respectively.
Combined speculative acceptance was 52/141 and 55/153. Minimum available RAM
was 25.79 and 24.64 GiB; sampled swap did not grow. At 262K, about 2.29 GiB of the
reported 23.98 GiB total VRAM remained. No OOM appears in the retained server
logs. Kernel-journal evidence for the successful runs is not part of this export,
so no stronger GPU-reset claim is made here.

These runs establish successful processing of nearly 254K actual input on GSQ,
with medium and MTP, followed by a short answer. They did not generate the whole
reserved 8K, test two concurrent long inputs, test images at full context, or
compare Q4 against GSQ on the same 254K input. Decode speed is a single 72/80-token
observation, not a median or a sustained coding benchmark. Cold prefill is the
practical cost: about 23 minutes for 254K.

## MTP compatibility

[`gsq-2026-09-mtp.jsonl`](../data/gsq-2026-09-mtp.jsonl) uses only draft-mtp,
ctx 8192 and parallel 1. Two requests per model accepted 155/148 draft tokens on
Q4 and 137/145 on GSQ. The logs confirm MTP context creation from the main GGUF.
GSQ needs no separate draft file in this setup. This is a compatibility test,
not a measurement of MTP's speedup against no speculation.

## What differs from August

August's pins and headlines remain historical. This round uses a newer build,
a second quantization, different prompts, mostly greedy sampling, and different
context/parallel settings. It has no recorded power or temperature telemetry,
no null pair, no cold perplexity gate, and no preserved per-run GPU-profile
readback or before/after dmesg checks. August's full methodology cannot be
claimed for these runs, and no energy-efficiency conclusion follows.

The early run interrupted by host suspend is excluded. Small quality pilots,
vision/tool smoke tests and reasoning-edge probes are not included as performance
evidence. The production model was not switched by these tests.

## Data and reproduction

All source JSONL rows are retained, including load, prompt and memory records.
The manifest counts only phases perf 24, mtp 4, profile 4, full_context 2; the
configuration file has nine server starts. Memory records are not extra requests.

Only personal home paths were replaced with `/home/user`. Script imports were
relocated to sibling files. No timings, responses, token arrays or memory samples
were removed. `gsq-2026-09-config.json` records original and exported SHA-256
values for data, logs, prompts and script copies. The manifest separately hashes
the exported data files. Rendered long prompts and frozen A/B text/token arrays
are under [`prompts/gsq-2026-09/`](../prompts/gsq-2026-09/).

Read-only checks, without starting a server:

```sh
python3 scripts/gsq_summary.py
python3 scripts/gsq_test_models.py --self-check
python3 scripts/gsq_full_context.py --self-check
bash scripts/check.sh
```

To rerun on a dedicated measurement host, adjust `/home/user/llm` paths and the
DRM card index in `gsq_benchmark.py` and `gsq_test_models.py`. Inspect the scripts
first: they stop an active user llama-launcher service for the test and restore
it afterward, and refuse busy ports 8086/8194. Save a fresh GPU profile, driver/
kernel inventory and build cache before starting. Use an empty output directory.

```sh
python3 scripts/gsq_test_models.py --phase perf --out results/gsq-perf
python3 scripts/gsq_test_models.py --phase mtp --out results/gsq-mtp
python3 scripts/gsq_test_models.py --phase profile --out results/gsq-profile
python3 scripts/gsq_full_context.py results/gsq-full-context
```

The copied harness preserves the source behavior rather than claiming portable
automation. Frozen prompts allow auditing the actual measurements even if a
different build renders or calibrates a different prompt on rerun.
