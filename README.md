# vLLM + Gemma 4 + MTP + ManthanQuant on NVIDIA DGX Spark GB10

**The complete GB10 inference stack in one repo: native Gemma 4, MTP speculative decoding, and ManthanQuant KV-cache compression.**

Stock vLLM does not work on the DGX Spark GB10. The Blackwell sm_121 GPU is missing from prebuilt NCCL kernels, CUTLASS FP8 tables, and Ray's memory heuristics. On top of that, Google Gemma 4 model support only exists in vLLM main (PR #38826) and is not in any stable release. This repository bundles everything needed to run Gemma 4 well on a GB10 — all 8 GB10 fixes, the Gemma 4 backport, a from-source fork build with MTP, the ManthanQuant KV-compression patch, ready-to-run launch scripts, and a benchmark suite with real results.

Three pieces, one repo — pick the layers you need:

| Layer | What it adds | Where |
|-------|--------------|-------|
| **Gemma 4** | Model support on GB10 (sm_121 fixes + backport, or native in the fork build) | `patches/`, `scripts/build-from-source.sh` |
| **MTP** | Speculative decoding for **gemma-4-26B-A4B-it** — ~1.6x single-stream decode, fork-only | `scripts/build-from-source.sh`, `scripts/launch-gemma4-mtp.sh` |
| **ManthanQuant** | 3-bit Lloyd-Max KV-cache compression (~5.1x) for longer context on unified memory | `third_party/manthanquant/`, `patches/manthanquant-kv-compression.sh` |

> **From-source build with native Gemma 4 + MTP speculative decoding.**
> [`scripts/build-from-source.sh`](#from-source-build-vllm-0221--gemma-4-mtp) compiles the
> [`atcuality2021/vllm`](https://github.com/atcuality2021/vllm) fork natively for
> aarch64/GB10 (vLLM 0.22.1, torch 2.11.0+cu130). Gemma 4 is then **native** — no
> file-copy backport — and you also get `gemma4_mtp` (MTP speculative decoding) and
> `gemma4_unified` (audio/video), which are **fork-only** and exist in no stock or
> upstream-main vLLM. On the MoE **gemma-4-26B-A4B-it**, MTP delivers a verified
> **~1.6x single-stream decode speedup at ~80% draft acceptance**. The file-copy
> backport below still works for existing 0.18.x installs.

> **ManthanQuant KV-cache compression is vendored in** under
> [`third_party/manthanquant/`](#kv-cache-compression-manthanquant) — no separate
> clone. It compresses the KV cache to 3 bits per element with pure-numpy Lloyd-Max
> on the Grace CPU cores (the GB10 sm_121 path deliberately avoids custom CUDA
> kernels, which collide with Triton at load). Stack it on top of an MTP launch with
> a single `MANTHANQUANT=1`.

---

## Table of Contents

- [Hardware: NVIDIA DGX Spark GB10](#hardware-nvidia-dgx-spark-gb10)
- [Quick Start](#quick-start)
- [From-Source Build (vLLM 0.22.1 + Gemma 4 MTP)](#from-source-build-vllm-0221--gemma-4-mtp)
- [KV Cache Compression (ManthanQuant)](#kv-cache-compression-manthanquant)
- [Full Installation Walkthrough](#full-installation-walkthrough)
- [All GB10 Fixes](#all-gb10-fixes)
- [Gemma 4 Patch Details](#gemma-4-patch-details)
- [Benchmark Results](#benchmark-results)
- [Per-Test Breakdown](#per-test-breakdown)
- [Recommended Models for GB10](#recommended-models-for-gb10)
- [Multi-Node Setup](#multi-node-setup)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Hardware: NVIDIA DGX Spark GB10

| Spec | Value |
|------|-------|
| SoC | NVIDIA Grace Blackwell GB10 |
| Architecture | ARM aarch64 (Grace CPU) + Blackwell GPU (sm_121) |
| GPU Memory | 128 GB unified (shared CPU/GPU) |
| CUDA Compute | sm_121 (Blackwell) |
| CUDA Toolkit | 13.0 |
| Interconnect | 200 GbE QSFP (for multi-node) |
| OS | Ubuntu (aarch64) |

Key differences from datacenter GPUs:
- **Unified memory** -- CPU and GPU share the same 128 GB pool. There is no separate VRAM. `nvidia-smi` reports memory as `[N/A]`.
- **sm_121** -- Not sm_120 or sm_120a. Many prebuilt GPU kernels skip this variant.
- **ARM host** -- aarch64, not x86_64. Some pip wheels are missing or behave differently.
- **Single GPU per node** -- No NVLink, no multi-GPU within a single machine. Multi-node via network only.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USER/vllm-gb10-gemma4.git
cd vllm-gb10-gemma4
./install.sh /path/to/vllm-env
```

This single command applies all GB10 fixes, installs the Gemma 4 patch, and sets up the benchmark tool.

After installation, launch Gemma 4:

```bash
./scripts/launch-gemma4.sh /path/to/gemma-4-31B-it
```

Or launch Qwen3-Omni (recommended for speed):

```bash
./scripts/launch-qwen-omni.sh /path/to/Qwen3-Omni-30B-A3B-Instruct
```

---

## From-Source Build (vLLM 0.22.1 + Gemma 4 MTP)

The [Quick Start](#quick-start) installer grafts Gemma 4 onto an existing vLLM
0.18.x via file copy. That works, but it cannot give you **MTP speculative
decoding** or the **`gemma4_unified`** (audio/video) architecture — those live
only in the [`atcuality2021/vllm`](https://github.com/atcuality2021/vllm) fork,
not in any stock release and not in upstream `main`.

To get them, build the fork from source. The build targets vLLM **0.22.1**,
where the Blackwell/unified-memory assert that tripped the older line is already
fixed upstream:

```bash
./scripts/build-from-source.sh ~/vllm-mtp-env 2
#                              ^venv          ^MAX_JOBS (keep low — see below)
```

What the build does:

1. Clones the fork at a pinned commit (`2a983c79a`, the same revision used in
   production) and verifies `gemma4_mtp.py` is present.
2. Creates an isolated venv and installs **torch 2.11.0+cu130** (the aarch64
   SBSA wheel the fork pins — build and runtime ABIs match).
3. Compiles the CUDA extensions with `TORCH_CUDA_ARCH_LIST="12.0+PTX"`. torch's
   bundled arch list stops at sm_120; the embedded PTX lets the driver
   JIT-compile to GB10's sm_121 at load time.
4. Builds + installs the wheel and asserts `Gemma4MTPModel` and
   `Gemma4ForCausalLM` are registered.

> **RAM safety.** A full-parallel CUDA compile can use several GiB per
> translation unit and will OOM a box that is also serving a model. Keep
> `MAX_JOBS` at 2–3 and watch `free -g`. The build is fully isolated in its own
> venv and produces a wheel — it never touches a running runtime env. Expect
> roughly 2 hours at `MAX_JOBS=2` on a GB10.

After the build, Gemma 4 is **native** (skip `patches/gemma4-backport.sh`). For
single-node serving you still want the CUTLASS FP8 + Ray OOM fixes; multi-node
also needs the NCCL sm_121 build:

```bash
./patches/cutlass-fp8-sm121.sh ~/vllm-mtp-env
./patches/ray-unified-memory.sh ~/vllm-mtp-env
```

### Launch with MTP

```bash
VLLM_VENV=~/vllm-mtp-env ./scripts/launch-gemma4-mtp.sh \
  ~/hf_models/gemma-4-26B-A4B-it \
  ~/hf_models/gemma-4-26B-A4B-it-assistant      # the MTP draft
```

The draft is the matching `*-assistant` checkpoint (`model_type:
gemma4_assistant`). The launcher also enables the **`gemma4` tool-call parser**
so tool calling and guided JSON work out of the box. Pass `none` as the draft
to serve without speculative decoding.

### Measured MTP result (gemma-4-26B-A4B-it, single-node GB10)

| Metric | Value |
|--------|------:|
| Draft acceptance rate | **79.5%** |
| Mean acceptance length | 1.79 (of max 2.0 at `num_speculative_tokens=1`) |
| Decode, MTP off (`--enforce-eager`) | ~12 tok/s |
| Decode, MTP on | **~19.5 tok/s (≈1.6x)** |
| Context / KV | 32K, 395K-token cache (12x concurrency) |

MTP only helps single-stream latency; it does not raise aggregate throughput at
high concurrency. The `--enforce-eager` flag (CUDA graphs off) is currently
required on GB10, which caps the no-MTP baseline.

---

## KV Cache Compression (ManthanQuant)

[`third_party/manthanquant/`](third_party/manthanquant) is the vendored
**ManthanQuant** package — 3-bit Lloyd-Max KV-cache compression built for GB10
unified memory. It is bundled directly in this repo so the whole stack (Gemma 4
+ MTP + KV compression) installs from one clone.

### Why this design on GB10

On the DGX Spark, the GPU and CPU share one 128 GB pool, and loading custom CUDA
extensions at import time conflicts with Triton on sm_121. ManthanQuant therefore
runs its quantizer in **pure numpy on the Grace CPU cores** — `.cpu().numpy()` is
near-free on unified memory because nothing physically moves. There is a faster
CUDA path (QJL + fused decode in `csrc/`), but it is **off by default and not used
on GB10**; it is meant for x86/datacenter GPUs.

### How it compresses

Per attention vector of dimension `D` (head_dim), bf16 is replaced by an L2 radius
plus a bit-packed 3-bit Lloyd-Max index per element:

| | bytes for D=256 |
|---|---|
| Original bf16 | `256 × 2 = 512` |
| Radius + 3-bit packed | `4 + ⌈256×3/8⌉ = 4 + 96 = 100` |
| **Ratio** | **5.12x** at **~0.978 cosine similarity** |

The 8 Lloyd-Max centroids are MSE-optimal for a unit Gaussian, which is what each
vector looks like after L2-normalization and `√D` scaling. The repo's proof suite
(`third_party/manthanquant/tests/test_compression_proof.py`) validates the ratio,
the quality bound, bit-packing round-trips, and edge cases — all 10 tests pass.

### Install

The from-source MTP build must already exist (it provides the venv to patch):

```bash
# one-time: install the package + patch the vLLM attention backends in the venv
./patches/manthanquant-kv-compression.sh ~/vllm-mtp-env
```

This editable-installs the **CPU-only** package (no nvcc) and patches the
`flash_attn` + `triton_attn` backends so KV is compressed after each layer.
Gemma 4 uses `triton_attn` on GB10 (vLLM hard-forces it for Gemma 4's
heterogeneous head dims). Revert any time:

```bash
./patches/manthanquant-kv-compression.sh ~/vllm-mtp-env --revert
```

### Activate at serve time

Compression is gated by an env flag so you can A/B it without re-patching. The
MTP launcher exposes it as `MANTHANQUANT=1`:

```bash
MANTHANQUANT=1 VLLM_VENV=~/vllm-mtp-env ./scripts/launch-gemma4-mtp.sh \
  ~/hf_models/gemma-4-26B-A4B-it \
  ~/hf_models/gemma-4-26B-A4B-it-assistant
```

### Verify it is actually running

ManthanQuant writes an **honest activation marker** only when the KV hook truly
fires (not merely when the patched file is imported):

```bash
cat ~/logs/manthanquant_active.flag   # one "kv_hook_first ..." line per worker pid
```

An empty/absent file means compression is **not** running — by design, the loaded
flag and the active flag are kept separate so "the module imported" is never
mistaken for "compression happened".

> **Scope note.** On GB10 the compressed cache is currently built and measured
> CPU-side; the fused compressed-decode path that reads it back is CUDA-only and
> disabled on sm_121, so attention still runs against the bf16 paged cache. The
> 5.12x is a faithful measurement of the compressed copy. Wiring the compressed
> decode into the GB10 path (for end-to-end VRAM savings) is the open follow-up.

---

## Full Installation Walkthrough

### Prerequisites

- NVIDIA DGX Spark (GB10, aarch64) with CUDA 13.0
- Python 3.12 with `python3.12-dev` installed
- vLLM 0.17.x or 0.18.x in a virtualenv (e.g., `~/vllm-env`)
- Internet access (for cloning vLLM source + pip installs)

### Step-by-Step

**1. Create vLLM virtualenv (if you don't have one)**

```bash
python3.12 -m venv ~/vllm-env
~/vllm-env/bin/pip install --upgrade pip
~/vllm-env/bin/pip install vllm==0.18.0
```

**2. Run the installer**

```bash
./install.sh ~/vllm-env
```

The installer runs these patches in order:
1. `patches/nccl-sm121-build.sh` -- Builds NCCL from source with sm_121 support
2. `patches/cutlass-fp8-sm121.sh` -- Disables CUTLASS FP8 (falls back to Triton)
3. `patches/ray-unified-memory.sh` -- Disables Ray OOM killer for unified memory
4. `patches/gemma4-backport.sh` -- Backports Gemma 4 from vLLM main

**3. Download a model**

```bash
# Gemma 4 31B (dense, slower on GB10 but high quality)
huggingface-cli download google/gemma-4-31B-it --local-dir ~/hf_models/gemma-4-31B-it

# Qwen3-Omni 30B (MoE, 7x faster on GB10 -- recommended)
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir ~/hf_models/Qwen3-Omni-30B-A3B-Instruct
```

**4. Launch**

```bash
source ~/vllm-env/bin/activate
./scripts/launch-gemma4.sh ~/hf_models/gemma-4-31B-it
```

---

## All GB10 Fixes

These are the 8 fixes required to run vLLM on the DGX Spark GB10. Each is included as a standalone patch script in `patches/`.

### Fix 1: NCCL -- Build from Source with sm_121

**Problem**: Pre-built NCCL (pip or deb packages) lacks GPU kernel support for Blackwell sm_121. Multi-node init fails with `Message truncated` errors.

**Fix**: Build NCCL v2.28.9 from source targeting sm_121:

```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl && git checkout v2.28.9-1
make -j$(nproc) src.build \
  NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" \
  CUDA_HOME=/usr/local/cuda
```

Set `LD_LIBRARY_PATH` to put the custom build first:
```bash
export LD_LIBRARY_PATH=$HOME/nccl/build/lib:$LD_LIBRARY_PATH
```

**Patch**: `patches/nccl-sm121-build.sh`

### Fix 2: CUTLASS FP8 -- Disable for sm_121

**Problem**: vLLM's prebuilt `_C.abi3.so` has CUTLASS FP8 kernels for sm_120/sm_120a but not sm_121. Calling `cutlass_scaled_mm` crashes with `RuntimeError: Error Internal`.

**Fix**: Force `cutlass_fp8_supported()` and `cutlass_block_fp8_supported()` to return `False`, along with their module-level constants. vLLM falls back to Triton-based FP8 kernels which work on sm_121.

**Critical**: Must patch both the functions AND the constants. Multiple callers invoke the functions directly.

**Patch**: `patches/cutlass-fp8-sm121.sh`

### Fix 3: Ray Unified Memory -- Disable OOM Killer

**Problem**: After loading model shards, Ray kills the worker because GPU memory (which is unified with system memory) exceeds its 0.95 threshold:
```
ray.exceptions.OutOfMemoryError: Memory on the node was 115.65GB / 121.69GB (0.950369)
```

**Fix**: Set `RAY_memory_usage_threshold=1.0` to disable the OOM killer. The GB10's unified memory architecture means model weights in GPU memory always count toward system memory usage.

**Patch**: `patches/ray-unified-memory.sh`

### Fix 4: QSFP Network -- MTU 9000 (Multi-Node)

For multi-node setups with 200 GbE QSFP direct cables, set MTU 9000 on both nodes for efficient tensor parallel communication.

### Fix 5: VLLM_HOST_IP -- Consistent IPs for Ray

Ray registers nodes with QSFP IPs but vLLM detects LAN IPs via socket. Set `VLLM_HOST_IP` on all nodes to force consistent IP reporting.

### Fix 6: NCCL Environment Variables

Force NCCL to use the QSFP interface for multi-node:
```bash
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

### Fix 7: Package Parity Across Nodes

All nodes in a tensor-parallel cluster must have identical Python package versions (triton, transformers, etc.). Install `python3.12-dev` on every node.

### Fix 8: Streaming Reasoning Tokens

vLLM with `--reasoning-parser` sends thinking tokens as `delta.reasoning` (not `<think>` tags). Backend streaming handlers must check for this field.

---

## Gemma 4 Patch Details

Gemma 4 support (PR #38826) is only in vLLM's main branch. The patch backports it to vLLM 0.18.x:

**What it does:**

1. **Upgrades transformers** -- Installs from GitHub main (gemma4 model_type not in any stable release)
2. **Copies model files** -- `gemma4.py`, `gemma4_mm.py`, `gemma4_utils.py` from vLLM main
3. **Copies RoPE implementation** -- `gemma4_rope.py` for Gemma 4's rotary embeddings
4. **Copies reasoning/tool parsers** -- `gemma4_reasoning_parser.py`, `gemma4_tool_parser.py`
5. **Patches model registry** -- Registers `Gemma4ForCausalLM` and `Gemma4ForConditionalGeneration`
6. **Patches base.py** -- Handles `null` sub_configs (Gemma 4 has `audio_config=null`)
7. **Patches utils.py** -- Loads named buffers (`layer_scalar`) that Gemma 4 requires

**Patch**: `patches/gemma4-backport.sh`

---

## Benchmark Results

Both models benchmarked on a single DGX Spark GB10 node (128 GB unified memory) using the included benchmark suite. 10 standardized tests covering reasoning, code generation, math, summarization, instruction following, creative writing, multi-turn conversation, JSON output, and long context.

### Head-to-Head Comparison

| Metric | Qwen3-Omni-30B | Gemma-4-31B-it | Winner |
|--------|---------------:|---------------:|--------|
| **Architecture** | MoE (30B total, 3B active) | Dense (31B) | -- |
| **Max Context** | 16,384 | 8,192 | Qwen |
| **TTFT (avg)** | 290.8 ms | 743.3 ms | Qwen (2.6x faster) |
| **TTFT (min)** | 126.3 ms | 655.2 ms | Qwen (5.2x faster) |
| **TTFT (p50)** | 127.2 ms | 786.3 ms | Qwen (6.2x faster) |
| **Avg TPS** | 28.2 tok/s | 3.8 tok/s | Qwen (7.4x faster) |
| **Median TPS** | 29.2 tok/s | 3.8 tok/s | Qwen (7.7x faster) |
| **Max TPS** | 29.8 tok/s | 3.9 tok/s | Qwen (7.6x faster) |
| **Pass Rate** | 10/10 (100%) | 9/10 (90%) | Qwen |

**Key finding**: Qwen3-Omni-30B is **7x faster** than Gemma 4 31B on GB10. This is because Qwen uses Mixture-of-Experts with only 3B parameters active per token, while Gemma 4 is a dense 31B model that activates all parameters. On unified memory hardware, the MoE advantage is massive.

---

## Per-Test Breakdown

| Test | Qwen3-Omni | | Gemma-4-31B | | |
|------|------:|------:|------:|------:|--------|
| | TPS | Result | TPS | Result | Winner |
| Multi-step Reasoning | 29.1 | PASS | 3.8 | PASS | Qwen |
| Python Code Generation | 29.8 | PASS | 3.9 | PASS | Qwen |
| Code Debugging | 29.8 | PASS | 3.9 | PASS | Qwen |
| Mathematical Reasoning | 29.4 | PASS | 3.9 | PASS | Qwen |
| Text Summarization | 26.8 | PASS | 3.8 | PASS | Qwen |
| Instruction Following | 23.9 | PASS | 3.6 | PASS | Qwen |
| Creative Writing | 29.5 | PASS | 3.5 | PASS | Qwen |
| Multi-turn Conversation | 28.9 | PASS | 3.8 | PASS | Qwen |
| Structured JSON Output | 25.4 | PASS | 3.9 | FAIL | Qwen |
| Long Context Understanding | 29.3 | PASS | 3.8 | PASS | Qwen |

**Gemma 4 JSON failure**: Gemma 4 produced repetitive/malformed JSON output, repeating the object multiple times instead of returning it once. All other tests passed.

### TTFT Per Test (ms)

| Test | Qwen3-Omni | Gemma-4-31B |
|------|------:|------:|
| Reasoning | 241.8 | 767.4 |
| Code Gen | 228.7 | 546.5 |
| Code Debug | 292.1 | 318.5 |
| Math | 183.7 | 297.4 |
| Summarization | 323.7 | 321.5 |
| Instruction Following | 178.9 | 516.9 |
| Creative Writing | 160.6 | 537.6 |
| Multi-turn | 186.5 | 539.9 |
| JSON Output | 216.4 | 516.5 |
| Long Context | 249.0 | 320.8 |

---

## Recommended Models for GB10

| Model | Type | Active Params | Speed | Quality | Context | Recommendation |
|-------|------|---------------|-------|---------|---------|----------------|
| **Gemma-4-26B-A4B-it** | MoE | 4B | ~19.5 tok/s (MTP) | Excellent | 32K | Best Gemma 4 on GB10. MoE + MTP + native tool calling. Needs the [from-source build](#from-source-build-vllm-0221--gemma-4-mtp); add [ManthanQuant](#kv-cache-compression-manthanquant) (`MANTHANQUANT=1`) to shrink the KV cache ~5x. |
| **Qwen3-Omni-30B-A3B** | MoE | 3B | 28 tok/s | 10/10 | 16K | Best overall for GB10. Fast, accurate, long context. |
| **Gemma-4-31B-it** | Dense | 31B | 3.8 tok/s | 9/10 | 8K | High quality but too slow for interactive use. Batch/offline only. |
| **Qwen3.5-122B-A10B-FP8** | MoE | 10B | ~15 tok/s* | Excellent | 16K | Best quality. Requires 2x GB10 nodes (tensor parallel). |
| **Qwen3.5-35B-A3B** | MoE | 3B | ~28 tok/s | Good | 16K | Fast, lower param count. Good for single-node. |

*Estimated. Multi-node speed depends on interconnect.

**Bottom line**: On GB10 unified memory, MoE models dominate. Dense models larger than ~8B are impractical for interactive use.

---

## Multi-Node Setup

For models that exceed single-node memory (e.g., Qwen3.5-122B-A10B-FP8), use Ray for tensor parallelism across two GB10 nodes connected via 200 GbE QSFP.

### Requirements

- 2x DGX Spark GB10 nodes
- Direct 200 GbE QSFP cable between them
- Identical vLLM environments on both nodes
- Custom NCCL built on both nodes

### Network Configuration

On each node, configure QSFP with MTU 9000:

```yaml
# /etc/netplan/01-qsfp.yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses:
        - 192.168.100.10/24  # Node 1: .10, Node 2: .11
      mtu: 9000
```

```bash
sudo netplan apply
```

### Launch Sequence

```bash
# Node 1 (head):
export VLLM_HOST_IP=192.168.100.10
export RAY_memory_usage_threshold=1.0
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export LD_LIBRARY_PATH=$HOME/nccl/build/lib:$LD_LIBRARY_PATH
ray start --head --port=6379

# Node 2 (worker):
export VLLM_HOST_IP=192.168.100.11
export RAY_memory_usage_threshold=1.0
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export LD_LIBRARY_PATH=$HOME/nccl/build/lib:$LD_LIBRARY_PATH
ray start --address=192.168.100.10:6379

# Node 1 (launch vLLM):
vllm serve /path/to/model \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.80 \
  --max-model-len 16384 \
  --trust-remote-code --enforce-eager \
  --enable-prefix-caching
```

### Multi-Node Config

See `configs/qwen-122b-multi-node.env` for a complete environment file.

---

## Architecture

```
vllm-gb10-gemma4/
|
|-- install.sh                    # One-command installer (file-copy backport)
|
|-- patches/
|   |-- nccl-sm121-build.sh       # Fix 1: Build NCCL for sm_121
|   |-- cutlass-fp8-sm121.sh      # Fix 2: Disable CUTLASS FP8
|   |-- ray-unified-memory.sh     # Fix 3: Ray OOM threshold
|   |-- gemma4-backport.sh        # Gemma 4 model support (0.18.x only)
|   |-- manthanquant-kv-compression.sh  # Install + patch ManthanQuant KV compression
|
|-- configs/
|   |-- gemma4-31b-single.env     # Single-node Gemma 4
|   |-- qwen-omni-30b-single.env  # Single-node Qwen Omni
|   |-- qwen-122b-multi-node.env  # Multi-node Qwen 122B
|
|-- scripts/
|   |-- build-from-source.sh      # Build the fork (native Gemma 4 + MTP)
|   |-- launch-gemma4.sh          # Launch Gemma 4 31B (dense)
|   |-- launch-gemma4-mtp.sh      # Launch Gemma 4 26B-A4B MoE + MTP (+ MANTHANQUANT=1)
|   |-- launch-qwen-omni.sh       # Launch Qwen3-Omni-30B
|   |-- run-benchmark.sh          # Run benchmark suite
|
|-- third_party/
|   |-- manthanquant/             # Vendored 3-bit KV-cache compression
|       |-- manthanquant/         #   cpu_quantize.py (numpy path), vllm_patch.py, ops.py
|       |-- csrc/                 #   CUDA QJL + fused-decode kernels (x86 only, off by default)
|       |-- install_vllm_patch.py #   Patches vLLM attention backends (honors VLLM_ENV)
|       |-- tests/                #   test_compression_proof.py (10 tests) + others
|
|-- benchmarks/
|   |-- model_benchmark.py        # Benchmark suite (10 tests)
|   |-- reports/
|       |-- benchmark_Qwen3-Omni-30B_*.json
|       |-- benchmark_gemma-4-31B-it_*.json
|
|-- README.md
|-- LICENSE                       # Apache 2.0


                     vLLM on DGX Spark GB10
                     ======================

  +------------------+     +------------------+
  |  DGX Spark #1    |     |  DGX Spark #2    |
  |  (Ray Head)      |     |  (Ray Worker)    |
  |                  |     |                  |
  |  ARM aarch64     |     |  ARM aarch64     |
  |  Blackwell GPU   |     |  Blackwell GPU   |
  |  128GB unified   |     |  128GB unified   |
  |                  |     |                  |
  |  vLLM serve      |     |  Ray worker      |
  |  Port 8000       |     |                  |
  +--------+---------+     +--------+---------+
           |                         |
           +-------- QSFP -----------+
                   200 GbE
                   MTU 9000

  Patches Applied:
  [NCCL sm_121] [CUTLASS FP8] [Ray OOM]
  [Gemma 4 backport] [VLLM_HOST_IP]
  [NCCL env vars] [Package parity]
```

---

## Troubleshooting

### `RuntimeError: Error Internal` on FP8 operations

CUTLASS FP8 kernels are not built for sm_121. Run `patches/cutlass-fp8-sm121.sh` to disable them.

### `ray.exceptions.OutOfMemoryError` after loading model

Ray's OOM killer triggers because unified memory counts GPU weights as system memory. Set `RAY_memory_usage_threshold=1.0`.

### `Message truncated: received 176 bytes instead of 172` (multi-node)

NCCL was not built with sm_121 support. Run `patches/nccl-sm121-build.sh` on all nodes.

### `ModuleNotFoundError: No module named 'triton'`

Missing on worker node. Install: `$VENV/bin/pip install triton`

### `Python.h not found` / Triton compile failures

Install dev headers: `sudo apt install -y python3.12-dev`

### Gemma 4: `KeyError: 'gemma4'` or model not recognized

The Gemma 4 patch was not applied. Run `patches/gemma4-backport.sh $VENV`.

### Gemma 4 MTP: `Transformers does not recognize this architecture` (gemma4_assistant)

The MTP draft (`*-assistant`, `model_type: gemma4_assistant`) and `gemma4_mtp`
are **fork-only**. A stock or backported vLLM cannot load them. Use the
[from-source build](#from-source-build-vllm-0221--gemma-4-mtp)
(`scripts/build-from-source.sh`), which bakes in `Gemma4MTPModel`.

### Tool calls leak into message content as `<|tool_call>...`

The wrong tool parser is active (e.g. the `hermes` fallback). Launch with
`--enable-auto-tool-choice --tool-call-parser gemma4`. `launch-gemma4-mtp.sh`
sets this automatically.

### ManthanQuant: `~/logs/manthanquant_active.flag` is empty / missing

The patched backend imported but the KV hook never fired. Common causes:
(1) you forgot `MANTHANQUANT=1` at launch (the launcher only sets
`MANTHANQUANT_ENABLED=1` when you pass it); (2) the backend the model actually
uses wasn't patched — Gemma 4 uses `triton_attn`, so re-run
`./patches/manthanquant-kv-compression.sh $VENV` and confirm it reports
`[triton_attn] OK`; (3) the patch went into a different venv — set
`VLLM_ENV=$VENV` (the patch script does this for you). The `loaded.flag`
accumulating is *not* proof — only `active.flag` is.

### ManthanQuant: `pip install` tries to compile `.cu` files and fails

The CUDA `_C` extension is opt-in. The default install is CPU-only and needs no
nvcc. If you hit nvcc errors you likely set `MANTHANQUANT_BUILD_CUDA=1` — unset
it on GB10. The active GB10 path is pure numpy and never loads `_C`.

### Gemma 4: `AttributeError: 'NoneType' object has no attribute 'dtype'`

The `base.py` null sub_config patch is missing. The Gemma 4 patch script handles this automatically.

### `nvidia-smi` shows `[N/A]` for memory

This is normal on GB10. The GPU uses unified memory shared with the CPU. Use `--gpu-memory-utilization` to control allocation.

### Model loads but inference is very slow (~3-4 tok/s for 30B+ dense)

This is expected for dense models on GB10 unified memory. Switch to an MoE model (e.g., Qwen3-Omni-30B) for 7x better throughput.

### vLLM reports wrong number of unique IPs (multi-node)

Set `VLLM_HOST_IP` on every node to the QSFP IP address. See [Multi-Node Setup](#multi-node-setup).

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
