---
layout: post
title: "From CUDA to MLX: How K-Search Brings Decades of Kernel Expertise to Apple Silicon"
date: 2026-07-21 09:00:00
author: Shiyi Cao (UC Berkeley), Gal Bloch (IBM Research)
# PRODUCTION (uncomment after uploading to /static/blog/cuda-to-mlx-k-search/):
# img: https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/cover.png
# PREVIEW (github.io) - use until server static upload:
img: assets/cuda-to-mlx-k-search/cover.png
excerpt_separator: <!--more-->
visible: True
show_comments: False
---

<!-- twitter -->
<meta name="twitter:title" content="From CUDA to MLX: How K-Search Brings Decades of Kernel Expertise to Apple Silicon">
<meta name="twitter:card" content="summary_large_image">
<!-- PRODUCTION: <meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/cover.png"> -->
<meta name="twitter:image" content="https://bairblog.github.io/assets/cuda-to-mlx-k-search/cover.png">

<meta name="keywords" content="CUDA, MLX, Apple Silicon, kernel optimization, evolutionary search, K-Search, FlashAttention, Mamba, state space models, Metal">
<meta name="description" content="IBM Research extends K-Search, the evolutionary kernel search framework from Berkeley Sky Lab, with a CUDA-to-MLX translation layer that transfers expert kernel knowledge to Apple Silicon, reaching 97% of FlashAttention performance and a ~20x faster Mamba SSM prefill.">
<meta name="author" content="Shiyi Cao, Gal Bloch">

<p style="text-align:center;">
<!-- PRODUCTION: <img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/cover.png" alt="Kernel knowledge transfer from CUDA to MLX"><br> -->
<img src="/assets/cuda-to-mlx-k-search/cover.png" alt="Kernel knowledge transfer from CUDA to MLX"><br>
</p>

We face a new epoch in computing. Hardware is changing rapidly — not just faster GPUs, but a growing range of chips from different vendors, each with its own architecture and often tailored to specific AI workloads. Software is changing just as fast, and AI coding tools now generate in minutes what took months of effort a few years ago. In this reality, it has become almost impossible for humans to keep pace with the change.

<!--more-->

With so much of computing now centered on AI, GPU kernels are a crucial component of its success. These are the low-level programs that run inside the GPU, and writing efficient ones is far from obvious — it takes years of expertise to get right. Transferring a kernel from one vendor's hardware to another is harder still, and often means rediscovering the same optimizations from scratch. The CUDA ecosystem, for example, has accumulated decades of hard-won kernel expertise: hand-tuned implementations of attention, state space models, and other critical operations representing thousands of engineering hours. Newer hardware ecosystems (Apple Silicon, custom AI accelerators, and others) are growing fast but lack this depth.

In this work we ask whether that expertise can be transferred automatically. We built on K-Search, an evolutionary kernel search framework developed at Berkeley Sky Lab that uses AI to optimize GPU kernels, and extended it with a backend for MLX — Apple's machine-learning framework for its own Apple Silicon chips. We developed a novel structured CUDA-to-MLX translation layer that lets K-Search take existing CUDA kernels as a knowledge base and adapt them into high-quality GPU kernels for Apple Silicon, rather than rebuilding from scratch.

We show that our approach reaches near-expert level performance on Apple Silicon for the Attention kernel, and a large prefill speedup over the community mlx-lm implementation on the Mamba SSM kernel; we report the numbers, and how much of the gain comes from the translation layer, in the sections below. Although we focus on MLX kernels for Apple Silicon, the method is not specific to MLX and applies to any ecosystem where CUDA expertise is transferable.

## Why MLX?

Apple's MLX framework has seen remarkable adoption since late 2023. With Apple Silicon in hundreds of millions of MacBooks and Mac Studios, MLX enables local AI inference without cloud costs. The unified memory architecture makes it especially attractive for mid-sized models (7B–70B parameters on M series chips).

Yet beneath this momentum lies a significant gap: many performance-critical kernels that the NVIDIA ecosystem takes for granted: paged attention, optimized SSM scan kernels, fused MoE routing are either absent or naive without hardware-specific tuning. MLX runs models correctly but often leaves significant performance on the table.

This gap is what motivates the rest of this post.

## What is K-Search?

K-Search is an evolutionary kernel optimization framework originally developed by our first author Shiyi Cao at UC Berkeley Sky Lab. Given a naive kernel and a hardware specification, it runs an iterative optimization loop: an LLM reasons about which optimizations to try next, a code-writing model generates candidate kernels, and those candidates are compiled and benchmarked on real hardware.

Measurements feed back into the search, which keeps refining, pursuing promising directions and dropping dead ends until performance converges.

Search is grounded by a Spec: a domain-specific document encoding hardware rules, optimization patterns, and mathematical constraints which keeps generated code from hallucinating invalid primitives and ensures candidates will actually compile and run efficiently.

In our runs, a single model (Gemini 3.5 Pro Preview) plays both roles: it maintains the reasoning state and writes the kernels. The reasoning half is prompted as a "GPU kernel performance engineer" and asked to work through a fixed analysis before proposing anything: classify the kernel (reduction, scan, attention/softmax, …), rewrite the reference computation in canonical form, map out data layout and access patterns, and hypothesize the likely bottleneck (bandwidth, latency, compute, or synchronization) in each runtime regime. Only then does it emit candidate optimizations, each as a single change implementable in one iteration.

We call the persistent reasoning state a *world model*. Rather than a flat list of things to try, it is a decision (prefix) tree: each root→leaf path composes a full optimization plan, and sibling branches are competing alternatives. Every node is scored — an `overall_rating` in [0, 10], a `confidence` in [0, 1], and per-node `impacts` on memory bandwidth, register pressure, and compute/hardware fit — so the search can rank partial plans and expand the most promising ones. The tree persists and grows across rounds: refining an idea adds a child node rather than overwriting its parent, and if the best score fails to improve for a few rounds (a stagnation window) the search backs off to explore an alternative branch. A single node, as it appears mid-run on the attention kernel, looks like this:

```
{
  "action": "Replace the threadgroup-memory softmax reduction
             with a register-only reduction: each SIMD group
             owns 8 query rows and reduces across lanes with
             simd_shuffle_xor, removing a threadgroup_barrier.",
  "difficulty_1_to_5": 4,
  "impacts": {
    "memory_bandwidth":  8,
    "register_pressure": 4,   // risk: spill if Br > 8
    "compute_hw_fit":    9    // SIMD width 32; keep tile 8x8
  },
  "overall_rating_0_to_10": 8,
  "confidence_0_to_1": 0.7
}
```

<p style="text-align:center;">
<!-- PRODUCTION: <img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/figure-01-ksearch-loop.png" alt="Overview of the K-Search loop" width="700"><br> -->
<img src="/assets/cuda-to-mlx-k-search/figure-01-ksearch-loop.png" alt="Overview of the K-Search loop" width="700"><br>
<i>
Figure 1: Overview of the K-Search loop (Cao et al., 2026).
</i>
</p>

## Building an MLX backend

To bring K-Search to Apple Silicon, we first built a native MLX backend. We implemented a full MLX-specific task adapter for K-Search, including:

- An MLX task backend in `k_search/tasks/` handling kernel compilation and execution on Apple Silicon via MLX's Metal/C++ APIs.
- Updated kernel generator prompts for writing and modifying Metal/MLX kernels.
- MLX-specific benchmarking integration using `mlx.core` measurement utilities.

## Translating CUDA expertise to MLX

However, the more interesting challenge was not simply running K-Search on MLX. The key insight is that expert CUDA kernels encode decades of optimization knowledge that is transferable to Apple GPU if you can bridge the conceptual gap. Simply handing an LLM a CUDA kernel and asking it to port it is not enough: without deep hardware context, it produces code that is syntactically valid but architecturally wrong (wrong tile sizes, invalid primitives, mismatched memory assumptions).

Our translation layer consists of:

- **Concept mapping tables:** A structured glossary of CUDA primitives and their MLX/Metal equivalents with hard constraints. For example:
  - `__shared__` maps to Metal `threadgroup` memory but with a hard 32 KB limit (vs. NVIDIA's 48 KB)
  - `warp_reduce` maps to MMA (preferred)
  - `__syncthreads()` becomes `threadgroup_barrier(mem_flags::mem_tg)`
  - H100's ~3.35 TB/s HBM3 maps to M3 Max's ~400 GB/s unified DRAM a bandwidth difference that reshapes which optimizations are worth pursuing.
- **MLX-specific hints and patterns:** Concrete code-level patterns for operations with no direct CUDA equivalent, such as register-based row reductions using `simd_shuffle_xor` in an 8×8 MMA tile layout, or the "exp2 trick" (replacing $exp(x)$ with $exp_2(x \log_2 e)$) for faster softmax on Apple's fast $exp_2$ hardware instruction.
- **Reusable assertions:** Expert kernel behaviors reframed as properties the evolutionary search must preserve, rather than code to copy.

## Does the translation layer matter?

To isolate the impact of the translation layer, we evaluate the same evolutionary optimization framework under four different configurations on an MLX attention kernel for Apple Silicon. The translation layer provides the optimizer with architecture-specific implementation knowledge extracted from high-performance kernels (e.g., FlashAttention-2), allowing the evolutionary search to reason about implementation strategies rather than starting from a naive kernel. We compare this against pure evolution, Claude Code operating under the same optimization budget, and a naive baseline.

<p style="text-align:center;">
<!-- PRODUCTION: <img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/figure-02-attention-optimizations.png" alt="Performance scaling of the Attention Kernel through stacked optimizations" width="700"><br> -->
<img src="/assets/cuda-to-mlx-k-search/figure-02-attention-optimizations.png" alt="Performance scaling of the Attention Kernel through stacked optimizations" width="700"><br>
<i>
Figure 2: Performance scaling of the Attention Kernel through stacked optimizations. The "Full Context" configuration successfully discovers and implements advanced strategies like double buffering and loop unrolling, achieving near-expert performance.
</i>
</p>

| Configuration | Description | Performance |
|---|---|---|
| Naive Baseline | No evolution | 0.02× |
| Naive + Evolve | Evolution, no translation context | 0.26× |
| Claude Code | Claude Code operating under the same run budget | 0.46× |
| Full Context | Evolution with translation context | 0.97× |

The jump from 0.26× to 0.97× illustrates how much the translation layer matters. With full context, the evolved kernel independently discovers the key optimizations in FlashAttention 2: threadgroup memory tiling, online softmax, K-transposition for memory access, and the exp2 trick. The last of these replaces every softmax exponential with a base-2 exponential,

$$e^x = 2^{x \log_2 e},$$

which is exact and lets the kernel use Apple's fast `fast::exp2()` hardware instruction directly instead of paying for a base conversion at runtime.

## A 20× faster prefill: the Mamba SSM kernel

To evaluate whether K-Search generalizes beyond attention kernels, we applied it to the state-space model (SSM) kernel used by Mamba. Unlike attention, the computational bottleneck is a recurrent state update rather than a softmax, providing a substantially different optimization challenge. We compare the evolved implementation against the community MLX implementation (mlx-lm) and the PyTorch reference implementation (mamba.py) on an M1 Max.

Evaluated on mamba-370m f16, M1 Max 64GB:

| Metric | mlx-mamba (ours) | mlx-lm (community) | mamba.py |
|---|---|---|---|
| Decode | 152 tok/s | 116 tok/s | 40 tok/s |
| Prefill L=512 | 5,751 tok/s | 329 tok/s | 1,089 tok/s |
| Prefill L=1024 | 6,010 tok/s | 327 tok/s | 1,127 tok/s |
| Prefill L=2048 | 6,612 tok/s | 326 tok/s | 1,092 tok/s |
| Prefill L=4096 | 6,743 tok/s | 339 tok/s | 1,042 tok/s |

The ~20× prefill speedup over mlx-lm comes down to one difference: mlx-lm does not implement a parallel scan for the SSM. The state recurrence

$$h_t = \bar{a}_t h_{t-1} + \bar{b}_t$$

looks inherently sequential, but each step can be written as a pair $(\bar{a}_t, \bar{b}_t)$ under the associative combine

$$(a_2, b_2) \circ (a_1, b_1) = \left(a_2 a_1,\ a_2 b_1 + b_2\right),$$

which reproduces the recurrence exactly. Because the operator is associative, the whole sequence can be evaluated with a parallel (prefix) scan in $O(\log N)$ dependent steps instead of $O(N)$. mlx-lm skips this and processes tokens one at a time, leaving most of Apple Silicon's compute idle; our evolved Metal kernel applies the scan and makes much fuller use of GPU throughput. The gain shows up in prefill, where the full sequence is available to scan in parallel, and not in single-token decode, where there is only one new token per step and no scan to parallelize — which is why the decode row is roughly flat while prefill is ~20×.

mamba.py is slow on both prefill and decode because it is a PyTorch reference implementation that falls back to CPU or MPS on Apple Silicon, forgoing the hardware-specific optimizations that MLX's Metal backend makes possible.

## What's next?

On the two kernels we studied, AI-driven evolutionary kernel search grounded in structured cross-platform translation knowledge reached near-expert performance on Apple Silicon without a team of GPU experts starting from scratch. We do not yet know how far this generalizes, but the result is encouraging.

For us the main takeaway is that the bottleneck was not the LLM's ability to write Metal code, but the quality of the context and constraints we gave it. Our CUDA translation layer converts existing NVIDIA kernel expertise into actionable guidance for Apple Silicon, and lets K-Search's evolutionary search do the rest.

We are actively extending this work in several directions: supporting new architectures, with current efforts focused on developing new kernels for the IBM Spyre AIU and broader hardware targets; adding more kernels such as paged attention and fused MoE routing; and improving integration with the K-Search evolution loop to make translation context even more automatic. If you are working on kernel optimization for non-CUDA hardware, we would love to hear from you.

## Try it yourself

The MLX backend is built on top of the open-source K-Search repo, so the results here can be reproduced directly. The steps are:

1\. Clone and install

```bash
git clone https://github.com/caoshiyi/K-Search.git
cd K-Search

uv pip install openai wandb
uv pip install git+https://github.com/caoshiyi/flashinfer-bench-ksearch.git
```

2\. Set your credentials

Open the relevant script under scripts/ and set three variables at the top:

```bash
KSEARCH_ROOT=/path/to/K-Search
API_KEY=your-llm-api-key
```

3\. Run kernel search

```bash
# Optimize Flash Attention on Apple Silicon (world-model mode)
bash scripts/mac_flash_attention_wm.sh

# Or a Mamba SSM kernel, e.g. the selective scan
bash scripts/mamba_selective_scan_fwd_wm.sh
```

Full CLI reference and documentation are in the README.

## Acknowledgements

This work was carried out by IBM Research and builds on K-Search from the UC Berkeley Sky Lab. We welcome collaboration and feedback from the MLX and broader AI systems communities.

We thank Assaf Toledo (IBM Research), Michael Factor (IBM Research), Gil Vernik (IBM Research), and Joseph E. Gonzalez (UC Berkeley) for their feedback and support throughout this work.
