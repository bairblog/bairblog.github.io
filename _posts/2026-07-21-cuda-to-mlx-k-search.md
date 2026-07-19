---
layout: post
title: "From CUDA to MLX: How K-Search Brings Decades of Kernel Expertise to Apple Silicon"
date: 2026-07-21 09:00:00
author: Shiyi Cao (UC Berkeley), Gal Bloch (IBM Research)
img: https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/cover.png
excerpt_separator: <!--more-->
visible: True
show_comments: False
---

<!-- twitter -->
<meta name="twitter:title" content="From CUDA to MLX: How K-Search Brings Decades of Kernel Expertise to Apple Silicon">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/cover.png">

<meta name="keywords" content="CUDA, MLX, Apple Silicon, kernel optimization, evolutionary search, K-Search, FlashAttention, Mamba, state space models, Metal">
<meta name="description" content="IBM Research extends K-Search, the evolutionary kernel search framework from Berkeley Sky Lab, with a CUDA-to-MLX translation layer that transfers expert kernel knowledge to Apple Silicon, reaching 97% of FlashAttention performance and a ~20x faster Mamba SSM prefill.">
<meta name="author" content="Shiyi Cao, Gal Bloch">

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/cover.png" alt="Kernel knowledge transfer from CUDA to MLX"><br>
</p>

The CUDA ecosystem has accumulated decades of hard-won kernel expertise: hand-tuned implementations of attention, state space models, and other critical operations representing thousands of engineering hours. Newer hardware ecosystems (Apple Silicon, custom AI accelerators, and others) are growing fast but lack this depth. Porting those optimizations by hand is slow, expensive, and requires hardware-specific expertise on M-series chips that is in short supply.

<!--more-->

IBM Research has extended K-Search, the evolutionary kernel search framework developed at Berkeley Sky Lab, with an MLX backend and a structured CUDA-to-MLX translation layer that enables the framework to systematically translate expert CUDA kernel knowledge into optimized implementations for Apple Silicon.

Rather than starting from scratch on each new platform, we treat existing CUDA kernels as a knowledge base and use structured translation layers to guide the search toward expert-quality implementations on the target hardware.

Applied to Apple Silicon (MLX), we reached **97% of FlashAttention performance** on the Attention kernel (vs. 26% without our translation patterns), and a **~20× prefill speedup** over the community mlx-lm implementation on the Mamba SSM kernel. The method is general and applies to any ecosystem where CUDA expertise is transferable.

## Background

The AI industry has spent years accumulating an enormous amount of optimization knowledge inside CUDA kernels. Every generation of NVIDIA hardware has inspired new tiling strategies, memory layouts, and algorithmic tricks that are now embedded in highly optimized implementations of attention, state space models, mixture-of-experts routing, and other core operations. As new hardware ecosystems emerge — from Apple Silicon to custom AI accelerators — much of that expertise remains trapped in CUDA code, forcing engineers to rediscover the same optimizations from scratch.

In this work, we show that this knowledge can be transferred automatically. By extending K-Search with a CUDA→MLX translation layer, we automatically generate high-performance kernels for Apple Silicon that approach expert implementations.

### Why MLX?

Apple's MLX framework has seen remarkable adoption since late 2023. With Apple Silicon in hundreds of millions of MacBooks and Mac Studios, MLX enables local AI inference without cloud costs. The unified memory architecture makes it especially attractive for mid-sized models (7B–70B parameters on M series chips).

Yet beneath this momentum lies a significant gap: many performance-critical kernels that the NVIDIA ecosystem takes for granted — paged attention, optimized SSM scan kernels, fused MoE routing — are either absent, naive, or community-written without hardware-specific tuning. MLX runs models correctly but often leaves significant performance on the table.

This is the gap we set out to close automatically.

### What is K-Search?

K-Search is an evolutionary kernel optimization framework originally developed by Shiyi Cao at UC Berkeley Sky Lab. Given a naive kernel and a hardware specification, it runs an iterative optimization loop: an LLM reasons about which optimizations to try next, a code-writing model generates candidate kernels, and those candidates are compiled and benchmarked on real hardware.

Measurements feed back into the search, which keeps refining, pursuing promising directions and dropping dead ends until performance converges.

Search is grounded by a Spec: a domain-specific document encoding hardware rules, optimization patterns, and mathematical constraints which keeps generated code from hallucinating invalid primitives and ensures candidates will actually compile and run efficiently.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/figure-01-ksearch-loop.png" alt="Overview of the K-Search loop" width="700"><br>
<i>
Figure 1: Overview of the K-Search loop (Cao et al., 2026). The framework operates on a Search State $S_t$ structured as a search tree. The tree consists of Closed nodes (blue, visited states with attached program like $x_{12}$) and a Frontier of Open nodes (orange, pending hypotheses like $u_{13}$). The workflow iterates through three phases: (1) Action Selection, where the most promising action node is retrieved from the frontier based on world model estimated priority score $V$; (2) Local Refinement, where a stochastic policy $\pi_{code}$ samples concrete implementations until stagnation; and (3) World Model Update, where the LLM reasons over the trajectory to update the search tree via Insert (adding new actions), Update (adjusting $V$, e.g., $u_{11}$ dropping from 0.9 to 0.6), and Prune (removing less promising nodes like $u_{10}$).
</i>
</p>

### Building an MLX Backend

To bring K-Search to Apple Silicon, we first built a native MLX backend. We first implemented a full MLX-specific task adapter for K-Search, including:

- An MLX task backend in `k_search/tasks/` handling kernel compilation and execution on Apple Silicon via MLX's Metal/C++ APIs.
- Updated kernel generator prompts for writing and modifying Metal/MLX kernels.
- MLX-specific benchmarking integration using `mlx.core` measurement utilities.

## Translating CUDA Expertise to MLX

However, the more interesting challenge was not simply running K-Search on MLX. The key insight is that expert CUDA kernels encode decades of optimization knowledge that is transferable to Apple GPU if you can bridge the conceptual gap. Simply handing an LLM a CUDA kernel and asking it to port it is not enough: without deep hardware context, it produces code that is syntactically valid but architecturally wrong (wrong tile sizes, invalid primitives, mismatched memory assumptions).

Our translation layer consists of:

- **Concept mapping tables:** A structured glossary of CUDA primitives and their MLX/Metal equivalents with hard constraints. For example:
  - `__shared__` maps to Metal `threadgroup` memory but with a hard 32 KB limit (vs. NVIDIA's 48 KB)
  - `warp_reduce` maps to MMA (preferred)
  - `__syncthreads()` becomes `threadgroup_barrier(mem_flags::mem_tg)`
  - H100's ~3.35 TB/s HBM3 maps to M3 Max's ~400 GB/s unified DRAM — a bandwidth difference that reshapes which optimizations are worth pursuing.
- **MLX-specific hints and patterns:** Concrete code-level patterns for operations with no direct CUDA equivalent, such as register-based row reductions using `simd_shuffle_xor` in an 8×8 MMA tile layout, or the "exp2 trick" (replacing $\exp(x)$ with $\exp_2(x \cdot \log_2 e)$) for faster softmax on Apple's fast exp2 hardware instruction.
- **Reusable assertions:** Expert kernel behaviors reframed as properties the evolutionary search must preserve, rather than code to copy.

## Optimizing the Attention Kernel

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/figure-02-attention-optimizations.png" alt="Performance scaling of the Attention Kernel through stacked optimizations" width="700"><br>
<i>
Figure 2: Performance scaling of the Attention Kernel through stacked optimizations. The "Full Context" configuration successfully discovers and implements advanced strategies like double buffering and loop unrolling, achieving near-expert performance.
</i>
</p>

| Configuration | Description | Performance |
|---|---|---|
| Naive Baseline | No evolution | 0.02× |
| Naive + Evolve | Evolution, no translation context | 0.26× |
| Claude Code | Claude Code operating under the same run budget | 0.46× |
| Full Context | Full translation context from the start | **0.97×** |

The jump from 0.26× to 0.97× illustrates how much the translation layer matters. With full context, the evolved kernel independently discovers the key optimizations in FlashAttention 2: threadgroup memory tiling, online softmax, K-transposition for memory access, and the exp2 trick.

## Mamba SSM Kernel: A 20× Faster Prefill

Evaluated on mamba-370m f16, M1 Max 64GB:

| Metric | mlx-mamba (ours) | mlx-lm (community) | mamba.py |
|---|---|---|---|
| Decode | 152 tok/s | 116 tok/s | 40 tok/s |
| Prefill L=512 | 5,751 tok/s | 329 tok/s | 1,089 tok/s |
| Prefill L=1024 | 6,010 tok/s | 327 tok/s | 1,127 tok/s |
| Prefill L=2048 | 6,612 tok/s | 326 tok/s | 1,092 tok/s |
| Prefill L=4096 | 6,743 tok/s | 339 tok/s | 1,042 tok/s |

The ~20× prefill speedup over mlx-lm comes down to one key difference: **mlx-lm does not implement a parallel scan** for the SSM. Without it, prefill processes tokens sequentially, leaving the vast majority of Apple Silicon's compute idle. Our evolved Metal kernel applies a parallel (prefix) scan, restructuring computation to O(log N) parallel steps and fully utilizing GPU throughput.

mamba.py is slow on both prefill and decode because it is a PyTorch reference implementation that falls back to CPU or MPS on Apple Silicon, forgoing the hardware-specific optimizations that MLX's Metal backend makes possible.

## Conclusion

This work demonstrates that AI-driven evolutionary kernel search, grounded in structured cross-platform translation knowledge, can reliably reach near-expert performance on new hardware without a team of GPU experts starting from scratch for each architecture.

The key finding: the bottleneck is not the LLM's ability to write Metal code, but the quality of the context and constraints you give it. Our CUDA translation layer converts decades of NVIDIA kernel expertise into actionable guidance for Apple Silicon, and lets K-Search's evolutionary search do the rest.

As hardware diversity rapidly expands across Apple Silicon, custom AI accelerators, and edge devices, the ability to automatically transfer kernel optimization knowledge between architectures becomes increasingly critical. This work represents an early but promising step toward that future, and there is much more to come.

### What's Next

We are actively extending this work in several directions: more kernels (paged attention, fused MoE routing), broader hardware targets beyond Apple Silicon, and improved integration with the K-Search evolution loop to make translation context even more automatic. If you are working on kernel optimization for non-CUDA hardware, we would love to hear from you.

### Try It Yourself

Our MLX backend is built on top of the open source K-Search repo. Here is how to get started:

**1. Clone and install**

```bash
git clone https://github.com/caoshiyi/K-Search.git
cd K-Search
uv pip install openai wandb
uv pip install git+https://github.com/caoshiyi/flashinfer-bench-ksearch.git
```

**2. Set your credentials**

Open the relevant script under `scripts/` and set three variables at the top:

```bash
KSEARCH_ROOT=/path/to/K-Search
API_KEY=your-llm-api-key
```

**3. Run kernel search**

```bash
bash scripts/mlx_mamba_wm.sh
```

Full CLI reference and documentation are in the README.

## Acknowledgements

This work was carried out by IBM Research and builds on K-Search from the UC Berkeley Sky Lab. We welcome collaboration and feedback from the MLX and broader AI systems communities.

We thank Assaf Toledo (IBM Research), Michael Factor (IBM Research), Gil Vernik (IBM Research), and Joseph E. Gonzalez (UC Berkeley) for their feedback and support throughout this work.
