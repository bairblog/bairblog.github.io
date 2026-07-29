---
layout: post
title: "From CUDA to MLX: How K-Search Brings Decades of Kernel Expertise to Apple Silicon"
date: 2026-07-29 09:00:00
author: "<a href=\"https://shiyicao.com/\">Shiyi Cao</a><sup class=\"author-affiliation-mark\">1</sup>, <a href=\"https://github.com/Gal-bloch\">Gal Bloch</a><sup class=\"author-affiliation-mark\">2</sup>, <a href=\"https://www.linkedin.com/in/assaf-toledo-a3608b27\">Assaf Toledo</a><sup class=\"author-affiliation-mark\">2</sup>, <a href=\"https://research.ibm.com/people/michael-factor\">Michael Factor</a><sup class=\"author-affiliation-mark\">2</sup>, <a href=\"https://www.linkedin.com/in/gil-vernik-1a50a316\">Gil Vernik</a><sup class=\"author-affiliation-mark\">2</sup>, and <a href=\"https://people.eecs.berkeley.edu/~jegonzal/\">Joseph E. Gonzalez</a><sup class=\"author-affiliation-mark\">1</sup>"
affiliations: "<span class=\"author-affiliation-mark\">1</span> UC Berkeley &nbsp;&nbsp;&nbsp; <span class=\"author-affiliation-mark\">2</span> IBM Research"
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
<meta name="author" content="Shiyi Cao, Gal Bloch, Assaf Toledo, Michael Factor, Gil Vernik, Joseph E. Gonzalez">

<p class="center">
<img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/cover.svg" onerror="this.onerror=null;this.src='{{ site.baseurl }}/assets/cuda-to-mlx-k-search/cover.svg';" alt="Kernel knowledge transfer from CUDA to MLX"><br>
</p>

<p class="cuda-mlx-fig-caption"><strong>Figure 1: CUDA-to-MLX optimization translation map.</strong> CUDA optimization knowledge can be translated into architecture-native MLX strategies rather than copied instruction-for-instruction.</p>

We face a new epoch in computing. Hardware is changing rapidly — not just faster GPUs, but a growing range of chips from different vendors, each with its own architecture and often tailored to specific AI workloads. Software is changing just as fast, and AI coding tools now generate in minutes what took months of effort a few years ago.

<!--more-->

With so much of computing now centered on AI, GPU kernels are a crucial component of its success. These are the low-level programs that run inside the GPU, and writing efficient ones is far from obvious — it takes years of expertise to get right. Transferring a kernel from one vendor's hardware to another is harder still, and often means rediscovering the same optimizations from scratch. The CUDA ecosystem, for example, has accumulated decades of hard-won kernel expertise: hand-tuned implementations of attention, state space models, and other critical operations representing thousands of engineering hours. Newer hardware ecosystems (Apple Silicon, custom AI accelerators, and others) are growing fast but lack this depth.

In this work we ask whether that expertise can be transferred automatically. We built on <a href="https://arxiv.org/abs/2602.19128">K-Search</a>, an evolutionary kernel search framework introduced by Cao et al. at Berkeley Sky Lab that uses AI to optimize GPU kernels, and extended it with a backend for MLX — Apple's machine-learning framework for its own Apple Silicon chips. We developed a novel structured CUDA-to-MLX translation layer that lets K-Search take existing CUDA kernels as a knowledge base and adapt them into high-quality GPU kernels for Apple Silicon, rather than rebuilding from scratch.

We show that our approach reaches near-expert level performance on Apple Silicon with 0.97x speedup compared to the native MLX Attention kernel, and up to a 20x prefill speedup over the community mlx-lm implementation on the Mamba SSM kernel; we report the numbers, and how much of the gain comes from the translation layer, in the sections below. Although we focus on MLX kernels for Apple Silicon, the method is not specific to MLX and applies to any ecosystem where CUDA expertise is transferable.

## Why MLX?

Apple's MLX framework has seen remarkable adoption since late 2023. With Apple Silicon in hundreds of millions of MacBooks and Mac Studios, MLX enables local AI inference without cloud costs. The unified memory architecture makes it especially attractive for mid-sized models (7B–70B parameters on M series chips).

Yet beneath this momentum lies a significant gap: many performance-critical kernels that the NVIDIA ecosystem takes for granted: paged attention, optimized SSM scan kernels, fused MoE routing are either absent or naive without hardware-specific tuning. MLX runs models correctly but often leaves significant performance on the table.

This gap is what motivates the rest of this post.

## What is K-Search?

K-Search is an evolutionary kernel optimization framework originally developed by our first author Shiyi Cao at UC Berkeley Sky Lab. Given a naive kernel and a hardware specification, it runs an iterative optimization loop: an LLM reasons about which optimizations to try next, a code-writing model generates candidate kernels, and those candidates are compiled and benchmarked on real hardware.

Measurements feed back into the search, which keeps refining, pursuing promising directions and dropping dead ends until performance converges.

<p class="center">
<img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/algorithm-01-k-search.svg" onerror="this.onerror=null;this.src='{{ site.baseurl }}/assets/cuda-to-mlx-k-search/algorithm-01-k-search.svg';" alt="Pseudocode for K-Search via co-evolving world models" width="460"><br>
</p>

<p class="cuda-mlx-fig-caption"><strong>Algorithm 1: K-Search via co-evolving world models.</strong> The search alternates between selecting the most promising action, instantiating and evaluating code until improvement stagnates, and evolving the world model through insert, update, and prune operations. Adapted from <a href="https://arxiv.org/abs/2602.19128">Cao et al. (2026)</a>.</p>

Search is grounded by a Spec: a domain-specific document encoding hardware rules, optimization patterns, and mathematical constraints which keeps generated code from hallucinating invalid primitives and ensures candidates will actually compile and run efficiently.

In our runs, a single model (Gemini 3.5 Pro Preview) plays both roles: it maintains the reasoning state and writes the kernels. The reasoning half is prompted as a "GPU kernel performance engineer" and asked to work through a fixed analysis before proposing anything: classify the kernel (reduction, scan, attention/softmax, …), rewrite the reference computation in canonical form, map out data layout and access patterns, and hypothesize the likely bottleneck (bandwidth, latency, compute, or synchronization) in each runtime regime. Only then does it emit candidate optimizations, each as a single change implementable in one iteration.

We call the persistent reasoning state a *world model*. Rather than a flat list of things to try, it is a decision (prefix) tree: each root→leaf path composes a full optimization plan, and sibling branches are competing alternatives. Every node is scored — an `overall_rating` in [0, 10], a `confidence` in [0, 1], and per-node `impacts` on memory bandwidth, register pressure, and compute/hardware fit — so the search can rank partial plans and expand the most promising ones. The tree persists and grows across rounds: refining an idea adds a child node rather than overwriting its parent, and if the best score fails to improve for a few rounds (a stagnation window) the search backs off to explore an alternative branch. A single node, as it appears mid-run on the attention kernel, looks like this:

<div class="cuda-mlx-world-model-code" markdown="1">

```json
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

</div>

<p class="cuda-mlx-fig-caption"><strong>Listing 1: Example K-Search world-model node.</strong> Each candidate optimization records a concrete action, estimated hardware impacts, an overall priority rating, and the model's confidence.</p>

<p class="center">
<img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/best-figure-01-ksearch-loop.svg" onerror="this.onerror=null;this.src='{{ site.baseurl }}/assets/cuda-to-mlx-k-search/best-figure-01-ksearch-loop.svg';" alt="Overview of the K-Search loop" width="700"><br>
</p>

<p class="cuda-mlx-fig-caption"><strong>Figure 2: Overview of K-Search.</strong> The framework operates on a <strong>Search State</strong> $S_t$ structured as a search tree. The tree consists of Closed nodes (blue, visited states with attached program like $x_{12}$) and a Frontier of Open nodes (orange, pending hypotheses like $u_{13}$). The workflow iterates through three phases: (1) <strong>Action Selection</strong>, where the most promising action node is retrieved from the frontier based on world model estimated priority score $V$; (2) <strong>Local Refinement</strong>, where a stochastic policy $\pi_{\mathrm{code}}$ samples concrete implementations until stagnation; and (3) <strong>World Model Update</strong>, where the LLM reasons over the trajectory to update the search tree via <em>Insert</em> (adding new actions), <em>Update</em> (adjusting $V$, e.g., $u_{11}$ dropping from 0.9 to 0.6), and <em>Prune</em> (removing less promising nodes like $u_{10}$).</p>

The original K-Search paper evaluated this search strategy on CUDA kernels from FlashInfer. Across GQA decode, MLA decode, MLA prefill, and MoE, K-Search improved more consistently than OpenEvolve and ShinkaEvolve over the same 120-iteration budget. These results establish the search framework we build on here; the remainder of this post asks whether its optimization knowledge can transfer beyond CUDA.

<p class="center">
<img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/best-figure-03-ksearch-main-results.svg" onerror="this.onerror=null;this.src='{{ site.baseurl }}/assets/cuda-to-mlx-k-search/best-figure-03-ksearch-main-results.svg';" alt="K-Search benchmark results compared with OpenEvolve and ShinkaEvolve" width="900"><br>
</p>

<p class="cuda-mlx-fig-caption"><strong>Figure 3: Main results from the original K-Search paper.</strong> Across three runs, K-Search achieves stronger best-so-far search scores, per-workload kernel performance, and speedup distributions than OpenEvolve and ShinkaEvolve on four FlashInfer CUDA kernels. Reproduced exactly from <a href="https://arxiv.org/abs/2602.19128">Cao et al. (2026)</a>.</p>

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

## Matching expert kernel performance: the Attention kernel

We evaluate three configurations of an MLX attention kernel for Apple Silicon: (1) a naive baseline, (2) pure evolution with no additional provided context, and (3) a full context translation layer, which supplies the optimizer with architecture-specific implementation knowledge extracted from high-performance kernels (e.g., FlashAttention-2), letting the evolutionary search reason about implementation strategies rather than starting from a naive kernel. Together, these three configurations let us isolate the exact impact of the translation layer.

<p class="center">
<img src="https://bair.berkeley.edu/static/blog/cuda-to-mlx-k-search/best-figure-02-attention-optimizations.svg" onerror="this.onerror=null;this.src='{{ site.baseurl }}/assets/cuda-to-mlx-k-search/best-figure-02-attention-optimizations.svg';" alt="Performance scaling of the Attention Kernel through stacked optimizations" width="700"><br>
</p>

<p class="cuda-mlx-fig-caption"><strong>Figure 4:</strong> Performance scaling of the Attention Kernel through stacked optimizations. The "Full Context" configuration successfully discovers and implements advanced strategies like double buffering and loop unrolling, achieving near-expert performance.</p>

The jump from 0.26× to 0.97× the speed of Apple's state-of-the-art attention kernel — illustrates how much the translation layer matters. With full context, the evolved kernel independently discovers the key optimizations in FlashAttention 2: threadgroup memory tiling, online softmax, K-transposition for memory access, and the exp2 trick. The last of these replaces every softmax exponential with a base-2 exponential,

$$e^x = 2^{x \log_2 e},$$

which is exact and lets the kernel use Apple's fast `fast::exp2()` hardware instruction directly instead of paying for a base conversion at runtime.

## A 20× faster prefill: the Mamba SSM kernel

To evaluate whether K-Search generalizes beyond attention kernels, we applied it to the state-space model (SSM) kernel used by Mamba. Unlike attention, the computational bottleneck is a recurrent state update rather than a softmax, providing a substantially different optimization challenge. We compare the evolved implementation against the community MLX implementation (mlx-lm) and the PyTorch reference implementation (mamba.py) on an M1 Max.

Evaluated on mamba-370m f16, M1 Max 64GB:

<div class="cuda-mlx-results-table" markdown="1">

| Metric | mlx-mamba (ours) | mlx-lm (community) | mamba.py |
| --- | --- | --- | --- |
| Decode | 152 tok/s | 116 tok/s | 40 tok/s |
| Prefill L=512 | 5,751 tok/s | 329 tok/s | 1,089 tok/s |
| Prefill L=1024 | 6,010 tok/s | 327 tok/s | 1,127 tok/s |
| Prefill L=2048 | 6,612 tok/s | 326 tok/s | 1,092 tok/s |
| Prefill L=4096 | 6,743 tok/s | 339 tok/s | 1,042 tok/s |

</div>

<p class="cuda-mlx-fig-caption"><strong>Table 1:</strong> Prefill and decode throughput on mamba-370m (f16, M1 Max 64GB). mlx-mamba (ours) reaches ~20× higher prefill throughput than the community mlx-lm baseline, while decode remains comparable.</p>

The ~20× prefill speedup over mlx-lm comes down to one difference: mlx-lm does not implement a parallel scan for the SSM. The state recurrence

$$h_t = \bar{a}_t h_{t-1} + \bar{b}_t$$

looks inherently sequential, but each step can be written as a pair $(\bar{a}_t, \bar{b}_t)$ under the associative combine

$$(a_2, b_2) \circ (a_1, b_1) = \left(a_2 a_1,\ a_2 b_1 + b_2\right),$$

which reproduces the recurrence exactly. Because the operator is associative, the whole sequence can be evaluated with a parallel (prefix) scan in $O(\log N)$ dependent steps instead of $O(N)$. mlx-lm skips this and processes tokens one at a time, leaving most of Apple Silicon's compute idle; our evolved Metal kernel applies the scan and makes much fuller use of GPU throughput. The gain shows up in prefill, where the full sequence is available to scan in parallel, and not in single-token decode, where there is only one new token per step and no scan to parallelize — which is why the decode row is roughly flat while prefill is ~20×.

mamba.py is slow on both prefill and decode because it is a PyTorch reference implementation that falls back to CPU or MPS on Apple Silicon, forgoing the hardware-specific optimizations that MLX's Metal backend makes possible.

## What's next?

On the two kernels we studied, AI-driven evolutionary kernel search grounded in structured cross-platform translation knowledge reached near-expert performance on Apple Silicon without a team of GPU experts starting from scratch. We do not yet know how far this generalizes, but the result is encouraging.

For us the main takeaway is that the bottleneck was not the LLM's ability to write Metal code, but the quality of the context and constraints we gave it. Our CUDA translation layer converts existing NVIDIA kernel expertise into actionable guidance for Apple Silicon, and lets K-Search's evolutionary search do the rest.

We are actively extending this work in several directions: supporting new architectures, with current efforts focused on developing new kernels for the IBM Spyre AIU and broader hardware targets; adding more kernels such as paged attention and fused MoE routing; and improving integration with the K-Search evolution loop to make translation context even more automatic.

## Acknowledgements

This work was carried out by IBM Research and builds on K-Search from the UC Berkeley Sky Lab (<a href="https://arxiv.org/abs/2602.19128">Cao et al., 2026</a>). We welcome collaboration and feedback from the MLX and broader AI systems communities. If you are working on kernel optimization for non-CUDA hardware, we would love to hear from you.

---

## Citation

```bibtex
@article{cao2026k,
  title={K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model},
  author={Cao, Shiyi and Mao, Ziming and Gonzalez, Joseph E and Stoica, Ion},
  journal={arXiv preprint arXiv:2602.19128},
  year={2026}
}
```

<div class="cuda-mlx-appendix" markdown="1">

## Appendix: Try it yourself

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

</div>
