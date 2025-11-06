---
layout:             post
title:              "RL without TD learning"
date:               2025-11-01 09:00:00
author:             <a href="https://seohong.me/">Seohong Park</a>
img:                /assets/rl-without-td-learning/teaser.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      True
---

<!-- twitter -->
<meta name="twitter:title" content="RL without TD learning">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/rl-without-td-learning/teaser.png">

<meta name="keywords" content="">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Seohong Park">

In this post, I'll introduce a reinforcement learning (RL) algorithm based on an "alternative" paradigm: **divide and conquer**. Unlike traditional methods, this algorithm is _not_ based on temporal difference (TD) learning (which has [scalability challenges](<https://seohong.me/blog/q-learning-is-not-yet-scalable/>)), and scales well to long-horizon tasks.

<p style="text-align:center;">
<img src="{{ site.baseurl }}/assets/rl-without-td-learning/teaser_short.png" alt="" width="100%">
<br>
<i style="font-size: 0.9em;">We can do Reinforcement Learning (RL) based on divide and conquer, instead of temporal difference (TD) learning.</i>
</p>

<!--more-->

## Problem setting: off-policy RL

Our problem setting is **off-policy RL**. Let's briefly review what this means.

There are two classes of algorithms in RL: on-policy RL and off-policy RL. On-policy RL means we can _only_ use fresh data collected by the current policy. In other words, we have to throw away old data each time we update the policy. Algorithms like PPO and GRPO (and policy gradient methods in general) belong to this category.

Off-policy RL means we don't have this restriction: we can use _any_ kind of data, including old experience, human demonstrations, Internet data, and so on. So off-policy RL is more general and flexible than on-policy RL (and of course harder!). Q-learning is the most well-known off-policy RL algorithm. In domains where data collection is expensive (_e.g._, **robotics**, dialogue systems, healthcare, etc.), we often have no choice but to use off-policy RL. That's why it's such an important problem.

As of 2025, I think we have reasonably good recipes for scaling up on-policy RL (_e.g._, PPO, GRPO, and their variants). However, we still haven't found a "scalable" _off-policy RL_ algorithm that scales well to complex, long-horizon tasks. Let me briefly explain why.

## Two paradigms in value learning: Temporal Difference (TD) and Monte Carlo (MC)

In off-policy RL, we typically train a value function using temporal difference (TD) learning (_i.e._, Q-learning), with the following Bellman update rule:

$$\begin{aligned} Q(s, a) \gets r + \gamma \max_{a'} Q(s', a'), \end{aligned}$$

The problem is this: the error in the next value $Q(s', a')$ propagates to the current value $Q(s, a)$ through bootstrapping, and these errors _accumulate_ over the entire horizon. This is basically what makes TD learning struggle to scale to long-horizon tasks (see [this post](<https://seohong.me/blog/q-learning-is-not-yet-scalable/>) if you're interested in more details).

To mitigate this problem, people have mixed TD learning with Monte Carlo (MC) returns. For example, we can do $n$-step TD learning (TD-$n$):

$$\begin{aligned} Q(s_t, a_t) \gets \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n \max_{a'} Q(s_{t+n}, a'). \end{aligned}$$

Here, we use the actual Monte Carlo return (from the dataset) for the first $n$ steps, and then use the bootstrapped value for the rest of the horizon. This way, we can reduce the number of Bellman recursions by $n$ times, so errors accumulate less. In the extreme case of $n = \infty$, we recover pure Monte Carlo value learning.

While this is a reasonable solution (and often [works well](<https://arxiv.org/abs/2506.04168>)), it is highly unsatisfactory. First, it doesn't _fundamentally_ solve the error accumulation problem; it only reduces the number of Bellman recursions by a constant factor ($n$). Second, as $n$ grows, we suffer from high variance and suboptimality. So we can't just set $n$ to a large value, and need to carefully tune it for each task.

Is there a fundamentally different way to solve this problem?

## The "Third" Paradigm: Divide and Conquer

My claim is that a _third_ paradigm in value learning, **divide and conquer**, may provide an ideal solution to off-policy RL that scales to arbitrarily long-horizon tasks.

<p style="text-align:center;">
<img src="{{ site.baseurl }}/assets/rl-without-td-learning/teaser.png" alt="" width="100%">
<br>
<i style="font-size: 0.9em;">Divide and conquer reduces the number of Bellman recursions logarithmically.</i>
</p>

The key idea of divide and conquer is to divide a trajectory into two equal-length segments, and combine their values to update the value of the full trajectory. This way, we can (in theory) reduce the number of Bellman recursions _logarithmically_ (not linearly!). Moreover, it doesn't require choosing a hyperparameter like $n$, and it doesn't necessarily suffer from high variance or suboptimality, unlike $n$-step TD learning.

Conceptually, divide and conquer really has all the nice properties we want in value learning. So I've long been excited about this high-level idea. The problem was that it wasn't clear how to actually do this in practice... until recently.

## A practical algorithm

In a [recent work](<https://arxiv.org/abs/2510.22512>) co-led with [Aditya](<https://aober.ai/>), we made meaningful progress toward realizing and scaling up this idea. Specifically, we were able to scale up divide-and-conquer value learning to highly complex tasks (as far as I know, this is the first such work!) at least in one important class of RL problems, _goal-conditioned RL_. Goal-conditioned RL aims to learn a policy that can reach any state from any other state. This provides a natural divide-and-conquer structure. Let me explain this.

The structure is as follows. Let's first assume that the dynamics is deterministic, and denote the shortest path distance ("temporal distance") between two states $s$ and $g$ as $d^*(s, g)$. Then, it satisfies the triangle inequality:

$$\begin{aligned} d^*(s, g) \leq d^*(s, w) + d^*(w, g) \end{aligned}$$

for all $s, g, w \in \mathcal{S}$.

In terms of values, we can equivalently translate this triangle inequality to the following _"transitive"_ Bellman update rule:

$$\begin{aligned} 
V(s, g) \gets \begin{cases}
\gamma^0 & \text{if } s = g, \\\\ 
\gamma^1 & \text{if } (s, g) \in \mathcal{E}, \\\\ 
\max_{w \in \mathcal{S}} V(s, w)V(w, g) & \text{otherwise}
\end{cases} 
\end{aligned}$$

where $\mathcal{E}$ is the set of edges in the environment's transition graph, and $V$ is the value function associated with the sparse reward $r(s, g) = 1(s = g)$. **Intuitively**, this means that we can update the value of $V(s, g)$ using two "smaller" values: $V(s, w)$ and $V(w, g)$, provided that $w$ is the optimal "midpoint" (subgoal) on the shortest path. This is exactly the divide-and-conquer value update rule that we were looking for!

### The problem

However, there's one problem here. The issue is that it's unclear how to choose the optimal subgoal $w$ in practice. In tabular settings, we can simply enumerate all states to find the optimal $w$ (this is essentially the Floyd-Warshall shortest path algorithm). But in continuous environments with large state spaces, we can't do this. Basically, this is why previous works have struggled to scale up divide-and-conquer value learning, even though this idea has been around for decades (in fact, it dates back to the very first work in goal-conditioned RL by [Kaelbling (1993)](<https://scholar.google.com/citations?view_op=view_citation&citation_for_view=IcasIiwAAAAJ:hC7cP41nSMkC>) -- see [our paper](<https://arxiv.org/abs/2510.22512>) for a further discussion of related works). The main contribution of our work is a practical solution to this issue.

### The solution

Here's our key idea: we _restrict_ the search space of $w$ to the states that appear in the dataset, specifically, those that lie between $s$ and $g$ in the dataset trajectory. Also, instead of searching for the optimal $\text{argmax}_w$, we compute a "soft" $\text{argmax}$ using [expectile regression](<https://arxiv.org/abs/2110.06169>). Namely, we minimize the following loss:

$$\begin{aligned} \mathbb{E}\left[\ell^2_\kappa (V(s_i, s_j) - \bar{V}(s_i, s_k) \bar{V}(s_k, s_j))\right], \end{aligned}$$

where $\bar{V}$ is the target value network, $\ell^2_\kappa$ is the expectile loss with an expectile $\kappa$, and the expectation is taken over all $(s_i, s_k, s_j)$ tuples with $i \leq k \leq j$ in a randomly sampled dataset trajectory.

This has two benefits. First, we don't need to search over the entire state space. Second, we prevent value overestimation from the $\max$ operator by instead using the "softer" expectile regression. We call this algorithm **Transitive RL (TRL)**. Check out [our paper](<https://arxiv.org/abs/2510.22512>) for more details and further discussions!

## Does it work well?

<div style="display: flex; justify-content: center; gap: 30px; margin: 30px 0;">
  <div style="text-align: center;">
    <video autoplay loop muted playsinline style="width: 350px;">
      <source src="{{ site.baseurl }}/assets/rl-without-td-learning/humanoidmaze.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <br>
    <i style="font-size: 0.9em;">humanoidmaze</i>
  </div>
  <div style="text-align: center;">
    <video autoplay loop muted playsinline style="width: 350px;">
      <source src="{{ site.baseurl }}/assets/rl-without-td-learning/puzzle.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <br>
    <i style="font-size: 0.9em;">puzzle</i>
  </div>
</div>

To see whether our method scales well to complex tasks, we directly evaluated TRL on some of the most challenging tasks in [OGBench](<https://seohong.me/projects/ogbench/>), a benchmark for offline goal-conditioned RL. We mainly used the hardest versions of humanoidmaze and puzzle tasks with large, 1B-sized datasets. These tasks are highly challenging: they require performing combinatorially complex skills across up to **3,000 environment steps**.

<p style="text-align:center;">
<img src="{{ site.baseurl }}/assets/rl-without-td-learning/table.png" alt="" width="100%">
<br>
<i style="font-size: 0.9em;">TRL achieves the best performance on highly challenging, long-horizon tasks.</i>
</p>

The results are quite exciting! Compared to many strong baselines across different categories (TD, MC, quasimetric learning, etc.), TRL achieves the best performance on most tasks.

<p style="text-align:center;">
<img src="{{ site.baseurl }}/assets/rl-without-td-learning/1b.svg" alt="" width="100%">
<br>
<i style="font-size: 0.9em;">TRL matches the best, individually tuned TD-$n$, <b>without needing to set $\boldsymbol{n}$</b>.</i>
</p>

This is my favorite plot. We compared TRL with $n$-step TD learning with different values of $n$, from $1$ (pure TD) to $\infty$ (pure MC). The result is really nice. TRL matches the best TD-$n$ on all tasks, **without needing to set $\boldsymbol{n}$**! This is exactly what we wanted from the divide-and-conquer paradigm. By recursively splitting a trajectory into smaller ones, it can _naturally_ handle long horizons, without having to arbitrarily choose the length of trajectory chunks.

The paper has a lot of additional experiments, analyses, and ablations. If you're interested, check out [our paper](<https://arxiv.org/abs/2510.22512>)!

## What's next?

In this post, I shared some promising results from our new divide-and-conquer value learning algorithm, Transitive RL. This is just the beginning of the journey. There are many open questions and exciting directions to explore:

* Perhaps the most important question is how to extend TRL to regular, reward-based RL tasks beyond goal-conditioned RL. Would regular RL have a similar divide-and-conquer structure that we can exploit? I'm quite optimistic about this, given that it is possible to convert any reward-based RL task to a goal-conditioned one at least in theory (see page 40 of [this book](<https://sites.google.com/view/goalconditioned-rl/>)).

* Another important challenge is to deal with stochastic environments. The current version of TRL assumes deterministic dynamics, but many real-world environments are stochastic, mainly due to partial observability. For this, ["stochastic" triangle inequalities](<https://arxiv.org/abs/2406.17098>) might provide some hints.

* Practically, I think there is still a lot of room to further improve TRL. For example, we can find better ways to choose subgoal candidates (beyond the ones from the same trajectory), further reduce hyperparameters, further stabilize training, and simplify the algorithm even more.

In general, I'm really excited about the potential of the divide-and-conquer paradigm. I [still](<https://seohong.me/blog/q-learning-is-not-yet-scalable/>) think one of the most important problems in RL (and even in machine learning) is to find a _scalable_ off-policy RL algorithm. I don't know what the final solution will look like, but I do think divide and conquer, or **recursive** decision-making in general, is one of the strongest candidates toward this holy grail (by the way, I think the other strong contenders are (1) model-based RL and (2) TD learning with some "magic" tricks). Indeed, several recent works in other fields have shown the promise of recursion and divide-and-conquer strategies, such as [shortcut models](<https://kvfrans.com/shortcut-models/>), [log-linear attention](<https://arxiv.org/abs/2506.04761>), and [recursive language models](<https://alexzhang13.github.io/blog/2025/rlm/>) (and of course, classic algorithms like quicksort, segment trees, FFT, and so on). I hope to see more exciting progress in scalable off-policy RL in the near future!

### Acknowledgments

I'd like to thank [Kevin](<https://kvfrans.com/>) and [Sergey](<https://people.eecs.berkeley.edu/~svlevine/>) for their helpful feedback on this post.

---

*This post originally appeared on [Seohong Park's blog](https://seohong.me/blog/rl-without-td-learning/).*
