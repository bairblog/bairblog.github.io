---
layout:             post
title:              "The Unsupervised Reinforcement Learning Benchmark"
date:               2021-12-14  12:00:00
author:             <a href="https://www.mishalaskin.com/edaaae9ed2b54016a66a0e315a9c9f63">Misha Laskin</a> and <a href="https://cs.nyu.edu/~dy1042/">Denis Yarats</a>
img:                /assets/unsupervised_rl/img0.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<!--
These are comments in HTML. The above header text is needed to format the
title, authors, etc. The "example_post" is an example representative image (not
GIF) that we use for each post for tweeting (see below as well) and for the
emails to subscribers. Please provide this image (and any other images and
GIFs) in the blog to the BAIR Blog editors directly.

The text directly below gets tweets to work. Please adjust according to your
post.

The `static/blog` directory is a location on the blog server which permanently
stores the images/GIFs in BAIR Blog posts. Each post has a subdirectory under
this for its images (titled `example_post` here, please change).

Keeping the post visbility as False will mean the post is only accessible if
you know the exact URL.

You can also turn on Disqus comments, but we recommend disabling this feature.
-->

<!-- twitter -->
<meta name="twitter:title" content="The Unsupervised Reinforcement Learning Benchmark">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/unsupervised_rl/img0.png">

<meta name="keywords" content="unsupervised learning, reinforcement learning, benchmark">
<meta name="description" content="Blog post about unsupervised reinforcement learning and benchmarking unsupervised RL algorithms">
<meta name="author" content="Misha Laskin, Denis Yarats">

<!--
The actual text for the post content appears below.  Text will appear on the
homepage, i.e., https://bair.berkeley.edu/blog/ but we only show part of the
posts on the homepage. The rest is accessed via clicking 'Continue'. This is
enforced with the `more` excerpt separator.
-->

<!-- # The Unsupervised Reinforcement Learning Benchmark
 -->
![img0.png](https://bair.berkeley.edu/static/blog/unsupervised-rl/img0.png)

# The shortcomings of supervised RL

Reinforcement Learning (RL) is a powerful paradigm for solving many problems of interest in AI, such as controlling autonomous vehicles, digital assistants, and resource allocation to name a few. We've seen over the last five years that, when provided with an extrinsic reward function, RL agents can master very complex tasks like playing Go, Starcraft, and dextrous robotic manipulation. While large-scale RL agents can achieve stunning results, ***even the best RL agents today are narrow.*** Most RL algorithms today can only solve the single task they were trained on and do not exhibit cross-task or cross-domain generalization capabilities.

A side-effect of the narrowness of today's RL systems is that ***today's RL agents are also very data inefficient***. If we were to train AlphaGo-like agents on many tasks each agent would likely require billions of training steps because today's RL agents don't have the capabilities to reuse prior knowledge to solve new tasks more efficiently. RL as we know it is supervised - agents overfit to a specific extrinsic reward which limits their ability to generalize.

<!--more-->

![img1.png](https://bair.berkeley.edu/static/blog/unsupervised-rl/img1.png)

![img2.png](https://bair.berkeley.edu/static/blog/unsupervised-rl/img2.png)

# Unsupervised RL as a path forward

To date, the most promising path toward generalist AI systems in language and vision has been through unsupervised pre-training. Masked casual and bi-directional transformers have emerged as scalable methods for pre-training language models that have shown unprecedented generalization capabilities. Siamese architectures and more recently masked auto-encoders have also become state-of-the-art methods for achieving fast downstream task adaptation in vision.

If we believe that pre-training is a powerful approach towards developing generalist AI agents, then it is natural to ask whether there exist self-supervised objectives that would allow us to pre-train RL agents. Unlike vision and language models which act on static data, RL algorithms actively influence their own data distribution. Like in vision and language, representation learning is an important aspect for RL as well but the unsupervised problem that is unique to RL is how agents can themselves generate interesting and diverse data trough self-supervised objectives. ***This is the unsupervised RL problem - how do we learn useful behaviors without supervision and then adapt them to solve downstream tasks quickly?***

# The unsupervised RL framework

Unsupervised RL is very similar to supervised RL. Both assume that the underlying environment is described by a Markov Decision Process (MDP) or a Partially Observed MDP, and both aim to maximize rewards. The main difference is that supervised RL assumes that supervision is provided by the environment through an extrinsic reward while unsupervised RL defines an intrinsic reward through a self-supervised task. Like supervision in NLP and vision, supervised rewards are either engineered or provided as labels by human operators which are hard to scale and limit the generalization of RL algorithms to specific tasks.

![img3.png](https://bair.berkeley.edu/static/blog/unsupervised-rl/img3.png)

At the Robot Learning Lab (RLL), we've been taking steps toward making unsupervised RL a plausible approach toward developing RL agents capable of generalization. To this end, we developed and released a benchmark for unsupervised RL with open-sourced PyTorch code for 8 leading or popular baselines.

## The Unsupervised Reinforcement Learning Benchmark (URLB)

While a variety of unsupervised RL algorithms have been proposed over the last few years, it has been impossible to compare them fairly due to differences in evaluation, environments, and optimization. For this reason, we built URLB which provides standardized evaluation procedures, domains, downstream tasks, and optimization for unsupervised RL algorithms

URLB splits training into two phases - a long unsupervised pre-training phase followed by a short supervised fine-tuning phase. The initial release includes three domains with four tasks each for a total of twelve downstream tasks for evaluation. 

![img4.png](https://bair.berkeley.edu/static/blog/unsupervised-rl/img4.png)

Most unsupervised RL algorithms known to date can be classified into three categories - knowledge-based, data-based, and competence-based. Knowledge-based methods maximize the prediction error or uncertainty of a predictive model (e.g. Curiosity, Disagreement, RND), data-based methods maximize the diversity of observed data (e.g. APT, ProtoRL), competence-based methods maximize the mutual information between states and some latent vector often referred to as the "skill" or "task" vector (e.g. DIAYN, SMM, APS). 

Previously these algorithms were implemented using different optimization algorithms (Rainbow DQN, DDPG, PPO, SAC, etc). As a result, unsupervised RL algorithms have been hard to compare. In our implementations we standardize the optimization algorithm such that the only difference between various baselines is the self-supervised objective. 

![img5.png](https://bair.berkeley.edu/static/blog/unsupervised-rl/img5.png)

We implemented and [released code for eight leading algorithms](https://github.com/rll-research/url_benchmark) supporting both state and pixel-based observations on domains based on the DeepMind Control Suite.

![img6.png](https://bair.berkeley.edu/static/blog/unsupervised-rl/img6.png)

By standardizing domains, evaluation, and optimization across all implemented baselines in URLB, ***the result is a first direct and fair comparison between these three different types of algorithms.*** 

![img7.png](https://bair.berkeley.edu/static/blog/unsupervised-rl/img7.png)

Above, we show aggregate statistics of fine-tuning runs across all 12 downstream tasks with 10 seeds each after pre-training on the target domain for 2M steps. We find that currently data-based methods (APT, ProtoRL) and RND are the leading approaches on URLB. 

We've also identified a number of promising directions for future research based on benchmarking existing methods. For example, competence-based exploration as a whole underperforms data and knowledge-based exploration. Understanding why this is the case is an interesting line for further research. For additional insights and directions for future research in unsupervised RL, we refer the reader to the [URLB paper](https://openreview.net/forum?id=lwrPkQP_is).

# Conclusion

Unsupervised RL is a promising path toward developing generalist RL agents. We've introduced a benchmark (URLB) for evaluating the performance of such agents. We've open-sourced code for both URLB and hope this enables other researchers to quickly prototype and evaluate unsupervised RL algorithms.

## Links

**Paper:** [URLB: Unsupervised Reinforcement Learning Benchmark](https://openreview.net/forum?id=lwrPkQP_is)
Michael Laskin\*, Denis Yarats\*, Hao Liu, Kimin Lee, Albert Zhan, Kevin Lu, Catherine Cang, Lerrel Pinto, Pieter Abbeel, NeurIPS, 2021, *these authors contributed equally* 

**Code:** [https://github.com/rll-research/url_benchmark](https://github.com/rll-research/url_benchmark)
