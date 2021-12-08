---
layout:             post
title:              "Recent Progress in Unsupervised Reinforcement Learning"
date:               2021-11-19  9:00:00
author:             <a href="https://www.mishalaskin.com/mlaskin/Misha-Laskin-edaaae9ed2b54016a66a0e315a9c9f63">Misha Laskin</a>
img:                assets/mi_sufficiency_analysis/image1.png
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
<meta name="twitter:title" content="Which Mutual Information Representation Learning Objectives are Sufficient for Control?">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/unsupervised-rl/img0.png">

<meta name="keywords" content="unsupervised reinforcement learning">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Misha Laskin">

![img0.png](https://bair.berkeley.edu/static/blog/unsupervised-rl//img0.png)

## The shortcomings of supervised RL

Reinforcement Learning (RL) is a powerful paradigm for solving many problems of interest in AI, such as controlling autonomous vehicles, digital assistants, and resource allocation to name a few. We've seen over the last five years that, when provided with an extrinsic reward function, RL agents can master very complex tasks like playing Go, Starcraft, and dextrous robotic manipulation. While large-scale RL agents can achieve stunning results, ***even the best RL agents today are narrow.*** Most RL algorithms today can only solve the single task they were trained on and do not exhibit cross-task or cross-domain generalization capabilities.

A side-effect of the narrowness of today's RL systems is that ***today's RL agents are also very data inefficient***. If we were to train AlphaGo-like agents on many tasks each agent would likely require billions of training steps because today's RL agents don't have the capabilities to reuse prior knowledge to solve new tasks more efficiently. RL as we know it is supervised - agents overfit to a specific extrinsic reward which limits their ability to generalize.

<!--more-->


![img1.png](https://bair.berkeley.edu/static/blog/unsupervised-rl/img1.png)

![img2.png](https://bair.berkeley.edu/static/blog/unsupervised-rl//img2.png)

## Unsupervised RL as a path forward

To date, the most promising path toward generalist AI systems in language and vision has been through unsupervised pre-training. Masked casual and bi-directional transformers have emerged as scalable methods for pre-training language models that have shown unprecedented generalization capabilities. Siamese architectures and more recently masked autoencoders have also become state-of-the-art methods for achieving fast downstream task adaptation.

If we believe that pre-training is a powerful approach towards developing generalist AI agents, then it is natural to ask whether there exist self-supervised objectives that would allow us to pre-train RL agents. Unlike vision and language models which act on static data, RL algorithms actively influence their own data distribution. Like in vision and language, representation learning is an important aspect for RL as well but the unsupervised problem that is unique to RL is how agents can themselves generate interesting and diverse data trough self-supervised objectives. ***This is the unsupervised RL problem - how do we learn useful behaviors without supervision and then adapt them to solve downstream tasks quickly?***

## The unsupervised RL framework

Unsupervised RL is very similar to supervised RL. Both assume that the underlying environment is described by a Markov Decision Process (MDP) or a Partially Observed MDP, and both aim to maximize rewards. The main difference is that supervised RL assumes that supervision is provided by the environment through an extrinsic reward while unsupervised RL defines an intrinsic reward through a self-supervised task. Like supervision in NLP and vision, supervised rewards are either engineered or provided as labels by human operators which are hard to scale and limit the generalization of RL algorithms to specific tasks.

![img3.png](https://bair.berkeley.edu/static/blog/unsupervised-rl//img3.png)

## Steps toward making unsupervised RL a reality

At the Robot Learning Lab (RLL), we've recently taken two steps toward making unsupervised RL a plausible approach toward developing RL agents capable of generalization. First, we developed and released a benchmark for unsupervised RL with open-sourced PyTorch code for 8 leading or popular baselines. Second, we used insights uncovered during the making of the benchmark to build a new state-of-the-art unsupervised RL algorithm called Contrastive Intrinsic Control (CIC).

### The Unsupervised Reinforcement Learning Benchmark (URLB)

While a variety of unsupervised RL algorithms have been proposed over the last few years, it has been impossible to compare them fairly due to differences in evaluation, environments, and optimization. For this reason, we built URLB which provides standardized evaluation procedures, domains, downstream tasks, and optimization for unsupervised RL algorithms

URLB splits training into two phases - a long unsupervised pre-training phase followed by a short supervised fine-tuning phase. The initial release includes three domains with four tasks each for a total of twelve downstream tasks for evaluation. 

![img4.png](https://bair.berkeley.edu/static/blog/unsupervised-rl//img4.png)

Most unsupervised RL algorithms known to date can be classified into three categories - knowledge-based, data-based, and competence-based. Knowledge-based methods maximize the prediction error or uncertainty of a predictive model (e.g. Curiosity, RND), data-based methods maximize the diversity of observed data (e.g. APT, count-based methods), competence-based methods maximize the mutual information between states and some latent vector often referred to as the "skill" or "task" vector (e.g. DIAYN, DADS). We implemented and released code for eight leading algorithms supporting both state and pixel-based observations on domains based on the DeepMind Control Suite. ***The result is a first direct and fair comparison between these three different types of algorithms.*** 

![img5.png](https://bair.berkeley.edu/static/blog/unsupervised-rl//img5.png)

We've also identified a number of promising directions for future research based on benchmarking existing methods. For more information, we refer the reader to the URLB paper.

### Unsupervised Skill Discovery with Contrastive Intrinsic Control

You may have noticed in the above figure that competence-based methods (in green) do substantially worse than the other two types of unsupervised RL algorithms. In this work, we asked why is this the case and what can we do to resolve it?

As a quick primer, competence-based algorithms maximize the mutual information between some observed variable such as states and a latent skill vector, which is usually sampled from noise.

![img6.png](https://bair.berkeley.edu/static/blog/unsupervised-rl//img6.png)

The mutual information is usually an intractable quantity and since we want to maximize it, we are usually better off maximizing a variational lower bound.

![img7.png](https://bair.berkeley.edu/static/blog/unsupervised-rl//img7.png)

The quantity `q(z|tau)` is referred to as the discriminator. In prior works, the discriminators are either classifiers over discrete skills or regressors over continuous skills. The problem is that classification and regression tasks need an exponential number of diverse data samples to be accurate. But in complex environments, there can be a very large number of skills and we therefore need discriminators capable of supporting large skill spaces. This tension between the need to support large skill spaces and the limitation of current discriminators leads us to propose Contrastive Intrinsic Control (CIC).

Contrastive Intrinsic Control (CIC) introduces a new contrastive density estimator to approximate the conditional entropy (the discriminator). Unlike visual contrastive learning, this contrastive objective operates over **state transitions** and **skill vectors**. This allows us to bring the powerful representation learning machinery from vision to unsupervised skill discovery.

![img8.png](https://bair.berkeley.edu/static/blog/unsupervised-rl//img8.png)

With the CIC objective we can define a new intrinsic reward for the exploration agent as well as an architecture for computing both terms in the mutual information decomposition.

![img9.png](https://bair.berkeley.edu/static/blog/unsupervised-rl//img9.png)

With explicit exploration through the state-transition entropy term and new contrastive discriminator, CIC adapts extremely efficiently to downstream tasks - outperforming prior competence-based approaches by **1.91x** and all prior exploration methods by **1.26x** on URLB.

![img10.png](https://bair.berkeley.edu/static/blog/unsupervised-rl//img10.png)

We provide more information in the CIC paper about how architectural details and skill dimension affect the performance of the CIC paper. The main takeaway from CIC is that there is nothing wrong with the competence-based objective of maximizing mutual information. However, what matters is how well we approximate this objective, especially in large skill spaces. 

## Conclusion

Unsupervised RL is a promising path toward developing generalist RL agents. We've introduced a benchmark (URLB) for evaluating the performance of such agents and a new algorithm (CIC) that resolves fundamental issues with prior competence-based unsupervised RL algorithms and, as a result, achieves leading results on URLB. We've open-sourced code for both URLB and CIC and hope this enables other researchers to quickly prototype and evaluate unsupervised RL algorithms.
