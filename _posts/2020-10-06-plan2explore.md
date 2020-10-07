---
layout:             post
title:              "Plan2Explore: Active Model-Building for Self-Supervised Visual Reinforcement Learning"
date:               2020-10-06 9:00:00
author:             <a href="https://www.seas.upenn.edu/~oleh/">Oleh Rybkin</a> and <a href="https://danijar.com/">Danijar Hafner</a> and <a href="https://www.cs.cmu.edu/~dpathak/">Deepak Pathak</a>
img:                assets/plan2explore/plan2explore.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      True
---

<meta name="twitter:title" content="Plan2Explore: Active Model-Building for Self-Supervised Visual Reinforcement Learning">
<meta name="twitter:card" content="summary_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/plan2explore/figure1_teaser.gif">

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure1_teaser.gif" height="" width="90%">
<br />
</p>

*This post is cross-listed [on the CMU ML blog][1]*.

To operate successfully in unstructured open-world environments, autonomous intelligent agents need to solve many different tasks and learn new tasks quickly. Reinforcement learning has enabled artificial agents to solve complex tasks both in <a href="https://deepmind.com/research/case-studies/alphago-the-story-so-far">simulation</a> and <a href="https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html"> real-world</a>. However, it requires collecting large amounts of experience in the environment, and the agent learns only that particular task, much like a student memorizing a lecture without understanding. Self-supervised reinforcement learning has emerged <a href="https://pathak22.github.io/noreward-rl/">as</a> <a href="https://arxiv.org/abs/1903.03698">an</a> <a href="https://arxiv.org/abs/1907.01657">alternative</a>, where the agent only follows an intrinsic objective that is independent of any individual task, analogously to <a href=" https://www.youtube.com/watch?v=SaJL4SLfrcY&ab_channel=InriaChannel">unsupervised representation learning</a>. After experimenting with the environment without supervision, the agent builds an understanding of the environment, which enables it to adapt to specific downstream tasks more efficiently.

In this post, we explain our recent publication that develops <a href="https://ramanans1.github.io/plan2explore/">Plan2Explore</a>. While many recent papers on self-supervised reinforcement learning have focused on <a href="https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html">model-free</a>  agents that can only capture knowledge by remembering behaviors practiced during self-supervision, our agent learns an internal <a href="https://bair.berkeley.edu/blog/2019/12/12/mbpo/">world model</a> that lets it extrapolate beyond memorized facts by predicting what will happen as a consequence of different potential actions. The world model captures general knowledge, allowing Plan2Explore to quickly solve new tasks through planning in its own imagination. In contrast to the model-free prior work, the world model further enables the agent to explore what it expects to be novel, rather than repeating what it found novel in the past. Plan2Explore obtains state-of-the-art zero-shot and few-shot performance on continuous control benchmarks with high-dimensional input images. To make it easy to experiment with our agent, we are open-sourcing the complete <a href="https://github.com/ramanans1/plan2explore">source code</a>.

<!--more-->

# How does Plan2Explore work?

At a high level, Plan2Explore works by training a world model, exploring to
maximize the information gain for the world model, and using the world model at
test time to solve new tasks (see figure above). Thanks to effective
exploration, the learned world model is general and captures information that
can be used to solve multiple new tasks with no or few additional environment
interactions. We discuss each part of the Plan2Explore algorithm individually
below. We assume a basic understanding of <a href="https://en.wikipedia.org/wiki/Reinforcement_learning">reinforcement
learning</a> in this post.

# Learning the world model

Plan2Explore learns a world model that predicts future outcomes given past
observations $o_{1:t}$ and actions $a_{1:t}$. To handle high-dimensional image
observations, we encode them into lower-dimensional features $h$ and use an <a href="https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html">RSSM</a>
model that predicts forward in a compact latent state-space $s$. The latent
state aggregates information from past observations and is trained for future
prediction, using a variational objective that reconstructs future
observations. Since the latent state learns to represent the observations,
during planning we can predict entirely in the latent state without decoding
the images themselves. The figure below shows our latent prediction
architecture.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure2_model.gif" height="" width="90%">
<br />
</p>

# A novelty metric for active model-building

To learn an accurate and general world model we need an exploration strategy
that collects new and informative data. To achieve this, Plan2Explore uses a
novelty metric derived from the model itself. The novelty metric measures the
expected information gained about the environment upon observing the new data.
As the figure below shows, this is approximated by the disagreement <a href="https://arxiv.org/abs/1612.01474">of</a>
<a href="https://pathak22.github.io/exploration-by-disagreement/">an</a>  <a href="https://arxiv.org/abs/2002.08791">ensemble</a> of $K$ latent models.
Intuitively, large latent disagreement reflects high model uncertainty, and
obtaining the data point would reduce this uncertainty. By maximizing latent
disagreement, Plan2Explore selects actions that lead to the largest information
gain, therefore improving the model as quickly as possible.


<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure3_disagreement.gif" height="" width="65%">
<br />
</p>

# Planning for future novelty

To effectively maximize novelty, we need to know which parts of the environment
are still unexplored. Most prior work on self-supervised exploration used
model-free methods that reinforce past behavior that resulted in novel
experience. This makes these methods slow to explore: since they can only
repeat exploration behavior that was successful in the past, they are unlikely
to stumble onto something novel. In contrast, Plan2Explore plans for expected
novelty by measuring model uncertainty of imagined future outcomes. By seeking
trajectories that have the highest uncertainty, Plan2Explore explores exactly
the parts of the environments that were previously unknown.

To choose actions $a$ that optimize the exploration objective, Plan2Explore
leverages the learned world model as shown in the figure below. The actions are
selected to maximize the expected novelty of the entire future sequence
$s_{t:T}$, using imaginary rollouts of the world model to estimate the novelty.
To solve this optimization problem, we use the <a href="https://ai.googleblog.com/2020/03/introducing-dreamer-scalable.html">Dreamer</a>
agent, which learns a policy $\pi_\phi$ using a value function and analytic
gradients through the model. The policy is learned completely inside the
imagination of the world model. During exploration, this imagination training
ensures that our exploration policy is always up-to-date with the current world
model and collects data that are still novel. The figure below shows the
imagination training process.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure4_policy.gif" height="" width="90%">
<br />
</p>

# Evaluation of curiosity-driven exploration behavior

We evaluate Plan2Explore on the <a href="https://github.com/deepmind/dm_control">DeepMind Control Suite</a>, which
features 20 tasks requiring different control skills, such as locomotion,
balancing, and simple object manipulation. The agent only has access to image
observations and no proprioceptive information. Instead of random exploration,
which fails to take the agent far from the initial position, Plan2Explore leads
to diverse movement strategies like jumping, running, and flipping, as shown in
the figure below. Later, we will see that these are effective practice episodes
that enable the agent to quickly learn to solve various continuous control
tasks.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure5_gif1.gif" height="190" width="">
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure5_gif2.gif" height="190" width="">
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure5_gif3.gif" height="190" width=""><br>
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure5_gif4.gif" height="190" width="">
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure5_gif5.gif" height="190" width="">
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure5_gif6.gif" height="190" width="">
<br />
</p>


# Evaluation of downstream task performance

Once an accurate and general world model is learned, we test Plan2Explore on
previously unseen tasks. Given a task specified with a reward function, we use
the model to optimize a policy for that task. Similar to our exploration
procedure, we optimize a new value function and a new policy head for the
downstream task. This optimization uses only predictions imagined by the model,
enabling Plan2Explore to solve new downstream tasks in a zero-shot manner
without any additional interaction with the world.

The following plot shows the performance of Plan2Explore on tasks from DM
Control Suite. Before 1 million environment steps, the agent doesn’t know the
task and simply explores. The agent solves the task as soon as it is provided
at 1 million steps, and keeps improving fast in a few-shot regime after that.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/plan2explore/figure6_plot.png" height="" width="">
<br />
</p>

Plan2Explore (<font color="green"><strong>—</strong></font>) is able to solve most of the tasks we benchmarked. Since prior
work on self-supervised reinforcement learning used model-free agents that are
not able to adapt in a zero-shot manner (<a href="https://pathak22.github.io/noreward-rl/">ICM</a>, <font color="blue"><strong>—</strong></font>), or did not use
image observations, we compare by adapting this prior work to our model-based
Plan2Explore setup. Our latent disagreement objective outperforms other
previously proposed objectives. More interestingly, the final performance of
Plan2Explore is comparable to the state-of-the-art  <a href="https://ai.googleblog.com/2020/03/introducing-dreamer-scalable.html">oracle</a>
agent that requires task rewards throughout training (<font color="yellow"><strong>—</strong></font>). In our <a href="https://arxiv.org/abs/2005.05960">paper</a>, we further report
performance of Plan2Explore in the zero-shot setting where the agent needs to
solve the task before any task-oriented practice.

# Future directions

Plan2Explore demonstrates that effective behavior can be learned through
self-supervised exploration only. This opens multiple avenues for future
research:

- First, to apply self-supervised RL to a variety of settings, future work will
  investigate different ways of specifying the task and deriving behavior from
  the world model. For example, the task could be specified with a
  demonstration, description of the desired goal state, or communicated to the
  agent in natural language.

- Second, while Plan2Explore is completely self-supervised, in many cases a
  weak supervision signal is available, such as in hard exploration games,
  human-in-the-loop learning, or real life. In such a semi-supervised setting,
  it is interesting to investigate how weak supervision can be used to steer
  exploration towards the relevant parts of the environment.

- Finally, Plan2Explore has the potential to improve the data efficiency of
  real-world robotic systems, where exploration is costly and time-consuming,
  and the final task is often unknown in advance.

By designing a scalable way of planning to explore in unstructured environments
with visual observations, Plan2Explore provides an important step toward
self-supervised intelligent machines.

<hr>

We would like to thank Georgios Georgakis and the editors of CMU and BAIR blogs for the useful feedback.

This post is based on the following paper:

- Planning to Explore via Self-Supervised World Models<br>
  Ramanan Sekar\*, Oleh Rybkin\*, Kostas Daniilidis, Pieter Abbeel, Danijar Hafner, Deepak Pathak<br>
  Thirty-seventh International Conference Machine Learning (ICML), 2020.<br>
  <a href="https://arxiv.org/abs/2005.05960">arXiv</a>, <a href="https://ramanans1.github.io/plan2explore/">Project Website</a>


[1]:https://blog.ml.cmu.edu/2020/10/06/plan2explore/
