---
layout:             post
title:              "Assessing Generalization in Deep Reinforcement Learning"
date:               2019-03-18 9:00:00
author:             <a href="http://cseweb.ucsd.edu/~cpacker/">Charles Packer</a>$^*$ and Katelyn Gao$^*$
img:                /assets/rl_gen/DRE_figure.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<p style="text-align:center;">
    <!--
    <img src="http://bair.berkeley.edu/static/blog/rl_gen/6_panel.mp4.gif">
    -->
    <img src="http://bair.berkeley.edu/static/blog/rl_gen/6_panel_1080_24.gif"
    width="600">
<i>
</i>
</p>


## TL;DR

We present a benchmark for studying generalization in deep reinforcement
learning (RL). Systematic empirical evaluation shows that vanilla deep RL
algorithms generalize better than specialized deep RL algorithms designed
specifically for generalization. In other words, simply training on varied
environments is so far the most effective strategy for generalization. The code
can be found at [https://github.com/sunblaze-ucb/rl-generalization][12] and the
full paper is at [https://arxiv.org/abs/1810.12282][11].

<!--more-->

## Motivation

The ability to adapt to new, unseen situations and environments is a hallmark of
human intelligence. For example, drivers are able to quickly adjust to changes
in the road surface and weather conditions, new traffic rules, and
idiosyncrasies of particular vehicles. Humans are able to adapt their driving to
new scenarios because they have a representation of the world that generalizes
beyond the city in which they normally drive.

Deep RL has emerged as an important family of techniques that may support the
development of intelligent systems. Recent advances driven by deep RL include
achieving human-level performance on complex games such as [Atari][1], [Go][2],
and [Starcraft][3].

As excitement surrounding deep RL continues to grow, so is awareness of
limitations and pitfalls such as overfitting: the tendency of policies trained
via RL to specialize to their training domain and break down when deployed in
different circumstances. [Several][4] [recent][5] papers used Atari games to
show that deep RL is susceptible to dramatically overfit to the idiosyncrasies
of training environments. This problem is not limited to Atari: severe
overfitting has also been demonstrated in [other domains][6]. A major
contributor to this problem is the fact that deep RL agents are commonly trained
and tested in the same environment and are thus not encouraged to learn
representations that generalize to previously unseen circumstances. As deep RL
enters the mainstream, it is clear that the brittleness of such systems to minor
changes in the environment can have serious implications for real-world systems,
and this problem must be addressed before applying deep RL to problems like
autonomous driving, where errors may be catastrophic.

Generalization in deep RL has been recognized as an important problem and is
under active investigation. In the past year, OpenAI has released two benchmarks
for generalization in deep RL: the [Retro contest][7], which tests if a
game-playing AI generalizes to previously unseen levels of the same game (Sonic
The Hedgehog), and [CoinRun][8], a new game environment that tests agents’
ability to generalize to new levels created using a procedural generator.
[Justesen et al.  (2018)][9] used procedural generation of video game levels
during training to improve generalization to human-designed levels at test time.
Earlier, DeepMind released a suite of [“AI safety” gridworlds][10] designed to
test the susceptibility of RL agents to scenarios that can trigger unsafe
behavior (spoiler: state-of-the-art deep RL algorithms are highly fallible and
exhibit unsafe behavior). Despite these and other efforts, there is a lack of
consistency in the experimental methodology adopted in the literature.  Each
project uses a different set of evaluation metrics, environments, and
environment variations over which agents are expected to generalize. The merits
of different algorithms thus remain difficult to compare.

Our contribution is a framework for investigating generalization in deep RL with
a clearly defined set of environments (based on those already common in deep RL
research), environment variations, and evaluation metrics. We consider
variations in environment dynamics, such as when the road becomes slippery due
to rain, rather than variations in the environment goal, such as driving to a
new location in your hometown.

As a baseline, we also conduct a systematic empirical study of the
generalization performance of deep RL algorithms, including recent specialized
techniques for tackling generalization in deep RL. We differentiate between
interpolation to environments similar to those seen during training and
extrapolation to environments that are beyond the ranges observed during
training. Good extrapolation performance is crucial to the safe deployment of RL
agents to the real world.

## Environments

We build on four classic control environments from [OpenAI Gym][13] and two
locomotion environments from [Roboschool][14] that have been used in prior work
on generalization in deep RL. Specifically, we introduce new versions of
Acrobot, CartPole, MountainCar, Pendulum, HalfCheetah, and Hopper.

To test agents’ ability to generalize to changes in environment dynamics, the
environments are modified so that we can vary certain parameters that directly
affect the dynamics. For example, for the two locomotion environments
(HalfCheetah and Hopper), we vary the parameters for robot power, torso density,
and joint friction.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/rl_gen/cheetah friction.mp4" width="600"><br>
<i>
HalfCheetah under varying joint frictions.
</i>
</p>



Each environment has three versions:

- Default (D), where the parameters are fixed at the default values used in Gym
  or Roboschool.

- Random (R), where the parameters are uniformly sampled from ranges surrounding
  the values in D. Practically, this represents the distribution of environments
  from which it is possible to obtain training data.

- Extreme (E), where the parameters are uniformly sampled outside the ranges
  used in R. The values are more extreme than those in R. The Extreme mode
  represents edge cases and other environments for which it is difficult to
  obtain training data.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/rl_gen/DRE_figure.png"
    width="600"><br>
<i>
In CartPole, we vary two parameters, the length and the mass of the pole; the
left panel shows a schematic of the parameter ranges in each version. In the
right panel we illustrate the environment when the length falls in the ranges of
the three versions (E in <b>red</b>, R in <b>green</b>, and D as a fixed point
in R).  The E range excludes the R (and D) range.
</i>
</p>




## Evaluation metrics

In the spirit of RL as [goal-seeking][15], we measure the probability of
achieving a predefined goal instead of a scalar reward. Doing so has the
advantages of being invariant to the environment parameters and unaffected by
reward shaping. Though reward shaping can be useful in getting agents to learn
desired behaviors, it creates problems when using reward to compare the merits
of various algorithms, since the reward equations are often not easily
understood and hard to compare, as well as prone to change across environment
versions. For example, the equation used to calculate reward in Roboschool’s
HalfCheetah is an obtuse combination of body positioning, electricity, torque,
collisions, etc. A poorly constructed reward can also lead to trained agents
with [undesirable behavior][16].  Instead of directly using reward as a success
metric, we chose goals with a real-world meaning; for example, on HalfCheetah
the goal is to walk 20 meters, and the reported success is the percent of
attempts in which the agent achieves this goal.

We design a set of three metrics for generalization performance.

1. Default: success rate on D when trained on D. This is the classic RL setting
and thus serves as a baseline for the other two metrics.

2. Interpolation: success rate on R when trained on R. This measures the
performance of an agent on environments similar to, but not exactly the same as,
those seen during training.

3. Extrapolation: geometric mean of the success rates on E when trained on R, on
R when trained on D, and on E when trained on D. This measures the performance
of an agent on environments different from those seen during training.

## Results

We first evaluate two popular deep RL algorithms, one from the policy gradient
family, [PPO][17], and one from the actor-critic family, [A2C][18]. We then
evaluate two specialized methods for training agents that generalize:
[EPOpt][19] and [RL$^2$][20]. EPOpt trains an agent to be robust to environment
variations by maximizing a risk-sensitive reward while RL$^2$ aims to learn a
policy that can adapt to the environment at hand using the observed trajectory.
As general-purpose techniques, EPOpt and RL$^2$ are each combined with the two
vanilla deep RL algorithms.

The following chart shows the three generalization metrics discussed above,
averaged over the six environments and then summarized over five runs of the
experiments. FF refers to a feed-forward network architecture for the
policy/value function and RC refers to a recurrent one. For visual clarity, we
only show a subset of the algorithms here (see the paper for the complete table
of results).

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/rl_gen/benchmark_results_vanilla.png"
    width="500">
<i>
</i>
</p>

Aside from PPO with the recurrent network architecture, A2C and PPO were able to
interpolate fairly well but had limited extrapolation success. In other words,
simply training on a random environment, without adding any specialized
mechanism, results in agents that can generalize to similar environments.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/rl_gen/benchmark_results_generalize.png"
    width="500">
<i>
</i>
</p>

EPOpt provided improved interpolation and extrapolation performance, but only
when combined with PPO and a feed-forward architecture. In fact, the improvement
was seen in the environments with continuous action spaces (Pendulum,
HalfCheetah, and Hopper), indicating that this may be important to the success
of EPOpt. In other cases, EPOpt does not perform as well as the vanilla deep RL
algorithms.

RL$^2$ proved difficult to train, leading to poor generalization performance. In
most cases, a working policy was not found even for the fixed environment.

### Case study: HalfCheetah

Differences in gait are readily observable in HalfCheetah, making it a good
candidate for a case study. We look at EPOpt-PPO with the feed-forward
architecture, the best-performing algorithm on this environment, and compare it
to PPO with the feed-forward architecture.



<p style="text-align:center;">
    <!--
    <img src="http://bair.berkeley.edu/static/blog/rl_gen/4_panel_5s.mp4"
    width="600">
    -->
    <img src="http://bair.berkeley.edu/static/blog/rl_gen/4_panel_5s_1080_24.gif"
    width="600">
<i>
</i>
</p>

When trained on the fixed environment, EPOpt-PPO learns an agent that, compared
to PPO, takes smaller steps. Because EPOpt is designed to be robust, this makes
intuitive sense; when faced with an unfamiliar (possibly dangerous) terrain,
humans do very much the same in order to avoid falling. When trained on the
random environment, compared to the fixed environment, both PPO and EPOpt-PPO
learn slower gaits with the head leaning forward and downwards. We hypothesize
that such a body position helps the robot find balance across a wider range of
environmental parameters.

## Most related work

The [results of the OpenAI Retro contest][21], the [paper accompanying
CoinRun][22], and [Justesen et al. (2018)][23] also suggest that training with
environment stochasticity (i.e. on a distribution of environments) is an
effective strategy for generalization. In a similar vein, [Zhang et al.
(2018)][24] show that in a fixed environment, using multiple random seeds to
generate trajectories during training improves generalization to possible
changes at test time.

## Moving forward

The results of our empirical evaluation have shown that it is crucial to perform
a thorough study of proposed algorithms on a variety of tasks. Despite EPOpt
being an intuitively general-purpose approach, it only showed promising results
with PPO and a feed-forward policy network. The difficulty in training RL$^2$
suggests that a reward signal alone may not be sufficient for an LSTM to learn
something useful about the environment at hand from the current trajectory, and
so a more structured architecture or a model-based approach may be needed for
robust automatic test-time adaptation.


[1]:https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/
[2]:https://deepmind.com/research/alphago/
[3]:https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/
[4]:https://arxiv.org/abs/1709.06009
[5]:https://arxiv.org/abs/1812.02850
[6]:https://arxiv.org/abs/1806.07937
[7]:https://openai.com/blog/retro-contest/
[8]:https://github.com/openai/coinrun
[9]:https://arxiv.org/abs/1806.10729
[10]:https://deepmind.com/research/publications/ai-safety-gridworlds/
[11]:https://arxiv.org/abs/1810.12282
[12]:https://github.com/sunblaze-ucb/rl-generalization
[13]:https://gym.openai.com/
[14]:https://blog.openai.com/roboschool/
[15]:http://incompleteideas.net/book/the-book.html
[16]:https://openai.com/blog/faulty-reward-functions/
[17]:https://openai.com/blog/openai-baselines-ppo/
[18]:https://openai.com/blog/baselines-acktr-a2c/
[19]:https://arxiv.org/abs/1610.01283
[20]:https://arxiv.org/abs/1611.02779
[21]:https://openai.com/blog/first-retro-contest-retrospective/
[22]:https://arxiv.org/abs/1812.02341
[23]:https://arxiv.org/abs/1806.10729
[24]:https://arxiv.org/abs/1806.07937
