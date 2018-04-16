---
layout:             post
title:              "Shared Autonomy via Deep Reinforcement Learning"
date:               2018-04-17 9:00:00
author:             <a href="https://people.eecs.berkeley.edu/~reddy/">Siddharth Reddy</a>
img:                /assets/shared-autonomy/javdani.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<p style="text-align:center;">
<img width="100%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/small-cat-gapped-opt.gif">
<br>
<i>
A blind, autonomous pilot (left), suboptimal human pilot (center), and combined
human-machine team (right) play the Lunar Lander game.
</i>
</p>

Imagine a drone pilot remotely flying a quadrotor, using an onboard camera to
navigate and land. Unfamiliar flight dynamics, terrain, and network latency can
make this system challenging for a human to control. One approach to this
problem is to train an autonomous agent to perform tasks like patrolling and
mapping without human intervention. This strategy works well when the task is
clearly specified and the agent can observe all the information it needs to
succeed. Unfortunately, many real-world applications that involve human users do
not satisfy these conditions: the user's intent is often private information
that the agent cannot directly access, and the task may be too complicated for
the user to precisely define. For example, the pilot may want to track a set of
moving objects (e.g., a herd of animals) and change object priorities on the fly
(e.g., focus on individuals who unexpectedly appear injured). *Shared autonomy*
addresses this problem by combining user input with automated assistance; in
other words, *augmenting* human control instead of replacing it.

<!--more-->


# Background

The idea of combining human and machine intelligence in a shared-control system
goes back to the early days of Ray Goertz's 
<a href="https://www.osti.gov/servlets/purl/1054625">master-slave manipulator</a>
in 1949, Ralph Mosher's 
<a href="http://www.dtic.mil/dtic/tr/fulltext/u2/701359.pdf">Hardiman exoskeleton</a> 
in 1969, and Marvin Minsky's call for 
<a href="https://spectrum.ieee.org/robotics/artificial-intelligence/telepresence-a-manifesto">telepresence</a>
in 1980. After decades of research in robotics, human-computer interaction, and
artificial intelligence, interfacing between a human operator and a
remote-controlled robot remains a challenge. According to a
<a href="https://www.cs.cmu.edu/~cga/drc/jfr-what.pdf">review</a> 
of the 2015 
<a href="https://www.darpa.mil/program/darpa-robotics-challenge">DARPA Robotics Challenge</a>,
"the most cost effective research area to improve robot performance is
Human-Robot Interaction....The biggest enemy of robot stability and performance
in the DRC was operator errors. Developing ways to avoid and survive operator
errors is crucial for real-world robotics. Human operators make mistakes under
pressure, especially without extensive training and practice in realistic
conditions."

<table class="col-3">
  <tbody><tr>
    <td style="text-align:center;">
      <img widt="100%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/goertz.png">
    </td>
    <td style="text-align:center;">
      <img width="100%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/bci.png">
    </td>
    <td style="text-align:center;">
      <img width="100%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/javdani.png">
    </td>
  </tr>
   <td><p style="text-align:center;">
      <i>Master-slave robotic manipulator <a href="https://www.osti.gov/servlets/purl/1054625">(Goertz, 1949)</a></i>
    </p></td>
   <td><p style="text-align:center;">
      <i>Brain-computer interface for neural prosthetics <a href="http://www.cell.com/neuron/abstract/S0896-6273(14)00739-9">(Shenoy & Carmena, 2014)</a></i>
    </p></td>
   <td><p style="text-align:center;">
      <i>An instantiation of shared autonomy <a href="https://arxiv.org/pdf/1503.07619.pdf">(Javdani et al., 2015)</a></i>
    </p></td>
</tbody></table>


One research thrust in shared autonomy approaches this problem by inferring the
user's goals and autonomously acting to achieve them. Chapter 5 of Shervin
Javdani's <a
href="https://repository.cmu.edu/cgi/viewcontent.cgi?article=2100&context=dissertations">Ph.D.
thesis</a> contains an excellent review of the literature. Such methods have
made progress toward better <a
href="https://people.csail.mit.edu/jalonsom/docs/17-schwartig-autonomy-icra.pdf">driver
assist</a>, <a href="https://arxiv.org/pdf/1503.05451.pdf">brain-computer
interfaces for prosthetic limbs</a>, and <a
href="http://www.roboticsproceedings.org/rss08/p16.pdf">assistive
teleoperation</a>, but tend to require prior knowledge about the world;
specifically, (1) a dynamics model that predicts the consequences of taking a
given action in a given state of the environment, (2) the set of possible goals
for the user, and (3) an observation model that describes the user's behavior
given their goal. Model-based shared autonomy algorithms are well-suited to
domains in which this knowledge can be directly hard-coded or learned, but are
challenged by unstructured environments with ill-defined goals and
unpredictable user behavior. We approached this problem from a different angle,
using *deep reinforcement learning* to implement *model-free* shared autonomy.

<a href="https://arxiv.org/pdf/1708.05866.pdf">Deep reinforcement learning</a>
uses neural network function approximation to tackle the curse of
dimensionality in high-dimensional, continuous state and action spaces, and has
recently achieved remarkable success in training autonomous agents from scratch
to <a href="https://www.nature.com/articles/nature14236">play video games</a>,
<a href="https://www.nature.com/articles/nature24270">defeat human world
champions at Go</a>, and <a href="https://arxiv.org/pdf/1504.00702.pdf">control
robots</a>. We have taken preliminary steps toward answering the following
question: *can deep reinforcement learning be useful for building flexible and
practical assistive systems?*

# Our Approach

To enable shared-control teleoperation with minimal prior assumptions, we
devised a model-free deep reinforcement learning algorithm for shared autonomy.
The key idea is to learn an end-to-end mapping from environmental observation
and user input to agent action, with task reward as the only form of
supervision. From the agent's perspective, the user acts like a prior policy
that can be fine-tuned, and an additional sensor generating observations from
which the agent can implicitly decode the user's private information. From the
user's perspective, the agent behaves like an adaptive interface that learns a
personalized mapping from user commands to actions that maximizes task reward.

<p style="text-align:center;">
<img align="middle" width="75%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/deepassist-diagram.png">
<br><br>
<i>
Fig. 1: An overview of our human-in-the-loop deep Q-learning algorithm for model-free shared autonomy
</i>
</p>

We used human-in-the-loop deep Q-learning to learn an approximate state-action
value function that computes the expected future return of an action given the
current environmental observation and the user's input. The assistive agent
executes the closest high-value action to the user's control input. The reward
function for the agent is a combination of known terms computed for every
state, and a terminal reward provided by the user upon succeeding or failing at
the task. See Fig. 1 for a high-level schematic of this process.

# Learning to Assist

<a href="https://arxiv.org/pdf/1706.00155.pdf">Prior work</a> has formalized
shared autonomy as a partially-observable Markov decision process (POMDP) in
which the user's goal is initially unknown to the agent and must be inferred in
order to complete the task. Existing methods tend to assume the following
components of the POMDP are known ex-ante: (1) the dynamics of the environment,
or the state transition distribution $T$; (2) the set of possible goals for the
user, or the goal space $\mathcal{G}$; and (3) the user's control policy given
their goal, or the user model $\pi_h$. In our work, we relaxed these three
standard assumptions. We introduced a model-free deep reinforcement learning
method, with variants that can also take advantage of a known user model and
goal space when they do exist, but still provide assistance even when they are
not available.

In our problem formulation, the transition distribution $T$, the user's policy
$\pi_h$, and the goal space $\mathcal{G}$ are no longer all necessarily known
to the agent. The reward function, which depends on the user's private
information, is 

$$
R(s, a, s') = \underbrace{R_{\text{general}}(s, a, s')}_\text{known} + \underbrace{R_{\text{feedback}}(s, a, s')}_\text{unknown, but observed}.
$$

This decomposition follows a structure typically present in shared autonomy:
there are some terms in the reward that are known, such as the need to avoid
collisions. We capture these in $R_{general}$. $R_{\text{feedback}}$ is
user-generated feedback that depends on their private information. We do not
know this function. We merely assume the agent is informed when the user
provides feedback (e.g., by pressing a button). In practice, the user might
simply indicate once per trial whether the agent succeeded or not.

## Incorporating User Input

Our method jointly embeds the agent's observation of the environment $s_t$ with the information from the user $u_t$ by simply concatenating them. Formally,
$$
\tilde{s}_t = \left[ \begin{array}{c} s_t \\ u_t \end{array} \right].
$$
The particular form of $u_t$ depends on the information that is available. When we do not know the set of possible goals $\mathcal{G}$, we use the user's actions $a^h_t$ as $u_t$. When we know more about the possible user goals and policy, we set $u_t$ to the inferred goal $\hat{g}_t$.

## Q-Learning with User Control


Model-free reinforcement learning with a human in the loop poses two
challenges: (1) maintaining informative user input and (2) minimizing the
number of interactions with the environment. (1) If the user input is a
suggested control, consistently ignoring the suggestion and taking a different
action can degrade the quality of user input, since humans rely on feedback
from their actions to perform real-time control tasks. Additionally, some user
policies may already be approximately optimal and only require fine-tuning. (2)
Many model-free reinforcement learning algorithms require a large number of
interactions with the environment, which may be impractical for human users.

To mitigate these two issues, we used <a
href="https://www.nature.com/articles/nature14236">deep</a> <a
href="http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf">Q-learning</a> to
learn an approximate state-action value function that can be used to select and
evaluate actions. We chose Q-learning because (a) it is an off-policy
algorithm, so we did not need to exactly follow the agent's policy and could
explicitly trade off control between the user and agent, and (b) off-policy
Q-learning tends to be more sample-efficient than policy gradient and Monte
Carlo value-based methods.

## Control Sharing

Motivated by the discussion of (1) and (a) in the previous section, we used the
following behavior policy to select actions during and after Q-learning: select
a feasible action closest to the user's suggestion, where an action is feasible
if it isn't that much worse than the optimal action. Formally,

$$
\pi_{\alpha}(a \mid \tilde{s}, a^h) = \delta\left(a = \mathop{\arg\max}\limits_{\{a : Q'(\tilde{s}, a) \geq (1 - \alpha) Q'(\tilde{s}, a^\ast)\}} f(a, a^h)\right),
$$

where $f$ is an action-similarity function and $Q'(\tilde{s}, a) = Q(\tilde{s},
a) - \min_{a' \in \mathcal{A}} Q(\tilde{s}, a')$ maintains a sane comparison
for negative Q values. The constant $\alpha \in [0, 1]$ is a hyperparameter
that controls the tolerance of the system to suboptimal human suggestions, or
equivalently, the amount of assistance.

# User Studies

We applied our method to two real-time assistive control problems: the <a
href="https://gym.openai.com/envs/LunarLander-v2/">Lunar Lander game</a> and a
quadrotor landing task. Both tasks involved controlling motion using a discrete
action space and low-dimensional state observations that include position,
orientation, and velocity information. In both tasks, the human pilot had
private information that was necessary to complete the task, but wasn't capable
of succeeding on their own.

## The Lunar Lander Game

The objective of the game was to land the vehicle between the flags without
crashing or flying out of bounds using two lateral thrusters and a main engine.
The assistive copilot could observe the lander's position, orientation, and
velocity, but not the position of the flags.

<table class="col-2">
  <tbody><tr>
    <td style="text-align:center;">
      <img width="100%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/human-solo-lander-opt.gif">
    </td>
    <td style="text-align:center;">
      <img width="100%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/human-assisted-lander-opt.gif">
    </td>
  </tr>
   <td><p>
      <i><b>Human Pilot (Solo):</b> The human pilot can't stabilize and keeps crashing.</i>
    </p></td>
    <td><p>
      <i><b>Human Pilot + RL Copilot:</b> The copilot improves stability while giving the pilot enough freedom to land between the flags.</i>
    </p></td>
</tbody></table>

Humans rarely beat the Lunar Lander game on their own, but with a copilot they did much better.

<p style="text-align:center;">
<img width="49%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/lander-user-study-fig.png" title="Lunar Lander User Study Success vs. Crash Rates">
<br>
<i>
Fig. 2a: Success and crash rates averaged over 30 episodes.
</i>
<br>
<img width="49%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/human-pilot-solo-traj.png" title="Lunar Lander User Study Solo Trajectories">
<img width="49%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/human-pilot-assisted-traj.png" title="Lunar Lander User Study Assisted Trajectories">
<br>
<i>
Fig. 2b-c: Trajectories followed by human pilots with and without a copilot on
Lunar Lander. Red trajectories end in a crash or out of bounds, green in
success, and gray in neither. The landing pad is marked by a star. For the sake
of illustration, we only show data for a landing site on the left boundary.
</i>
</p>

One of the drawbacks of analyzing Lunar Lander is that the game interface and
physics do not reflect the complexity and unpredictability of a real-world
robotic shared autonomy task. To evaluate our method in a more realistic
environment, we formulated a task for a human pilot flying a real quadrotor.


## Quadrotor Landing Task

The objective of the task was to land a <a
href="https://www.parrot.com/us/drones/parrot-ardrone-20-elite-edition#parrot-ardrone-20-elite-edition">Parrot
AR-Drone 2</a> on a small, square landing pad at some distance from its initial
take-off position, such that the drone's first-person camera was pointed at a
random object in the environment (e.g., a red chair), without flying out of
bounds or running out of time. The pilot used a keyboard to control velocity,
and was blocked from getting a third-person view of the drone so that they had
to rely on the drone's first-person camera feed to navigate and land. The
assistive copilot observed position, orientation, and velocity, but did not
know which object the pilot wanted to look at.

<table class="col-2">
  <tbody><tr>
    <td style="text-align:center;">
			<img width="100%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/human-solo-quad-opt.gif">
		</td>
    <td style="text-align:center;">
      <img width="100%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/human-assisted-quad-opt.gif">
		</td>
  </tr>
  <tr>
    <td><p>
      <i><b>Human Pilot (Solo):</b> The pilot's display only showed the drone's first-person view, so pointing the camera was easy but finding the landing pad was hard.</i>
		</p></td>
    <td><p>
      <i><b>Human Pilot + RL Copilot:</b> The copilot didn't know where the pilot wanted to point the camera, but it knew where the landing pad was. Together, the pilot and copilot succeeded at the task.</i>
		</p></td>
  </tr>
</tbody></table>

Humans found it challenging to simultaneously point the camera at the desired
scene and navigate to the precise location of a feasible landing pad under time
constraints. The assistive copilot had little trouble navigating to and landing
on the landing pad, but did not know where to point the camera because it did
not know what the human wanted to observe after landing. Together, the human
could focus on pointing the camera and the copilot could focus on landing
precisely on the landing pad.

<p style="text-align:center;">
<img width="49%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/quad-user-study-fig.png" title="Quadrotor User Study Success vs. Crash Rates">
<br>
<i>
Fig. 3a: Success and crash rates averaged over 20 episodes.
</i>
<br>
<img width="49%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/human-pilot-solo-traj-quad.png" title="Quadrotor User Study Solo Trajectories">
<img width="49%" src="http://bair.berkeley.edu/static/blog/shared-autonomy/human-pilot-assisted-traj-quad.png" title="Quadrotor User Study Assisted Trajectories">
<br>
<i>
Fig. 3b-c: A bird's-eye view of trajectories followed by human pilots with and without a copilot on the quadrotor landing task. Red trajectories end in a crash or out of bounds, green in success, and gray in neither. The landing pad is marked by a star.
</i>
</p>

Our results showed that combined pilot-copilot teams significantly outperform individual pilots and copilots.

# What's Next?

Our method has a major weakness: model-free deep reinforcement learning
typically requires lots of training data, which can be burdensome for human
users operating physical robots. We mitigated this issue in our experiments by
pretraining the copilot in simulation without a human pilot in the loop.
Unfortunately, this is not always feasible for real-world applications due to
the difficulty of building high-fidelity simulators and designing rich
user-agnostic reward functions $R_{\text{general}}$. We are currently exploring
different approaches to this problem.

If you want to learn more, check out our pre-print on arXiv: *Siddharth Reddy,
Sergey Levine, Anca Dragan, <a href="https://arxiv.org/abs/1802.01744">Shared
Autonomy via Deep Reinforcement Learning</a>, arXiv, 2018.*

The paper will appear at <a href="http://www.roboticsconference.org/">Robotics:
Science and Systems 2018</a> from June 26-30. To encourage replication and
extensions, we have released <a href="https://github.com/rddy/deepassist">our
code</a>. Additional videos are available through the <a
href="https://sites.google.com/view/deep-assist">project website</a>.
