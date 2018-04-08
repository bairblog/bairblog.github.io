---
layout:             post
title:              "Towards a Virtual Stuntman"
date:               2018-04-10 9:00:00
author:             <a href="https://xbpeng.github.io/">Xue Bin (Jason) Peng</a>
img:                /assets/stuntman/teaser.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<p style="text-align:center;">
<img width="750" src="http://bair.berkeley.edu/static/blog/stuntman/teaser.gif">
<br>
<i>
Simulated humanoid performing a variety of highly dynamic and acrobatic skills.
</i>
</p>

Motion control problems have become standard benchmarks for reinforcement
learning, and deep RL methods have been shown to be effective for a diverse
suite of tasks ranging from manipulation to locomotion. However, characters
trained with deep RL often exhibit unnatural behaviours, bearing artifacts such
as jittering, asymmetric gaits, and <a
href="http://www.youtube.com/watch?v=hx_bgoTF7bs&t=1m28s">excessive movement of
limbs</a>. Can we train our characters to produce more natural behaviours?

<!--more-->

A wealth of inspiration can be drawn from computer graphics, where the
physics-based simulation of natural movements have been a subject of intense
study for decades. The greater emphasis placed on motion quality is often
motivated by applications in film, visual effects, and games. Over the years, a
rich body of work in physics-based character animation have developed
controllers to produce robust and natural motions for a large corpus of <a
href="https://www.youtube.com/watch?v=Mh8t_TuI3B4">tasks</a> and <a
href="https://www.cs.ubc.ca/~van/papers/2011-TOG-quadruped/index.html">characters</a>.
These methods often leverage human insight to incorporate task-specific control
structures that provide strong inductive biases on the motions that can be
achieved by the characters (e.g. <a
href="https://www.cs.ubc.ca/~van/papers/2013-TOG-MuscleBasedBipeds/index.html">finite-state
machines</a>, <a href="http://www.delasa.net/slip/index.html">reduced
models</a>, and <a
href="http://mrl.snu.ac.kr/research/ProjectManyMuscle/index.html">inverse
dynamics</a>). But as a result of these design decisions, the controllers are
often specific to a particular character or task, and controllers developed for
walking may not extend to more dynamic skills, where human insight becomes
scarce.

In this work, we will draw inspiration from the two fields to take advantage of
the generality afforded by deep learning models while also producing
naturalistic behaviours that rival the state-of-the-art in full body motion
simulation in computer graphics. We present a conceptually simple RL framework
that enables simulated characters to learn highly dynamic and acrobatic skills
from reference motion clips, which can be provided in the form of mocap data
recorded from human subjects. Given a single demonstration of a skill, such as a
spin-kick or a backflip, our character is able to learn a robust policy to
imitate the skill in simulation. Our policies produce motions that are nearly
indistinguishable from mocap.

{% include youtubePlayer.html id="lPdXtR8Ar-E" %}

# Motion Imitation

In most RL benchmarks, simulated characters are represented using simple models
that provide only a crude approximation of real world dynamics. Characters are
therefore prone to exploiting idiosyncrasies of the simulation to develop
unnatural behaviours that are infeasible in the real world. Incorporating more
realistic <a
href="https://www.crowdai.org/challenges/nips-2017-learning-to-run">biomechanical
models</a> can lead to more natural behaviours. But constructing high-fidelity
models can be extremely challenging, and the resulting motions may nonetheless
be unnatural.

An alternative is to take a data-driven approach, where reference motion capture
of humans provides examples of natural motions. The character can then be
trained to produce more natural behaviours by imitating the reference motions.
Imitating motion data in simulation has a 
<a href="http://graphics.cs.cmu.edu/?p=671">long</a> <a href="https://dl.acm.org/citation.cfm?id=2422388">history</a> in 
computer animation and has seen some recent 
<a href="https://xbpeng.github.io/projects/DeepLoco/index.html">demonstrations with deep RL</a>. 
While the results do appear more natural, they are still far from being able to
faithfully reproduce a wide variety of motions.

In this work, our policies will be trained through a motion imitation task,
where the goal of the character is to reproduce a given kinematic reference
motion. Each reference motion is represented by a sequence of target poses
$\{\hat{q}_0, \hat{q}_1,\ldots,\hat{q}_T\}$, where $\hat{q}_t$ is the target
pose at timestep $t$. The reward function is to minimize the least squares pose
error between the target pose $\hat{q}_t$ and the pose of the simulated
character $q_t$,

$$r_t = {\rm exp}\Big[-2 \|\hat{q}_t - q_t \|^2 \Big]$$

While more sophisticated methods have been applied for motion imitation, we
found that simply minimizing the tracking error (along with a couple of
additional insights) works surprisingly well. The policies are trained by
optimizing this objective using <a href="https://arxiv.org/abs/1707.06347">PPO</a>.

With this framework, we are able to develop policies for a rich repertoire of
challenging skills ranging from locomotion to acrobatics, martial arts to
dancing.

<p style="text-align:center;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/humanoid_sideflip.gif" height="160" style="margin: 10px;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/humanoid_cartwheel.gif" height="160" style="margin: 10px;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/humanoid_kipup.gif" height="160" style="margin: 10px;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/humanoid_speed_vault.gif" height="160" style="margin: 10px;">
<br>
<i>
The humanoid learns to imitate various skills. The blue character is the
simulated character, and the green character is replaying the respective mocap
clip. Top left: sideflip. Top right: cartwheel. Bottom left: kip-up. Bottom
right: speed vault.
</i>
</p>

Next, we compare our method with previous results that used (e.g. <a
href="https://arxiv.org/abs/1707.02201">generative adversarial imitation
learning (GAIL)</a>) to imitate mocap clips. Our method is substantially simpler
than GAIL and it is able to better reproduce the reference motions. The
resulting policy avoids many of the artifacts commonly exhibited by deep RL
methods, and enables the character to produce a fluid life-like running gait.

<p style="text-align:center;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/humanoid_run.gif" height="250" style="margin: 10px;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/deepmind.gif" height="250" style="margin: 10px;">
<br>
<i>
Comparison of our method (left) and work from Merel et al. [2017] using GAIL to
imitate mocap data. Our motions appear significantly more natural than previous
work using deep RL.
</i>
</p>

# Insights

## Reference State Initialization (RSI)

Suppose the character is trying to imitate a backflip. How would it know that
doing a full rotation midair will result in high rewards? Since most RL
algorithms are retrospective, they only observe rewards for states they have
visited. In the case of a backflip, the character will have to observe
successful trajectories of a backflip before it learns that those states will
yield high rewards. But since a backflip can be very sensitive to the initial
conditions at takeoff and landing, the character is unlikely to accidentally
execute a successful trajectory through random exploration. To give the
character a hint, at the start of each episode, we will initialize the character
to a state sampled randomly along the reference motion. So sometimes the
character will start on the ground, and sometimes it will start in the middle of
the flip. This allows the character to learn which states will result in high
rewards even before it has acquired the proficiency to reach those states.

<p style="text-align:center;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/no_rsi.png" height="280" style="margin: 10px;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/rsi.png" height="280" style="margin: 10px;">
<br>
<i>
RSI provides the character with a richer initial state distribution by
initializing it to random point along the reference motion.
</i>
</p>

Below is a comparison of the backflip policy trained with RSI and without RSI,
where the character is always initialized to a fixed initial state at the start
of the motion. Without RSI, instead of learning a flip, the policy just cheats
by hopping backwards.


<p style="text-align:center;">
<img width="750" src="http://bair.berkeley.edu/static/blog/stuntman/backflip_ablation.gif">
<br>
<i>
Comparison of policies trained without RSI or ET. RSI and ET can be crucial for
learning more dynamics motions. Left: RSI+ET. Middle: No RSI. Right: No ET.
</i>
</p>

## Early Termination (ET)

Early termination is a staple for RL practitioners, and it is often used to
improve simulation efficiency. If the character gets stuck in a state from which
there is no chance of success, then the episode is terminated early, to avoid
simulating the rest. Here we show that early termination can in fact have a
significant impact on the results. Again, let’s consider a backflip. During the
early stages of training, the policy is terrible and the character will spend
most of its time falling. Once the character has fallen, it can be extremely
difficult for it to recover. So the rollouts will be dominated by samples where
the character is just struggling in vain on the ground. This is analogous to the
class imbalance problem encountered by other methodologies such as supervised
learning. This issue can be mitigated by terminating an episode as soon as the
character enters such a futile state (e.g. falling). Coupled with RSI, ET helps
to ensure that a larger portion of the dataset consists of samples close to the
reference trajectory. Without ET the character never learns to perform a flip.
Instead, it just falls and then tries to mime the motion on the ground.

# More Results

In total, we have been able to learn over 24 skills for the humanoid just by
providing it with different reference motions.

<p style="text-align:center;">
<img width="750" src="http://bair.berkeley.edu/static/blog/stuntman/all_skills.gif">
<br>
<i>
Humanoid trained to imitate a rich repertoire of skills.
</i>
</p>

In addition to imitating mocap clips, we can also train the humanoid to perform
some additional tasks like kicking a randomly placed target, or throwing a ball
to a target.

<p style="text-align:center;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/humanoid_strikc_spinkick.gif" height="225" style="margin: 10px;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/humanoid_throw.gif" height="225" style="margin: 10px;">
<br>
<i>
Policies trained to kick and throw a ball to a random target.
</i>
</p>

We can also train a simulated Atlas robot to imitate mocap clips from a human.
Though the Atlas has a very different morphology and mass distribution, it is
still able to reproduce the desired motions. Not only can the policies imitate
the reference motions, they can also recover from pretty significant
perturbations.

<p style="text-align:center;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/atlas_spinkick.gif" height="200" style="margin: 10px;">
<img src="http://bair.berkeley.edu/static/blog/stuntman/atlas_backflip.gif" height="200" style="margin: 10px;">
<br>
<i>
Atlas trained to perform a spin-kick and backflip. The policies are robust to
significant perturbations.
</i>
</p>

But what do we do if we don’t have mocap clips? Suppose we want to simulate a
T-Rex. For various <a
href="https://www.nationalgeographic.com/science/prehistoric-world/dinosaur-extinction/">reasons</a>,
it is a bit difficult to mocap a T-Rex. So instead, we can have an artist
hand-animate some keyframes and then train a policy to imitate those.

<p style="text-align:center;">
<img width="750" src="http://bair.berkeley.edu/static/blog/stuntman/t-rex.gif">
<br>
<i>
Simulated T-Rex trained to imitate artist-authored keyframes.
</i>
</p>

By why stop at a T-Rex? Let's train a lion:

<p style="text-align:center;">
<img width="750" src="http://bair.berkeley.edu/static/blog/stuntman/lion3d_run.gif">
<br>
<i>
Simulated lion.
</i>
</p>

and a dragon:

<p style="text-align:center;">
<img width="750" src="http://bair.berkeley.edu/static/blog/stuntman/dragon.gif">
<br>
<i>
Simulated dragon with a 418D state space and 94D action space.
</i>
</p>

The story here is that a simple method ends up working surprisingly well. Just
by minimizing the tracking error, we are able to train policies for a diverse
collection of characters and skills. We hope this work will help inspire the
development of more dynamic motor skills for both simulated characters and
robots in the real world. Exploring methods for imitating motions from more
prevalent sources such as video is also an exciting avenue for scenarios that
are challenging to mocap, such as animals and cluttered environments. 

To learn more, [check out our paper][1].

We would like to thank the co-authors of this work: Pieter Abbeel, Sergey
Levine, and Michiel van de Panne. This project was done in collaboration with
the University of British Columbia.

[1]:https://xbpeng.github.io/projects/DeepMimic/index.html
