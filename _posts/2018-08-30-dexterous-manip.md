---
layout:             post
title:              "Dexterous Manipulation with Reinforcement Learning: Efficient, General, and Low-Cost"
date:               2018-08-30 12:00:00
author:             <a href="https://people.eecs.berkeley.edu/~abhigupta/">Abhishek Gupta</a>, <a href="https://www.linkedin.com/in/henry-zhu-711a1411a/">Henry Zhu</a>, <a href="https://homes.cs.washington.edu/~aravraj/">Aravind Rajeswaran</a>, <a href="https://vikashplus.github.io/index.html">Vikash Kumar</a>, and <a href="https://people.eecs.berkeley.edu/~svlevine/">Sergey Levine</a>
img:                /assets/dex_manip/dex_manip_img.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/teaser.gif"
         height="300"
         alt="..."/>
    <br/>
</p>

In this post, we demonstrate how deep reinforcement learning (deep RL) can be
used to learn how to control dexterous hands for a variety of manipulation
tasks. We discuss how such methods can learn to make use of low-cost hardware,
can be implemented efficiently, and how they can be complemented with techniques
such as demonstrations and simulation to accelerate learning.

<!--more-->

## Why Dexterous Hands?

A majority of robots in use today use simple parallel jaw grippers as
manipulators, which are sufficient for structured settings like factories.
However, manipulators that are capable of performing a wide array of tasks are
essential for unstructured human-centric environments like the home.
Multi-fingered hands are among the most versatile manipulators, and enable a
wide variety of skills we use in our everyday life such as moving objects,
opening doors, typing, and painting.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/missing_img2.png"
         height="180"
         alt="..."/>
    <br/>
</p>


Unfortunately, controlling dexterous hands is extremely difficult, which limits
their use. High-end hands can also be extremely expensive, due to delicate
sensing and actuation. Deep reinforcement learning offers the promise of
automating complex control tasks even with cheap hardware, but many applications
of deep RL use huge amounts of simulated data, making them expensive to deploy
in terms of both cost and engineering effort. Humans can learn motor skills
efficiently, without a simulator and without the computational power of a data
center.

We will first show that deep RL can in fact be used to learn complex
manipulation behaviors by training directly in the real world, with modest
computation and low-cost robotic hardware, and without any model or simulator.
We then describe how learning can be further accelerated by incorporating
additional sources of supervision, including demonstrations and simulation. We
demonstrate learning on two separate hardware platforms: an inexpensive
custom-built 3-fingered hand (the Dynamixel Claw), which costs under \\$2500, and
the higher-end Allegro hand, which costs about \\$15,000.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/missing_img1.png"
         height="230"
         alt="..."/>
    <br/>
    <i>
    Left: Dynamixel Claw. Right: Allegro Hand.
    </i>
</p>


## Model-free Reinforcement Learning in the Real World

Deep RL algorithms learn by trial and error, maximizing a user-specified reward
function from experience. We’ll use a valve rotation task as a working example,
where the hand must open a valve or faucet by rotating it 180 degrees.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/hand_valve_rotation.gif"
         height="300"
         alt="..."/>
    <br/>
    <i>
    Illustration of valve rotation task.
    </i>
</p>

The reward function simply consists of the negative distance between the current
and desired valve orientation, and the hand must figure out on its own how to
move to rotate it. A central challenge in deep RL is in using this weak reward
signal to find a complex and coordinated behavior strategy (a *policy*) that
succeeds at the task. The policy is represented by a multilayer neural network.
This typically requires a large number of trials, which has led some researchers
to consider deep RL methods to only be suitable in simulation. However, this
imposes major limitations on their applicability: learning directly in the real
world makes it possible to learn any task from experience, while using
simulators requires designing a suitable simulation, modeling the task and the
robot, and carefully adjusting their parameters to achieve good results. We will
show later that simulation can accelerate learning substantially, but we first
demonstrate that existing RL algorithms can in fact learn this task directly on
real hardware.

A variety of algorithms should be suitable. We use [Truncated Natural Policy
Gradient][1] to learn the task, which requires about 9 hours on real hardware.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/iter40_valve.gif"
         height="230"
         alt="..."/>
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/iter80_valve.gif"
         height="230"
         alt="..."/>
    <br/>
    <i>
    Learning progress of the dynamixel claw on valve rotation.
    </i>
</p>

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/real_rl_valve.gif"
         height="300"
         alt="..."/>
    <br/>
    <i>
    Final Trained Policy on valve rotation.
    </i>
</p>

The direct RL approach is appealing for a number of reasons. It requires minimal
assumptions, and is thus well suited to autonomously acquire a large repertoire
of skills. Since this approach assumes no information other than access to a
reward function, it is easy to relearn the skill in a modified environment, for
example when using a different object or a different hand -- in this case, the
Allegro hand.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/allegrohand.gif"
         height="420"
         alt="..."/>
    <br/>
    <i>
    360° valve rotation with Allegro Hand.
    </i>
</p>

The same exact method can learn to rotate the valve when we use a different
material. We can learn how to rotate a valve made out of foam. This can be quite
difficult to simulate accurately, and training directly in the real world allows
us to learn without needing accurate simulations.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/foamscrew.gif"
         height="360"
         alt="..."/>
    <br/>
    <i>
    Dynamixel claw rotating a foam screw.
    </i>
</p>

The same approach takes 8 hours to solve a different task, which requires
flipping an object 180 degrees around the horizontal axis, without any
modification.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/flip_hand.gif"
         height="230"
         alt="..."/>
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/real_rl_flipping.gif"
         height="230"
         alt="..."/>
    <br/>
    <i>
    Dynamixel claw flipping a block.
    </i>
</p>

These behaviors were learned with low cost hardware (<\\$2500) and a single
consumer desktop computer.

## Accelerating Learning with Human Demonstrations

While model-free RL is extremely general, incorporating supervision from human
experts can help accelerate learning further. One way to do this, which we
describe in our paper on [Demonstration Augmented Policy Gradient (DAPG)][2], is
to incorporate human demonstrations into the reinforcement learning process.
Related approaches have been proposed in the context of [off-policy RL][3],
[Q-learning][4], and [other robotic tasks][5]. The key idea behind DAPG is that
demonstrations can be used to accelerate RL in two ways

<img 
src="http://bair.berkeley.edu/static/blog/dex_manip/missing_img3.png"
height="300"
align="right"
hspace="30"
alt="..."
/>

1. Provide a good initialization for the policy via behavior cloning.

2. Provide an auxiliary learning signal *throughout* the learning process to
guide exploration using a trajectory tracking auxiliary reward.

The auxiliary objective during RL prevents the policy from diverging from the
demonstrations during the RL process. Pure behavior cloning with limited data is
often ineffective in training successful policies due to distribution drift and
limited data support. RL is crucial for robustness and generalization and use of
demonstrations can substantially accelerate the learning process.  We previously
validated this algorithm in simulation on a variety of tasks, shown below, where
each task used only 25 human demonstrations collected in virtual reality.  

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/task_relocate.gif"
         height="160"
         alt="..."/>
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/task_hammer.gif"
         height="160"
         alt="..."/>
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/task_pen.gif"
         height="160"
         alt="..."/>
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/task_door.gif"
         height="160"
         alt="..."/>
    <br/>
    <i>
    Behaviors learned in simulation with DAPG: object pickup, tool use, in-hand,
    door opening.
    </i>
</p>

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/dapg_robustness.gif"
         height="350"
         alt="..."/>
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/pure_rl_robustness.gif"
         height="350"
         alt="..."/>
    <br/>
    <i>
    Behaviors robust to size and shape variations.
    </i>
</p>

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/natural_motion_dapg.gif"
         height="350"
         alt="..."/>
    <br/>
    <i>
    Natural and smooth behavior.
    </i>
</p>

In the real world, we can use this algorithm with the dynamixel claw to
significantly accelerate learning. The demonstrations are collected with
kinesthetic teaching, where a human teacher moves the fingers of the robots
directly in the real world. This brings down the training time on both tasks to
under 3 hours.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/dapg_valve.gif"
         height="230"
         alt="..."/>
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/dapg_flipping.gif"
         height="230"
         alt="..."/>
    <br/>
    <i>
    Left: Valve rotation policy with DAPG. Right: Flipping policy with DAPG.
    </i>
</p>

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/missing_img4.png"
         height="320"
         alt="..."/>
    <br/>
    <i>
    Learning Curves of RL from scratch on hardware vs DAPG.
    </i>
</p>

Demonstrations provide a natural way to incorporate human priors and accelerate
the learning process. Where high quality successful demonstrations are
available, augmenting RL with demonstrations has the potential to substantially
accelerate RL. However, obtaining demonstrations may not be possible for all
tasks or robot morphologies, necessitating the need to also pursue alternate
acceleration schemes.

## Accelerating Learning with Simulation

A simulated model of the task can help augment the real world data with large
amounts of simulated data to accelerate the learning process. For the simulated
data to be representative of the complexities of the real world, randomization
of various simulation parameters is often necessitated. This kind of
randomization has previously been observed to produce [robust][6] policies, and
can facilitate transfer in the face of both [visual][7] and [physical][8]
[discrepancies][9].

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/sim2real.gif"
         height="320"
         alt="..."/>
    <br/>
    <i>
    Policy for valve rotation transferred from simulation using randomization.
    </i>
</p>


Transfer from simulation has also been explored in [concurrent work][10] for
dexterous manipulation, and in a number of prior works for tasks such as
[picking and placing][11], [visual servoing][12], and [locomotion][13]. While
simulation to real transfer enabled by randomization is an appealing option,
especially for fragile robots, it has a number of limitations. First, the
resulting policies can end up being overly conservative due to the
randomization, a phenomenon that has been widely observed in the field of robust
control. Second, the particular choice of parameters to randomize is crucial for
good results, and insights from one task or problem domain may not transfer to
others. Third, increasing the amount of randomization results in more complex
models tremendously increasing the training time and required computational
resources ([Andrychowicz et al][10] report 100 years of simulated experience,
training in 50 hours on thousands of CPU cores). Directly training in the real
world may be more efficient and lead to better policies. Finally, and perhaps
most importantly, an accurate simulator must be constructed manually, with each
new task modeled by hand in the simulation, which requires substantial time and
expertise. However, leveraging simulations appropriately can tremendously
accelerate the learning, and more systematic transfer methods are an important
direction for future work.

## Accelerating Learning with Learned Models

In some of [our previous work][14], we also studied how learned dynamics models
can accelerate real-world reinforcement learning without access to manually
engineered simulators. In this approach, local derivatives of the dynamics are
approximated by fitting time-varying linear systems, which can then be used to
locally and iterative improve a policy. This approach can acquire a variety of
in-hand manipulation strategies from scratch in the real world. Furthermore, we
see that the same algorithm can even [learn to control][15] a pneumatic soft
robotic hand to perform a number of dexterous behaviors

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/learned_local_models_adroit.gif"
         height="230"
         alt="..."/>
    <img src="http://bair.berkeley.edu/static/blog/dex_manip/softhand.gif"
         height="230"
         alt="..."/>
    <br/>
    <i>
    Left: Adroit robotic hand performing in-hand manipulation. Right: Pneumatic
    Soft RBO Hand performing dexterous tasks.
    </i>
</p>

However, the performance of methods with learned models is limited by the
quality of the model that can be learned, and in practice asymptotic performance
is often still higher with the best model-free algorithms. Further study of
model-based reinforcement learning for efficient and effective real-world
learning is a promising research direction.


## Takeaways and Challenges

While training in the real world is general and broadly applicable, it has several challenges of its own.

1. Due to the requirement to take a large number of exploratory actions, we
observed that the hands often heat up quickly, which requires pauses to avoid
damage.

2. Since the hands must attempt the task multiple times, we had to build an
automatic reset mechanism. In the future, a promising direction to remove this
requirement is to [automatically learn][16] [reset policies][17].

3. Reinforcement learning methods require rewards to be provided, and this
reward must still be designed manually. Some of our recent work has looked at
automating reward specification.

However, enabling robots to learn complex skills directly in the real world is
one of the best paths forward to developing truly generalist robots. In the same
way that humans can learn directly from experience in the real world, robots
that can acquire skills simply by trial and error can explore novel solutions to
difficult manipulation problems and discover them with minimal human
intervention. At the same time, the availability of demonstrations, simulators,
and other prior knowledge can further reduce training times.

<hr>

The work in this post is based on these papers: 

- [Optimal control with learned local models: Application to dexterous manipulation][18]
- [Learning Dexterous Manipulation for a Soft Robotic Hand from Human Demonstration][19]
- [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations][20]

*A complete paper on the new robotic experiments will be released soon. The
research was conducted by Henry Zhu, Abhishek Gupta, Vikash Kumar, Aravind
Rajeswaran, and Sergey Levine. Collaborators on earlier projects include Emo
Todorov, John Schulman, Giulia Vezzani, Pieter Abbeel, Clemens Eppner.*

[1]:https://arxiv.org/abs/1703.02660
[2]:https://arxiv.org/abs/1709.10087
[3]:https://arxiv.org/abs/1709.10089
[4]:https://arxiv.org/abs/1704.03732
[5]:http://is.tuebingen.mpg.de/fileadmin/user_upload/files/publications/ICRA2009-Kober_5661[0].pdf
[6]:https://arxiv.org/abs/1610.01283
[7]:https://arxiv.org/abs/1703.06907
[8]:https://xbpeng.github.io/projects/SimToReal/2018_SimToReal.pdf
[9]:https://arxiv.org/abs/1803.10371
[10]:https://arxiv.org/abs/1808.00177
[11]:https://arxiv.org/abs/1707.02267
[12]:https://arxiv.org/abs/1712.07642
[13]:https://arxiv.org/abs/1804.10332
[14]:https://homes.cs.washington.edu/~todorov/papers/KumarICRA16.pdf
[15]:https://arxiv.org/abs/1603.06348
[16]:https://arxiv.org/abs/1711.06782
[17]:http://rll.berkeley.edu/reset_controller/reset_controller.pdf
[18]:https://homes.cs.washington.edu/~todorov/papers/KumarICRA16.pdf
[19]:https://arxiv.org/abs/1603.06348
[20]:https://arxiv.org/abs/1709.10087
