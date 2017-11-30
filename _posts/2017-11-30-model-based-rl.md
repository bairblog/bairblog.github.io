---
layout:     post
title:      "Model-based Reinforcement Learning with Neural Network Dynamics"
date:       2017-11-30 9:00:00
author:     Anusha Nagabandi, Gregory Kahn
visible:    True
img:        /assets/model-rl/fig_1a.png
excerpt_separator: <!--more-->
show_comments: True
---

<p style="text-align:center;">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_1a.png" height="240" style="margin: 10px;" alt="fig1a">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_1b.gif" height="240" style="margin: 10px;" alt="fig1b">
<br>
<i>
Fig 1. A learned neural network dynamics model enables a hexapod robot to learn
to run and follow desired trajectories, using just 17 minutes of real-world
experience.
</i>
</p>

Enabling robots to act autonomously in the real-world is difficult. [Really,
really difficult][1]. Even with expensive robots and teams of world-class
researchers, robots still have difficulty autonomously navigating and
interacting in complex, unstructured environments.

Why are autonomous robots not out in the world among us? Engineering systems
that can cope with all the complexities of our world is hard. From nonlinear
dynamics and partial observability to unpredictable terrain and sensor
malfunctions, robots are particularly susceptible to Murphy’s law: everything
that can go wrong, will go wrong. Instead of fighting Murphy’s law by coding
each possible scenario that our robots may encounter, we could instead choose to
embrace this possibility for failure, and enable our robots to learn from it.
Learning control strategies from experience is advantageous because, unlike
hand-engineered controllers, learned controllers can adapt and improve with more
data. Therefore, when presented with a scenario in which everything does go
wrong, although the robot will still fail, the learned controller will hopefully
correct its mistake the next time it is presented with a similar scenario. In
order to deal with complexities of tasks in the real world, current
learning-based methods often use deep neural networks, which are powerful but
not data efficient: These trial-and-error based learners will most often still
fail a second time, and a third time, and often thousands to millions of times.
The sample inefficiency of modern deep reinforcement learning methods is one of
the main bottlenecks to leveraging learning-based methods in the real-world.

We have been investigating sample-efficient learning-based approaches with
neural networks for robot control. For complex and contact-rich simulated
robots, as well as real-world robots (Fig. 1), our approach is able to learn
locomotion skills of trajectory-following using only minutes of data collected
from the robot randomly acting in the environment. In this blog post, we’ll
provide an overview of our approach and results. More details can be found in
our research papers listed at the bottom of this post, including [this paper][2]
with [code here][18].

<!--more-->


## Sample efficiency: model-free versus model-based

Learning robotic skills from experience typically falls under the umbrella of
reinforcement learning. Reinforcement learning algorithms can generally be
divided into two categories: model-free, which learn a policy or value function, and
model-based, which learn a dynamics model. While model-free deep reinforcement
learning algorithms are capable of learning a wide range of robotic skills, they
typically suffer from [very][3] [high][4] [sample][5] [complexity][6], often
requiring millions of samples to achieve good performance, and can typically
only learn a single task at a time. Although some prior work has deployed these
model-free algorithms for [real-world manipulation tasks][7], the high sample
complexity and inflexibility of these algorithms has hindered them from being
widely used to learn locomotion skills in the real world.

Model-based reinforcement learning algorithms are generally regarded as being
[more sample efficient][8]. However, to achieve good sample efficiency, these
model-based algorithms have conventionally used either relatively simple
[function][9] [approximators][10], which fail to generalize well to complex
tasks, or probabilistic dynamics models such as [Gaussian][11] [processes][12],
which generalize well but have difficulty with complex and high-dimensional
domains, such as systems with frictional contacts that induce discontinuous
dynamics.  Instead, we use medium-sized neural networks to serve as function
approximators that can achieve excellent sample efficiency, while still being
expressive enough for generalization and application to various complex and
high-dimensional locomotion tasks.


## Neural Network Dynamics for Model-Based Deep Reinforcement Learning

In our work, we aim to extend the successes that deep neural network models have
seen in other domains into model-based reinforcement learning. Prior efforts to
combine neural networks with model-based RL in recent years have not achieved
the kinds of results that are competitive with simpler models, such as [Gaussian
processes][11]. For example, [Gu et. al.][13] observed that even linear models
achieved better performance for synthetic experience generation, while [Heess
et. al.][14] saw relatively modest gains from including neural network models
into a model-free learning system. Our approach relies on a few crucial
decisions. First, we use the learned neural network model within a model
predictive control framework, in which the system can iteratively replan and
correct its mistakes. Second, we use a relatively short horizon look-ahead so
that we do not have to rely on the model to make very accurate predictions far
into the future. These two relatively simple design decisions enable our method
to perform a wide variety of locomotion tasks that have not previously been
demonstrated with general-purpose model-based reinforcement learning methods
that operate directly on raw state observations.

A diagram of our model-based reinforcement learning approach is shown in Fig. 2.
We maintain a dataset of trajectories that we iteratively add to, and we use
this dataset to train our dynamics model. The dataset is initialized with random
trajectories. We then perform reinforcement learning by alternating between
training a neural network dynamics model using the dataset, and using a model
predictive controller (MPC) with our learned dynamics model to gather additional
trajectories to aggregate onto the dataset. We discuss these two components
below.

<p style="text-align:center;">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_2.png" width="600" alt="fig2">
<br>
<i>
Fig 2. Overview of our model-based reinforcement learning algorithm.
</i>
</p>


### Dynamics Model

We parameterize our learned dynamics function as a deep neural network,
parameterized by some weights that need to be learned. Our dynamics function
takes as input the current state $s_t$ and action $a_t$, and outputs the
predicted state difference $s_{t+1}-s_t$. The dynamics model itself can be
trained in a supervised learning setting, where collected training data comes in
pairs of inputs $(s_t,a_t)$ and corresponding output labels $(s_{t+1},s_t)$. 

Note that the “state” that we refer to above can vary with the agent, and it can
include elements such as center of mass position, center of mass velocity, joint
positions, and other measurable quantities that we choose to include. 


### Controller

In order to use the learned dynamics model to accomplish a task, we need to
define a reward function that encodes the task. For example, a standard “x_vel”
reward could encode a task of moving forward. For the task of trajectory
following, we formulate a reward function that incentivizes staying close to the
trajectory as well as making forward progress along the trajectory. 

Using the learned dynamics model and task reward function, we formulate a
model-based controller. At each time step, the agent plans $H$ steps into the
future by randomly generating $K$ candidate action sequences, using the learned
dynamics model to predict the outcome of those action sequences, and selecting
the sequence corresponding to the highest cumulative reward (Fig. 3). We then
execute only the first action from the action sequence, and then repeat the
planning process at the next time step. This replanning makes the approach
robust to inaccuracies in the learned dynamics model.

<p style="text-align:center;">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_3.png" width="500" alt="fig3">
<br>
<i>
Fig 3. Illustration of the process of simulating multiple candidate action
sequences using the learned dynamics model, predicting their outcome, and
selecting the best one according to the reward function.
</i>
</p>



## Results

We first evaluated our approach on a variety of MuJoCo agents, including the
swimmer, half-cheetah, and ant. Fig. 4 shows that using our learned dynamics
model and MPC controller, the agents were able to follow paths defined by a set
of sparse waypoints. Furthermore, our approach used only *minutes* of random
data to train the learned dynamics model, showing its sample efficiency.

Note that with this method, we trained the model only once, but simply by
changing the reward function, we were able to apply the model at runtime to a
variety of different desired trajectories, without a need for separate
task-specific training.

<p style="text-align:center;">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_4a.gif" height="140" style="margin: 6px;" alt="fig4a">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_4b.gif" height="140" style="margin: 6px;" alt="fig4b">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_4c.gif" height="140" style="margin: 6px;" alt="fig4c"> <br>
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_4d.gif" height="140" style="margin: 6px;" alt="fig4d">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_4e.gif" height="140" style="margin: 6px;" alt="fig4e">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_4f.gif" height="140" style="margin: 6px;" alt="fig4f">
<br>
<i>
Fig 4: Trajectory following results with ant, swimmer, and half-cheetah. The
dynamics model used by each agent in order to perform these various trajectories
was trained just once, using only randomly collected training data.
</i>
</p>

What aspects of our approach were important to achieve good performance? We
first looked at the effect of varying the MPC planning horizon H. Fig. 5 shows
that performance suffers if the horizon is too short, possibly due to
unrecoverable greedy behavior. For half-cheetah, performance also suffers if the
horizon is too long, due to inaccuracies in the learned dynamics model. Fig. 6
illustrates our learned dynamics model for a single 100-step prediction, showing
that open-loop predictions for certain state elements eventually diverge from
the ground truth. Therefore, an intermediate planning horizon is best to avoid
greedy behavior while minimizing the detrimental effects of an inaccurate model.

<p style="text-align:center;">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_5.png" alt="fig5">
<br>
<i>
Fig 5: Plot of task performance achieved by controllers using different horizon
values for planning. Too low of a horizon is not good, and neither is too high
of a horizon.
</i>
</p>

<p style="text-align:center;">
<img
src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_6.png" width="600" alt="fig6">
<br>
<i>
Fig 6: A 100-step forward simulation (open-loop) of the dynamics model, showing
that open-loop predictions for certain state elements eventually diverge from
the ground truth.
</i>
</p>

We also varied the number of initial random trajectories used to train the
dynamics model. Fig. 7 shows that although a higher amount of initial training
data leads to higher initial performance, data aggregation allows even low-data
initialization experiment runs to reach a high final performance level. This
highlights how on-policy data from reinforcement learning can improve sample
efficiency. 

<p style="text-align:center;">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_7.png" alt="fig7">
<br>
<i>
Fig 7: Plot of task performance achieved by dynamics models that were trained
using differing amounts of initial random data.
</i>
</p>

It is worth noting that the final performance of the model-based controller is
still substantially lower than that of a very good model-free learner (when the
model-free learner is trained with thousands of times more experience). This
suboptimal performance is sometimes referred to as "model bias," and is a known
issue in model-based RL. To address this issue, we also proposed a hybrid
approach that combines model-based and model-free learning to eliminate the
asymptotic bias at convergence, though at the cost of additional experience.
This hybrid approach, as well as additional analyses, are available in our
paper.

## Learning to run in the real world

<p style="text-align:center;">
<img
src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_8.png" width="400" alt="fig8">
<br>
<i>
Fig 8: The VelociRoACH is 10 cm in length, approximately 30 grams in weight, can
move up to 27 body-lengths per second, and uses two motors to control all six
legs.
</i>
</p>

Since our model-based reinforcement learning algorithm can learn locomotion
gaits using orders of magnitude less experience than model-free algorithms, it
is possible to evaluate it directly on a real-world robotic platform. In other
work, we studied how this method can learn entirely from real-world experience,
acquiring locomotion gaits for a millirobot (Fig. 8) completely from scratch.

Millirobots are a promising robotic platform for many applications due to their
small size and low manufacturing costs. However, controlling these millirobots
is difficult due to their underactuation, power constraints, and size. While
hand-engineered controllers can sometimes control these millirobots, they often
have difficulties with dynamic maneuvers and complex terrains. We therefore
leveraged our model-based learning technique from above to enable the
VelociRoACH millirobot to do trajectory following. Fig. 9 shows that our
model-based controller can accurately follow trajectories at high speeds, after
having been trained using only 17 minutes of random data.

<p style="text-align:center;">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_9a.gif" height="200" style="margin: 10px;" alt="fig9a">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_9b.gif" height="200" style="margin: 10px;" alt="fig9b"> <br>
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_9c.gif" height="200" style="margin: 10px;" alt="fig9c">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/fig_9d.gif" height="200" style="margin: 10px;" alt="fig9d">
<br>
<i>
Fig 9: The VelociRoACH following various desired trajectories, using our
model-based learning approach.
</i>
</p>

To analyze the model’s generalization capabilities, we gathered data on both
carpet and styrofoam terrain, and we evaluated our approach as shown in Table 1.
As expected, the model-based controller performs best when executed on the same
terrain that it was trained on, indicating that the model incorporates knowledge
of the terrain. However, performance diminishes when the model is trained on
data gathered from both terrains, which likely indicates that more work is
needed to develop algorithms for learning models that are effective across
various task settings. Promisingly, Table 2 shows that performance increases as
more data is used to train the dynamics model, which is an encouraging
indication that our approach will continue to improve over time (unlike
hand-engineered solutions).

<p style="text-align:center;">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/table_1.png" width="600" alt="table1">
<br>
<i>
Table 1: Trajectory following costs incurred for models trained with different
types of data and for trajectories executed on different surfaces.
</i>
</p>

<p style="text-align:center;">
<img src="https://people.eecs.berkeley.edu/~nagaban2/misc/bair_blog_figs/table_2.png" width="600" alt="table2">
<br>
<i>
Table 2: Trajectory following costs incurred during the use of dynamics
models trained with differing amounts of data.
legs.
</i>
</p>

We hope that these results show the promise of model-based approaches for
sample-efficient robot learning and encourage future research in this area.

<hr>

We would like to thank Sergey Levine and Ronald Fearing for their feedback.

This post is based on the following papers:

- **Neural Network Dynamics Models for Control of Under-actuated Legged Millirobots** <br>
  A Nagabandi, G Yang, T Asmar, G Kahn, S Levine, R Fearing <br>
  [Paper][16]

- **Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning** <br>
  A Nagabandi, G Kahn, R Fearing, S Levine <br>
  [Paper][17], [Website][15], [Code][18]

[1]:https://www.youtube.com/watch?v=g0TaYhjpOfo
[2]:https://arxiv.org/pdf/1708.02596.pdf
[3]:http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf
[4]:https://people.eecs.berkeley.edu/~pabbeel/papers/2015-ICML-TRPO.pdf
[5]:https://arxiv.org/pdf/1611.02247.pdf
[6]:https://web.eecs.umich.edu/~baveja/Papers/ICML2016.pdf
[7]:https://arxiv.org/pdf/1610.00633.pdf
[8]:http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.436.44&rep=rep1&type=pdf
[9]:http://papers.nips.cc/paper/5444-learning-neural-network-policies-with-guided-policy-search-under-unknown-dynamics.pdf
[10]:http://ieeexplore.ieee.org/document/6907424/
[11]:http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf
[12]:http://ieeexplore.ieee.org/document/7010608/
[13]:https://arxiv.org/pdf/1603.00748.pdf
[14]:https://arxiv.org/pdf/1510.09142.pdf
[15]:https://sites.google.com/view/mbmf
[16]:https://arxiv.org/abs/1711.05253
[17]:https://arxiv.org/abs/1708.02596
[18]:https://github.com/nagaban2/nn_dynamics
