---
layout:             post
title:              "Reverse Curriculum Generation for Reinforcement Learning Agents"
date:               2017-12-19 9:00:00
author:             Carlos Florensa
img:                /assets/BAIR_Logo.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

Reinforcement Learning (RL) is a powerful technique capable of solving complex tasks such as locomotion (Schulman et al. 2015), Atari games (Mnih et al. 2015), racing games (Lillicrap et al. 2015), and robotic manipulation tasks (Levine et al. 2016), all through training an agent to optimize behaviors over a reward function. There are many tasks, however, for which it is **hard to design a reward function that is both conducive to training and that yields the desired behavior once optimized**. Suppose we want a robotic arm to learn how to place a ring onto a peg. The most natural reward function would be for an agent to receive a reward of 1 at the desired end configuration and 0 everywhere else.  However, the required motion for this task--to align the ring at the top of the peg and then slide it to the bottom--is impractical to learn under such a binary reward, because the usual random exploration of our initial policy is unlikely to ever reach the goal, as seen in Video 1a. Alternatively, one can try to shape the reward function (Ng et al. 1999; Popov et al. 2017) to potentially alleviate this problem, but finding a good shaping requires considerable expertise and experimentation. For example, directly minimizing the distance between the center of the ring and the bottom of the peg leads to an unsuccessful policy that smashes the ring against the peg, as in Video 1b.  

<table class="col-2">
  <tr>
    <td style="text-align:center;">
			<img src="http://bair.berkeley.edu/static/blog/reverse_curriculum/ring_fail_cross.gif"
			alt="ring_fail_cross">
		</td>
    <td style="text-align:center;">
			<img src="http://bair.berkeley.edu/static/blog/reverse_curriculum/ring_shapping_cross.gif"
			alt="ring_shapping_cross">
		</td>
  </tr>
  <tr>
    <td><p>
      <i>Video 1a. A randomly initialized policy is unable to reach the goal from most start positions, hence being unable to learn.</i>
		</p></td>
    <td><p>
      <i>Video 1b. Shaping the reward with a penalty on the distance from the ring center to the peg bottom yields an undesired behavior.
</i>
		</p></td>
  </tr>
</table>

<!--more-->

## Curriculum instead of Reward Shaping

We would like to train an agent to reach the goal from any starting position, without requiring an expert to shape the reward. Clearly, not all starting positions are equally difficult. In particular, even a random agent that is placed near to the goal will be able to reach the goal some of the time, receive a reward, and hence start learning! This acquired knowledge can then be bootstrapped to solve the task starting from further away from the goal. By **choosing the ordering of the starting positions that we use in training**, we can exploit this underlying structure of the problem and improve learning efficiency. A key advantage of this technique is that the **reward function is not modified**, and optimizing the sparse reward directly is less prone to yielding undesired behaviors. Ordering a set of related tasks to be learned is referred to as **curriculum learning**, and a central question for us is how to choose this task ordering. Our method, which we explain in more detail below, uses the performance of the learning agent to automatically generate a curriculum of tasks which start from the goal and expand outwards.

### Reverse Curriculum Intuition
In *goal-oriented* tasks the aim is to reach a desired configuration from any start state. For example, in the ring-on-peg task introduced above, we desire to place the ring on the peg starting from any configuration. From most start positions, the random exploration of our initial policy never reaches the goal and hence perceives no reward. Nevertheless, it can be seen in Video 2a how a random policy is likely to reach the bottom of the peg if it is initialized from a nearby position. Then, once we have learned how to reach the goal from around the goal, learning from further away is easier since the agent already knows how to proceed if exploratory actions drive its state nearby the goal, as in Video 2b. Eventually, the agent successfully learns to reach  the goal from a wide range of starting positions, as in Video 2c.

<p style="text-align:center;">
<img height = "200" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/ring_curr1.gif" title="ring_curr1">
<img height = "200" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/ring_curr2.gif" title="ring_curr2">
<img height = "200" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/ring_curr3.gif" title="ring_curr3">
<br>
<i>
Video 2a-c. Our method strives to learn first starting nearby the goal, and then progressively expands in reverse the positions from where it starts.
</i>
</p>

This method of learning in reverse, or growing outwards from the goal, draws inspiration from Dynamic Programming methods, where the solutions to easier sub-problems are used to compute the solution to harder problems.

### Starts of Intermediate Difficulty (SoID)

To implement this reverse curriculum, we need to ensure that this outwards expansion happens at the right pace for the learning agent. In other words, we want to mathematically describe a set of starts that tracks the current agent performance and provides a good learning signal to our Reinforcement Learning algorithm. In particular we focus on Policy Gradient algorithms, which improve a parameterized policy by taking steps in the direction of an estimated gradient of the total expected reward . This gradient estimation is usually a variation of the original REINFORCE (Williams 1992), which is estimated by collecting $N$ on-policy trajectories $$\{\tau^i\}_{i=1..N}$$ starting from states $$\{s^i_0\}_{i=1..N}$$.

$$\nabla_\theta\eta=\frac{1}{N}\sum_{i=1}^N\nabla_\theta\log\pi_\theta(\tau^i)[R(\tau^i, s^i_0)-R_\theta(s^i_0)] \quad (1)$$

In *goal-oriented* tasks, the trajectory reward $R(\tau^i, s^i_0)$ is binary, indicating whether the policy reached the goal. Therefore, the usual baseline $R_\theta(s^i_0)$ estimates the probability of reaching the goal if the current policy $\pi_\theta$ is executed starting from $s^i_0$. Hence, we see from Eq. (1) that the terms of the sum corresponding to trajectories collected from starts $s^i_0$ that have success probability 0 or 1 will vanish. These are “wasted” trajectories as they do not contribute to the estimation of the gradient -- they are either too hard or too easy. A similar analysis was already introduced in our prior work on multi-task RL (Held et al. 2017). In this case, to avoid training from starts from where our current policy either never gets to the goal or already masters it, we introduce the concept of “Start of Intermediate Difficulty” (SoID), which are start states $s_0$ that satisfy:

$$s_0:R_{min} < R_\theta(s_0) < R_{max} \quad (2)$$

The values of $$R_{min}$$ and $$R_{max}$$ have the straightforward interpretation of minimum success probability acceptable for training from that start and maximum success probability above which we prefer to focus on training from other starts. In all our experiments we used 10% and 90%.


## Automatic Generation of the Reverse Curriculum

From the above intuition and derivation, we would like to train our policy with trajectories starting from SoID states. Unfortunately, finding all starts that exactly satisfy Eq. (2) at every policy update is intractable, and hence we introduce an efficient approximation to automatically generate this reverse curriculum: we sample states nearby the starts that were estimated to be SoID during the previous iteration. To do that, we propose a way to *filter out non-SoID* starts using the trajectories collected during the last training iteration and then *sample nearby* states. The full algorithm is illustrated in Video 3 and details are given below.

{% include youtubePlayer.html id="GTwocZfJdWU" %}
<p style="text-align:center;">
<i>
Video 3. Animation illustrating the main steps of our algorithm, and how it automatically produces a curriculum adapted to the current agent performance.
</i>
</p>

### Filtering out Non-SoID

At every policy gradient training iteration, we collect $$N$$ trajectories from some start positions $$\{s^i_0\}_{i=1..N}$$. For most start states, we collect at least three trajectories starting from there, and hence we can compute a Monte Carlo estimate of the success probability of our policy from those starts $$R_\theta(s^i_0)$$. For every $$s^i_0$$ that the estimate is not within the fixed bounds $$R_{min}$$ and $$R_{max}$$, we discard this start so that we don’t train from it during the next iteration. Starts that were SoID for a previous policy might not be SoID for the current policy because they are now mastered or because the updated policy got worse at them, so it is important to keep filtering out the non-SoID starts to maintain a curriculum adapted to the current agent performance.

### Sampling Nearby

After filtering the non-SoID we need to obtain new SoIDs to keep expanding the starts from where we train. We do that by sampling states nearby the remaining SoID because those have a similar level of difficulty for the current policy -- and hence might also be SoID. But what is a good way to sample nearby a certain state $$s^i_0$$? We propose to take random exploratory actions from that $$s^i_0$$, and record the visited states. This technique is preferable to applying noise in state space directly because that might yield states that are not even feasible or that cannot be reached by executing actions from the original $$s^i_0$$.

### Assumptions

To initialize the algorithm, we need to **seed it with one start at the goal** $$s^g$$ and then run brownian motion from it, train from the collected starts, filter out the non-SoID and iterate. This is usually easy to provide when specifying the problem, and is a milder assumption than requiring a full demonstration of how to get to that point.

Our algorithm exploits the capability of **choosing the start distribution** from where the collected trajectories start. This is the case in many systems, like all simulated ones.


## Application to Robotics

*Navigation* to a fixed goal and *fine-grained manipulation* to a desired configuration are two examples of goal-oriented robotics tasks. We analyze how the proposed algorithm automatically generates a reverse curriculum for the following tasks: Point-mass Maze (Fig. Xa), Ant Maze (Fig. Xb), Ring-on-Peg (Fig. Xc) and Key insertion (Fig. Xd).

<p style="text-align:center;">
<img height = "175" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/point_mass.png" title="point_mass">
<img height = "175" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/ant_maze.png" title="ant_maze">
<img height = "175" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/ring_on_peg.png" title="ring_on_peg">
<img height = "175" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/key_insertion.png" title="key_insertion">
<br>
<i>
Fig Xa-d. Tasks where we illustrate the performance of our method (from left to right): Point-mass Maze, Ant Maze, Ring-on-Peg, Key insertion.
</i>
</p>

### Point-mass Maze

In this task we want to learn how to reach the end of the red area in the upper-right corner of Fig Xa from any start point within the maze. We see in Fig. XX that a randomly initialized policy -- as we have at iteration $$i=1$$, has a success probability of zero from everywhere but around the goal. The second row of Fig. XX shows how our algorithm proposes start positions nearby the goal at $$i=1$$. We see in the subsequent columns that the starts generated by our method keep tracking the area where the training policy succeeds sometimes but not always, hence giving a good learning signal for any policy gradient learning method.

<p style="text-align:center;">
<img width = "90%" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/performance.png" title="performance">
<br>
<i>
Fig XX. Snapshots of the policy performance and the starts generated by our reverse curriculum (replay buffer not depicted for clarity), always tracking the regions at an intermediate level of difficulty.
</i>
</p>

To avoid forgetting how to reach the goal from some areas, we keep a replay buffer of all starts that were SoID for any previous policy. In every training iteration we sample a fraction of the trajectories starting from states in this replay.

### Ant Maze Navigation

In robotics, a complex coordinated motion is often required to reach the desired configuration. For example, a quadruped like the one in Fig. Xb needs to know how to coordinate all its torques to move and advance towards the goal. As seen in a final policy reported in Video Y, our algorithm is able to learn this behavior even when only a success/failure reward is provided when reaching the goal! The reward function was not modified to include any distance-to-goal, Center of Mass speed, or exploration bonus.

<p style="text-align:center;">
<img height="250" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/ant_maze.gif" title="ant_maze_gif">
<br>
<i>
Video Y: Emergence of complex coordination and use of environment contacts when training with our reverse curriculum method - even under sparse rewards.
</i>
</p>

### Fine-grained Manipulation

Our method can also tackle complex robotic manipulation problems like the ones depicted in Fig Xc and Xd. Both task have a seven Degrees of Freedom arm and have complex contact constraints. The first task requires the robot to insert a ring down to the bottom of the peg, and the second task seeks to insert a key in a lock, rotate 90 degrees clockwise, insert it further and rotate 90 degrees counterclockwise. In both cases, a reward is only granted when the desired end configuration is reached. State-of-the-art RL algorithms without curriculum are unable to learn how to solve the task, but with our reverse curriculum generation we can obtain a successful policy from a wide range of start positions, as observed in Videos Z and W.

<p style="text-align:center;">
<img width = "30%" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/ring_success.gif" title="ring_success">
<img width = "30%" src="http://bair.berkeley.edu/static/blog/reverse_curriculum/key_success.gif" title="key_success">
<br>
<i>
Video Z. (left) and Video W. (right): Final policies obtained with our reverse curriculum approach in the Ring-on-Peg and the Key Insertion tasks. The agent succeeds from a wide range of initial positions and is able to leverage the contacts to guide itself.
</i>
</p>


## Conclusions and Future Directions

Recently RL methods have been moving away from the single-task paradigm to tackle sets of tasks. This is an effort to get closer to real-world scenarios, where every time a tasks needs to be executed there are always variations in the starting configuration, goal or other parameters. Therefore it is of utmost importance to advance the field of curriculum learning to exploit the underlying structure of these sets of tasks. Our Reverse Curriculum strategy is a step in this direction, yielding impressive results in locomotion and complex manipulation tasks that cannot be solved without a curriculum.

Furthermore, it can be observed in the videos of our final policy for the manipulation tasks that the agent has learned to exploit the contacts in the environment instead of avoiding them. Therefore, the learning based aspect of the presented method has a great potential to tackle problems that classical motion planning algorithms could struggle with, such as environments with non-rigid objects or with uncertainties in the task geometric parameters. We also leave as future work to combine our curriculum-generation approach with domain randomization methods (Tobin et al. 2017) to obtain policies that are transferable to the real world.

If you want to learn more, check out our paper published in the Conference on Robot Learning:

*Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, Pieter Abbeel. [Reverse Curriculum Generation for Reinforcement Learning][1]. In CoRL 2017.*

------
We would like to thank the co-authors of this work, who also provided very valuable feedback for this blog post: David Held, Markus Wulfmeier, Michael R. Zhang and Pieter Abbeel.


## Bibliography:

### References Cited in this Blog Post:
Andrew Ng, Daisha Harada. 1999. “Policy Invariance under Reward Transformations Theory and Application to Reward Shaping.”

Held, David, Xinyang Geng, Carlos Florensa, and Pieter Abbeel. 2017. “Automatic Goal Generation for Reinforcement Learning Agents.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1705.06366.

Levine, Sergey, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. 2016. “End-to-End Training of Deep Visuomotor Policies.” Journal of Machine Learning Research: JMLR 17 (39):1–40.

Lillicrap, Timothy P., Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. 2015. “Continuous Control with Deep Reinforcement Learning.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1509.02971.

Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, et al. 2015. “Human-Level Control through Deep Reinforcement Learning.” Nature 518 (7540):529–33.

Popov, Ivaylo, Nicolas Heess, Timothy Lillicrap, Roland Hafner, Gabriel Barth-Maron, Matej Vecerik, Thomas Lampe, Yuval Tassa, Tom Erez, and Martin Riedmiller. 2017. “Data-Efficient Deep Reinforcement Learning for Dexterous Manipulation.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1704.03073.

Schulman, John, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. 2015. “High-Dimensional Continuous Control Using Generalized Advantage Estimation.” In . http://arxiv.org/abs/1506.02438.

Tobin, Josh, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. 2017. “Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.” In . http://arxiv.org/abs/1703.06907.

Williams, Ronald J. 1992. “Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.” Machine Learning 8 (3-4). Kluwer Academic Publishers:229–56.

### Other References:
A. Graves, M. G. Bellemare, J. Menick, R. Munos, and K. Kavukcuoglu. Automated Curriculum Learning for Neural Networks. arXiv preprint, arXiv:1704.03003, 2017.

L. Jiang, D. Meng, Q. Zhao, S. Shan, and A. G. Hauptmann. Self-paced curriculum learning. In AAAI, volume 2, page 6, 2015.

M. Asada, S. Noda, S. Tawaratsumida, and K. Hosoda. Purposive behavior acquisition for a real robot by Vision-Based reinforcement learning. Machine Learning, 1996.

A. Karpathy and M. Van De Panne. Curriculum learning for motor skills. In Canadian Conference on Artificial Intelligence, pages 325–330. Springer, 2012.

J. Schmidhuber. POWER PLAY : Training an Increasingly General Problem Solver by Continually Searching for the Simplest Still Unsolvable Problem. Frontiers in Psychology, 2013.

A. Baranes and P.-Y. Oudeyer. Active learning of inverse models with intrinsically motivated goal exploration in robots. Robotics and Autonomous Systems, 61(1), 2013.

S. Sharma and B. Ravindran. Online Multi-Task Learning Using Biased Sampling. arXiv preprint arXiv: 1702.06053, 2017. 9

S. Sukhbaatar, I. Kostrikov, A. Szlam, and R. Fergus. Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play. arXiv preprint, arXiv: 1703.05407, 2017.

A. Rajeswaran, K. Lowrey, E. Todorov, and S. Kakade. Towards generalization and simplicity in continuous control. arXiv preprint, arXiv:1703.02660, 2017.

. Tedrake, I. R. Manchester, M. Tobenkin, and J. W. Roberts. Lqr-trees: Feedback motion planning via sums-of-squares verification. The International Journal of Robotics Research, 29 (8):1038–1052, 2010.


J. A. Bagnell, S. Kakade, A. Y. Ng, and J. Schneider. Policy search by dynamic programming. Advances in Neural Information Processing Systems, 16:79, 2003.

S. Kakade and J. Langford. Approximately Optimal Approximate Reinforcement Learning. International Conference in Machine Learning, 2002.

E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012.

J. Kuffner and S. LaValle. RRT-connect: An efficient approach to single-query path planning. In IEEE International Conference on Robotics and Automation, volume 2, pages 995–1001. IEEE, 2000.

[1]:http://proceedings.mlr.press/v78/florensa17a/florensa17a.pdf
