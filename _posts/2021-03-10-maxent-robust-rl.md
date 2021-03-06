---
layout:             post
title:              "Maximum Entropy RL (Provably) Solves Some Robust RL Problems"
date:               2021-03-10  9:00:00
author:             Ben Eysenbach
img:                assets/maxent-robust-rl/peg_maxent_screenshot.png
excerpt_separator:  <!--more-->
visible:            False
show_comments:      False
---

<!-- twitter -->
<meta name="twitter:title" content="Maximum Entropy RL (Provably) Solves Some Robust RL Problems">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/maxent-robust-rl/peg_maxent_screenshot.png">

<meta name="keywords" content="reinforcement, learning">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Ben Eysenbach">

Nearly all real-world applications of reinforcement learning involve some degree of shift between the training environment and the testing environment. However, prior work has observed that even small shifts in the environment cause most RL algorithms to perform markedly worse [Jason's domain randomization paper, EP Opt]. As we aim to scale reinforcement learning algorithms and apply them in the real world, it is increasingly important to learn policies that are robust to changes in the environment.

Broadly, prior approaches to handling distribution shift in RL aim to maximize performance in either the average case or the worst case. Methods such as domain randomization train a policy on a distribution of environments, and optimize the average performance of the policy on these environments. While this approach has been successfully applied to a number of areas [self-driving, locomotion, manipulation, video games], its success rests critically on the design of the distribution of environments [automatic domain randomization]. Moreover, policies that do well on average are not guaranteed to get high reward on every environment. The policy that gets the highest reward on average might get very low reward on a small fraction of environments. The second set of approaches, typically referred to as robust RL, focus on the worst-case scenarios. The aim is to find a policy that gets high reward on every environment within some set. Robust RL can equivalently be viewed as a two-player game between the policy and an environment adversary [Drew's CoRL talk]. The policy tries to get high reward, while the environment adversary tries to tweak the dynamics and reward function of the environment so that the policy gets lower reward. One important property of the robust approach is that, unlike domain randomization, it is invariant to the ratio of easy and hard tasks. Whereas robust RL always evaluates a policy on the most challenging tasks, domain randomization will predict that the policy is better if it is evaluated on a distribution of environments with more easy tasks.

<!--more-->

Prior work has suggested a number of algorithms for solving robust RL problems. Generally, these algorithms all follow the same recipe: take an existing RL algorithm and add some additional machinery on top to make it robust. For example, robust value iteration [Bagnell, Nilim] uses Q-learning as the base RL algorithm, and modifies the Bellman update by solving a convex optimization problem in the inner loop of each Bellman backup. Similarly, [Pinto '17] uses TRPO as the base RL algorithm and periodically updates the environment based on the behavior of the current policy. These prior approaches are often difficult to implement and, even once implemented correctly, they requiring tuning of many additional hyperparameters. Might there be a simpler approach, an approach that does not require additional hyperparameters and additional lines of code to debug?

To answer this question, we are going to focus on a type of RL algorithm known as maximum entropy RL, or MaxEnt RL for short [Ziebart, Todorov, Toussaint, Haarnoja, Theaodorou, Kappen]. MaxEnt RL is a slight variant of standard RL aims to learn a policy that gets high reward while acting as randomly as possible (formally, it maximizes the entropy of the policy). Some prior work has observed empirically that maximum entropy (MaxEnt) RL algorithms appear to be robust to some disturbances the environment [Learning to walk via deep reinforcement learning, Svqn: Sequential variational soft q-learning networks]. To the best of our knowledge, no prior work has proven the folklore theorem that MaxEnt RL is robust to environmental disturbances.

In a recent paper, we prove that every MaxEnt RL problem corresponds to maximizing a lower bound on a robust RL problem. Thus, when you run MaxEnt RL, you are implicitly solving a robust RL problem. To the best of our knowledge, this is the first proof of that folklore theorem, and provides a theoretically-justified explanation for the empirical robustness of MaxEnt RL. In the rest of this post, we'll provide some intuition into why MaxEnt RL should be robust and what sort of perturbations MaxEnt RL is robust to. We'll also show some experiments demonstrating the robustness of MaxEnt RL.


<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/maxent-robust-rl/pusher_empty_standard.gif" width="45%">
<img src="https://bair.berkeley.edu/static/blog/maxent-robust-rl/pusher_empty_maxent.gif" width="45%">
<br>
</p>

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/maxent-robust-rl/pusher_obstacle_standard.gif" width="45%">
<img src="https://bair.berkeley.edu/static/blog/maxent-robust-rl/pusher_obstacle_maxent.gif" width="45%">
<br>
</p>

Intuition: So, why would we expect MaxEnt RL to be robust to disturbances in the environment? Recall that MaxEnt RL trains policies to not only maximize reward, but to do so while acting as randomly as possible. In essence, the policy itself is injecting as much noise as possible into the environment, so it gets to "practice" recovering from disturbances. Thus, if the change in dynamics appears like just a disturbance in the original environment, our policy has already been trained on such data. Another way of viewing MaxEnt RL is as learning many different ways of solving the task. [Kappen] For example, let's look at the task shown in videos above: we want the robot to push the white object to the green region. The top two videos show that standard RL always takes the shortest path to the goal, whereas MaxEnt RL takes many different paths to the goal. Now, let's imagine that we add an obstacle (red blocks) to the environment that wasn't included during training. As shown in the videos in the bottom row, the policy learned by standard RL almost always collides with the obstacle, rarely reaching the goal. In contrast, the MaxEnt RL policy often chooses routes around the obstacle, continuing to reach the goal for a large fraction of trials.


Theory:
We now formally describe the technical results from the paper. The aim here is not to provide a full proof (see the paper Appendix for that), but instead to build some intuition for what the technical results say. Our main result is that, when you apply MaxEnt RL with some reward function and some dynamics, you are actually maximizing a lower bound on the robust RL objective. To explain this result, we must first define the MaxEnt RL objective: $J_{MaxEnt}(\pi, p, r)$ is the entropy-regularized cumulative return of policy $\pi$ when evaluated using dynamics $p(s' | s, a)$ and reward function $r$. While we will train the policy using one dynamics $p$, we will evaluate the policy on a different dynamics, $\tilde{p}$, chosen by the adversary. We can now formally state our main result as follows:

....

TODO
