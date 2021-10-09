---
layout:             post
title:              "Which Mutual Information Representation Learning Objectives are Sufficient for Control?"
date:               2021-10-11  9:00:00
author:             <a href="https://katerakelly.github.io/">Kate Rakelly</a>
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
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/mi_analysis/image1.png">

<meta name="keywords" content="reinforcement learning, representation learning">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Kate Rakelly">

<!--
The actual text for the post content appears below.  Text will appear on the
homepage, i.e., https://bair.berkeley.edu/blog/ but we only show part of the
posts on the homepage. The rest is accessed via clicking 'Continue'. This is
enforced with the `more` excerpt separator.
-->
Processing raw sensory inputs is crucial for applying deep RL algorithms to real-world problems.
For example, autonomous vehicles must make decisions about how to drive safely given information flowing from cameras, radar, and microphones about the conditions of the road, traffic signals, and other cars and pedestrians.
However, direct “end-to-end” RL that maps sensor data to actions (Figure 1, top) can be very difficult because the inputs are high-dimensional, noisy, and contain redundant information.
Instead, the challenge can be broken down into two problems (Figure 1, bottom): (1) extract a representation of the sensory inputs that retains only the relevant information, and (2) perform RL with these representations of the inputs as the system state.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/mi_sufficiency_analysis/overview.png" width="50%">
<br>
<i><b>Figure 1. </b>State representation learning for RL.</i>
</p>

A wide variety of algorithms have been proposed to tackle the first step of representation learning in an unsupervised fashion [CITE generative models].
Recently, contrastive learning methods (such as [CPC][1], [ATC][2], and [SGI][3]) have proven highly effective on RL benchmarks such as Atari.
These methods learn lossy representations that discard some parts of the input, prompting a natural question -- are the representations learned via MI-based objectives guaranteed to be sufficient for control?
In other words, do they contain all the information necessary to solve downstream control problems, or might they throw out some information that may be required?
For example, in the self-driving car scenario, if the representation discards the locations of other cars, the vehicle would be unable to respond to other drivers’ actions.
Surprisingly, we find that some widely used objectives are not sufficient, and in fact **do** discard information that may be needed for downstream tasks.


<!--more-->

## Defining the Sufficiency of a State Representation
As introduced above, a state representation is a function of the raw sensory inputs that discards irrelevant and redundant information.
Formally, we define a state representation $\phi_Z$ as a stochastic mapping from the original state space $\mathcal{S}$ (the raw inputs from all the car’s sensors) to a representation space $\mathcal{Z}$: $p(Z | S=s)$.
In our analysis, we assume that the original state $\mathcal{S}$ is Markovian, so each state representation $\mathcal{Z}$ is a function of only the current state.
We depict the representation learning problem as a graphical model in Figure 2.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/mi_sufficiency_analysis/graphical_model.png" width="30%">
<br>
<i><b>Figure 2. </b>The representation learning problem in RL as a graphical model.</i>
</p>

We will say that a representation is sufficient if it is guaranteed that an RL algorithm using that representation can learn and represent the optimal policy.
Since we are interested in unsupervised representation learning methods that don’t have access to a task reward, to call a representation sufficient we require that it is sufficient for **all** optimal policies for all possible reward functions in the given MDP.
To define sufficiency formally, we make use of a result from [Li et al. 2006][4].
This paper proves that if a state representation is capable of representing the optimal $Q$-function, then $Q$-learning is guaranteed to converge when run with that representation as the state input (if you’re interested, see Theorem 4 in that paper).
We’ll use the ability to represent the optimal $Q$-function as our definition of sufficiency for a state representation.

## Analyzing Representations learned via MI Maximization
Now that we’ve established how we will evaluate representations, let’s turn to the methods of learning them.
As mentioned above, we aim to study the popular class of contrastive learning methods that maximize MI-based objectives.
To simplify the analysis, we analyze representation learning in isolation from the other aspects of RL by assuming the existence of an offline dataset on which to perform representation learning.
This paradigm of offline representation learning followed by online RL is becoming increasingly popular, particularly in applications such as robotics where collecting data is onerous ([Zhan et al. 2020][5], [Kipf et al. 2020][6]).
Our question is therefore whether the objective is sufficient on its own, not as an auxiliary objective for RL.
We assume the dataset has full support on the state space, which can be guaranteed by an epsilon-greedy exploration policy, for example.
An objective may have more than one maximizing representation, so we call a representation learning *objective* sufficient if *all* the representations that maximize that objective are sufficient.
In our paper, we analyze three representative objectives from the literature in terms of sufficiency, two of which we will discuss here.

### Representations Learned by Maximizing “Forward Information”
We begin with an objective that seems likely to retain a great deal of state information in the representation.
It is closely related to learning a forward dynamics model in latent representation space, and to methods proposed in prior works ([Nachum et al. 2018][7], [Shu et al. 2020][8], [Schwarzer et al. 2021][9]): $J_{fwd} = I(Z_{t+1}; Z_t, A_t)$.
Intuitively, this objective seeks a representation in which the current state and action are maximally informative of the representation of the next state.
Therefore, everything predictable in the original state $\mathcal{S}$ should be preserved in $\mathcal{Z}$, since this would maximize the MI.
Formalizing this intuition, we are able to prove that all representations learned via this objective are guaranteed to be sufficient (see the proof of Proposition 1 in the paper).

It’s worth noting here that, since we proved sufficiency for all reward functions, representations that maximize $J_{fwd}$ are actually capable of representing **any** optimal $Q$-function that was possible in the original MDP.
This begs the question: what information is the representation $\phi_Z$ actually able to discard?
The answer is that $\phi_Z$ can discard time-independent information in $S$.
For example, if a light flashes randomly at each timestep, $\phi_Z$ would be free to ignore the light.
Note that $\phi_Z$ is still sufficient even if the reward function depends on the flashing light because since the light cannot be predicted, any policy is as good as any other.
Still, most signals in realistic scenarios, including distracting ones that we may like our agents to ignore, are temporally correlated, and therefore would not be discarded by $J_{fwd}$.
Is there another objective that can learn sufficient but lossier representations?

### Representations Learned by Maximizing “Inverse Information”
Next, we consider what we term an “inverse information” objective: $I_{inv} = I(Z_{t+k}; A_t | Z_t)$.
One way to maximize this objective is by learning an inverse dynamics model -- predicting the action given the current and next state -- and many prior works have employed a version of this objective ([Agrawal et al. 2016][10], [Gregor et al. 2016][11], [Zhang et al. 2018][12] to name a few).
Intuitively, this objective is appealing because it preserves all the state information that the agent can influence with its actions.
It therefore may seem like a good candidate for a sufficient objective that discards more information than $J_{fwd}$.
However, we can actually construct a realistic scenario in which a representation that maximizes this objective is not sufficient.

For example, consider the MDP shown on the left side of Figure 3 in which an autonomous vehicle is approaching a traffic light. The agent has two actions available, stop or go. The reward for following traffic rules depends on the color of the stoplight, and is denoted by a red X (low reward) and green check mark (high reward). On the right side of the figure, we show a state representation that also maximizes $J_{inv}$ but is not sufficient to represent the optimal policy. In this representation, the color of the stoplight is not represented in the two states on the left, allowing them to be aliased and represented as a single state. Intuitively, $J_{inv}$ is maximized by this representation because the agent has no control over the stoplight, so representing it does not increase MI.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/mi_sufficiency_analysis/inv_counterexample.png" width="50%">
<br>
<i><b>Figure 3. </b>Counterexample proving the insufficiency of $J_{inv}$.</i>
</p>

Assuming deterministic dynamics and a uniform policy, we can show this computationally. In Figure 4, we plot the values of $J_{fwd}$ and $J_{inv}$ for different state representations, ordered on the x-axis by the value of $I(Z; S)$, or how much information is retained by the representation (the representation that aliases all states is the furthest left, while the identity representation is the furthest right and plotted with a star). The representation with aliased states depicted on the right side of Figure 3 is plotted with a diamond. This representation achieves the same value of $J_{inv}$ as the original state representation, but value iteration run with this representation fails to learn the optimal policy. The issue appears to be that practical reward functions can depend on elements outside the agent’s control. Intuitively, if the representation fails to capture the stoplight, but the reward depends on it, it seems that we may be able to resolve the issue by requiring that the representation also be capable of predicting the reward at that state. However, this is still not enough to guarantee sufficiency - the representation on the right side of Figure 3 is still a counterexample since the aliased states have the same reward. The crux of the problem is that representing the action that connects two states is not enough to be able to choose the best action. Still, while $J_{inv}$ is insufficient in the general case, it would be revealing to characterize the set of MDPs for which $J_{inv}$ can be proven to be sufficient. We see this as an interesting future direction.


<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/mi_sufficiency_analysis/inv_counterexample_plot.png" width="50%">
<br>
<i><b>Figure 4. </b>Values of $J_{fwd}$ compared to $J_{inv}$ for different state representations.</i>
</p>

### Representations Learned by Maximizing “State Information”
The final objective we consider was proposed in [Oord et al. 2018][1], and resembles $J_{fwd}$ but omitting the action: $J_{state} = I(Z_t; Z_{t+1})$. Does omitting the action from the MI objective impact its sufficiency? It turns out the answer is yes. The intuition is that maximizing this objective can yield insufficient representations when the variation in the next state depends entirely on the action. For example, consider a car driving at dusk. If the reward depends on turning on the headlights when it gets dark (an action), a state representation maximizing $J_{state}$ could fail to capture the state of the headlights. For brevity, we’ll leave the discussion of this objective here -- see our paper for the full analysis.

## Can Sufficiency Matter in Deep RL?
To understand whether the sufficiency of state representations can matter in practice, we perform simple proof-of-concept experiments with deep RL agents and image observations. To separate representation learning from RL, we first optimize each representation learning objective on a dataset of offline data, (similar to the protocol in [Stooke et al. 2020][2]). We collect the fixed datasets using a random policy, which is sufficient to cover the state space in our environments. We then freeze the weights of the state encoder learned in the first phase and train RL agents with the representation as state input.

We experiment with a simple video game MDP that has a similar characteristic to the self-driving car example described earlier. In this game called *catcher*, from the [PyGame suite][16], the agent controls a paddle that it can move back and forth to catch fruit that falls from the top of the screen (see Figure 5, left). A positive reward is given when the fruit is caught and a negative reward when the fruit is not caught. The episode terminates after one piece of fruit falls. Analogous to the self-driving example, the agent does not control the position of the fruit, and so a representation that maximizes $I_{inv}$ might discard that information. However, representing the fruit is crucial to obtaining reward, since the agent must move the paddle underneath the fruit to catch it. We learn representations with $I_{inv}$ and $I_{fwd}$, optimizing $I_{fwd}$ with noise contrastive estimation [(NCE)][12], and $I_{inv}$ by training an inverse model via maximum likelihood. To select the most compressed representation from among those that maximize each objective, we apply an information bottleneck of the form $\min I(Z; S)$. We also compare to running RL from scratch with the image inputs, which we call ``end-to-end.” For the RL algorithm, we use the [Soft Actor-Critic][14] algorithm.

We observe in Figure 5 (middle) that indeed the representation trained to maximize $I_{inv}$ results in RL agents that converge slower and to a lower asymptotic expected return. To better understand what information the representation contains, we then attempt to learn a neural network decoder from the learned representation to the position of the falling fruit. We report the mean error achieved by each representation in Figure 5, right. The representation learned by $I_{inv}$ incurs a high error, indicating that the fruit is not precisely captured by the representation, while the representation learned by $I_{fwd}$ incurs low error.

<!-- TODO: how can I put three images next to each other here? -->
<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/mi_sufficiency_analysis/catcher_game.png" width="33%">
<img src="https://bair.berkeley.edu/static/blog/mi_sufficiency_analysis/catcher_game.png" width="33%">
<img src="https://bair.berkeley.edu/static/blog/mi_sufficiency_analysis/catcher_game.png" width="33%">
<br>
<i><b>Figure 5. </b>(left) Illustration of the *catcher* game. (middle) Performance of RL agents trained with different state representations. (right) Accuracy of reconstructing ground truth state elements from state representations.</i>
</p>

### Increasing observation complexity with visual distractors
To make the representation learning problem more challenging, we repeat this experiment with visual distractors added to the agent’s observations. We randomly generate images of 10 circles of different colors and replace the background of the game with these images (see Figure 6, left for example observations). As in the previous experiment, we plot the performance of an RL agent trained with the frozen representation as input (Figure 6, middle), as well as the error of decoding true state elements from the representation (Figure 6, right). The difference in performance between sufficient ($I_{fwd}$) and insufficient ($I_{inv}$) objectives is even more pronounced in this setting than in the plain background setting. With more information present in the observation in the form of the distractors, insufficient objectives that do not optimize for representing all the required state information may be "distracted" by representing the background objects instead, resulting in low performance. In this more challenging case, end-to-end RL from images fails to make any progress on the task, demonstrating the difficulty of end-to-end RL.

<!-- TODO: how can I put three images next to each other here? -->
<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/mi_sufficiency_analysis/distractor_observation.png" width="50%">
<br>
<i><b>Figure 6. </b>(left) Example agent observations with distractors. (middle) Performance of RL agents trained with different state representations. (right) Accuracy of reconstructing ground truth state elements from state representations.</i>
</p>

## Conclusion
In light of these results, we think it’s important to understand both the characteristics of the representation learning objective and the set of tasks that may be learned via RL with the state representation, in order to ensure that the representation learning objective preserves important state elements. $J_{fwd}$ is sufficient for general MDPs, but lacks a notion of ``task-relevance” as it must be equally predictive of all predictable elements in the state, and so may be a poor choice for some problems. On the other hand, $J_{inv}$ is capable of discarding more information, but is not sufficient in general. These results lead to further questions: What are the characteristics of MDPs for which $J_{inv}$ is sufficient? And is it possible to construct an objective that is sufficient in general but has a maximizing representation that contains less information than the smallest representation that maximizes $J_{fwd}$?

Further, extending the proposed framework to partially observed problems would be more reflective of realistic applications. In this setting, analyzing generative models such as VAEs in terms of sufficiency is an interesting problem. Prior work has shown that maximizing the ELBO alone cannot control the content of the learned representation (e.g., [Alemi et al. 2018][15]). We conjecture that the zero-distortion maximizer of the ELBO would be sufficient, while other solutions need not be. Overall, we hope that our proposed framework can drive research in designing better algorithms for unsupervised representation learning for RL.


<hr>

<i>This post is based on the paper “Which Mutual Information Representation Learning Objectives are Sufficient for Control?”, to be presented at Neurips 2021. Thank you to Sergey Levine and Abhishek Gupta for their valuable feedback on this blog post.</i>

[1]:https://arxiv.org/abs/1807.03748
[2]:https://arxiv.org/abs/2009.08319
[3]:https://arxiv.org/abs/2106.04799
[4]:http://rbr.cs.umass.edu/aimath06/proceedings/P21.pdf
[5]:https://arxiv.org/abs/2012.07975
[6]:https://arxiv.org/abs/1911.12247
[7]:https://arxiv.org/abs/1810.01257
[8]:https://arxiv.org/abs/2003.01086
[9]:https://arxiv.org/abs/2007.05929
[10]:https://arxiv.org/abs/1606.07419
[11]:https://arxiv.org/abs/1611.07507
[12]:https://arxiv.org/abs/1804.10689
[13]:https://proceedings.mlr.press/v9/gutmann10a.html
[14]:https://arxiv.org/abs/1801.01290
[15]:https://arxiv.org/abs/1711.00464
[16]:https://pygame.org
