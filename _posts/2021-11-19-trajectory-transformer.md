---
layout:             post
title:              "Sequence Modeling Solutions<br> for Reinforcement Learning Problems"
date:               2021-11-19  9:00:00
author:             <a href="https://people.eecs.berkeley.edu/~janner/">Michael Janner</a>
img:                assets/trajectory-transformer/outlines_transformer.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<!-- twitter -->
<meta name="twitter:title" content="Sequence Modeling Solutions for Reinforcement Learning Problems">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/trajectory_transformer/humanoid_padded.png">

<meta name="keywords" content="trajectory, transformer, reinforcement, learning, RL">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Michael Janner">

<title>Sequence Modeling Solutions for Reinforcement Learning Problems</title>


<!-- begin section I: introduction -->

<p style="text-align:center; margin-top:-40px;">
    <br>
    <video width="100%" autoplay playsinline muted>
        <source src="https://bair.berkeley.edu/static/blog/trajectory_transformer/rollout_transformer_compressed.mp4" type="video/mp4">
    </video>
    <video width="100%" autoplay playsinline muted>
        <source src="https://bair.berkeley.edu/static/blog/trajectory_transformer/rollout_single_compressed.mp4" type="video/mp4">
    </video>
    <p width="80%" style="text-align:center; margin-left:10%; margin-right:10%; padding-bottom: -10px;">
        <i style="font-size: 0.9em;">
        Long-horizon predictions of (top) the <b><span style="color:#D62728;">Trajectory Transformer</span></b> compared to those of (bottom) a <b><span style="color:#D62728;">single-step</span></b> dynamics model.
        </i>
    </p>
<br>

<p>
    Modern <a href="https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html">machine</a> <a href="https://arxiv.org/abs/1807.03748">learning</a> <a href="https://www.nature.com/articles/s41586-021-03819-2">success</a> <a href="https://arxiv.org/abs/2002.05709">stories</a> often have one thing in common: they use methods that scale gracefully with ever-increasing amounts of data.
    This is particularly clear from recent advances in sequence modeling, where simply increasing the size of a stable architecture and its training set leads to <a href="https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf">qualitatively</a> <a href="https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">different</a> <a href="https://arxiv.org/abs/2005.14165">capabilities</a>.<sup id="fnref:anderson"><a href="#fn:anderson" class="footnote"><font size="-2">1</font></a></sup>
</p>

<p>
    Meanwhile, the situation in reinforcement learning has proven more complicated.
    While it has been possible to apply reinforcement learning algorithms to <a href="https://journals.sagepub.com/doi/full/10.1177/0278364917710318">large</a>-<a href="https://www.science.org/doi/10.1126/science.aar6404">scale</a> <a href="https://arxiv.org/abs/1912.06680">problems</a>, generally there has been much more friction in doing so.
    In this post, we explore whether we can alleviate these difficulties by tackling the reinforcement learning problem with the toolbox of sequence modeling.
    The end result is a generative model of trajectories that looks like a <a href="https://arxiv.org/abs/1706.03762">large language model</a> and a planning algorithm that looks like <a href="https://kilthub.cmu.edu/articles/journal_contribution/Speech_understanding_systems_summary_of_results_of_the_five-year_research_effort_at_Carnegie-Mellon_University_/6609821/1">beam search</a>.
    Code for the approach can be found <a href="https://github.com/JannerM/trajectory-transformer">here</a>.
</p>

<!--more-->

<h3 id="models">The Trajectory Transformer</h3>

<p>
    The standard framing of reinforcement learning focuses on decomposing a complicated long-horizon problem into smaller, more tractable subproblems, leading to dynamic programming methods like $Q$-learning and an emphasis on Markovian dynamics models.
    However, we can also view reinforcement learning as analogous to a sequence generation problem, with the goal being to produce a sequence of actions that, when enacted in an environment, will yield a sequence of high rewards.
</p>

<p>
    Taking this view to its logical conclusion, we begin by modeling the trajectory data provided to reinforcement learning algorithms with a Transformer architecture, the current tool of choice for natural language modeling.
    We treat these trajectories as unstructured sequences of discretized states, actions, and rewards, and train the Transformer architecture using the standard cross-entropy loss.
    Modeling all trajectory data with a single high-capacity model and scalable training objective, as opposed to separate procedures for dynamics models, policies, and $Q$-functions, allows for a more streamlined approach that removes much of the usual complexity.
</p>

<center>
    <img width="80%" style="padding-top: 20px; padding-bottom: 20px" src="https://bair.berkeley.edu/static/blog/trajectory_transformer/architecture.png">
    <br>
    <p width="80%" style="text-align:center; margin-left:10%; margin-right:10%; padding-bottom: 10px;">
        <i style="font-size: 0.9em;">
        We model the distribution over $N$-dimensional states $\mathbf{s}_t$, $M$-dimensional actions $\mathbf{a}_t$, and scalar rewards $r_t$ using a Transformer architecture.
        </i>
    </p>
</center>

<!-- begin section II: models -->

<h3 id="models">Transformers as dynamics models</h3>

<p>
    In many model-based reinforcement learning methods, compounding prediction errors cause long-horizon rollouts to be too unreliable to use for control, necessitating either <a href="https://arxiv.org/abs/1909.11652">short-horizon planning</a> or <a href="https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.6005&rep=rep1&type=pdf">Dyna-style</a> combinations of <a href="https://arxiv.org/abs/1906.08253">truncated model predictions and value functions</a>.
    In comparison, we find that the Trajectory Transformer is a substantially more accurate long-horizon predictor than conventional single-step dynamics models.
</p>

<center>
    <table>
      <tr>
        <th width="45%" style="border-top: 0px;">
            <img src="https://bair.berkeley.edu/static/blog/trajectory_transformer/error_blog.png" width="100%">
        </th>
        <th width="55%" style="border-top: 0px;">
            <b style="font-size: 0.8em;">Transformer</b>
            <br>
            <img src="https://bair.berkeley.edu/static/blog/trajectory_transformer/outlines_transformer.png" width="100%">
            <b style="font-size: 0.8em;">Single-step</b>
            <br>
            <img src="https://bair.berkeley.edu/static/blog/trajectory_transformer/outlines_single_step.png" width="100%">
            <br>
        </th>
      </tr>
    </table>
    <div style="width: 90%;">
        <p style="text-align:center;">
            <i style="font-size: 0.9em;">Whereas the single-step model suffers from compounding errors that make its long-horizon predictions physically implausible, the Trajectory Transformer's predictions remain visually indistinguishable from <a href="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/outlines_reference.png">rollouts in the reference environment</a>.</i>
        </p>
    </div>
</center>
<br clear="left"/>

<p>
    This result is exciting because planning with learned models is notoriously finicky, with neural network dynamics models often being too inaccurate to benefit from more sophisticated planning routines.
    A higher quality predictive model such as the Trajectory Transformer opens the door for importing effective trajectory optimizers that previously would have only served to <a href="https://arxiv.org/abs/1802.10592">exploit the learned model</a>.
</p>

<p>
    We can also inspect the Trajectory Transformer as if it were a standard language model.
    A common strategy in machine translation, for example, is to <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">visualize the intermediate token weights</a> as a proxy for token dependencies.
    The same visualization applied to here reveals two salient patterns:
</p>

<center>
    <img width="30%" style="padding-top: 10px; padding-right: 60px;" src="https://bair.berkeley.edu/static/blog/trajectory_transformer/markov.png">
    <img width="30%" src="https://bair.berkeley.edu/static/blog/trajectory_transformer/striated.png">
    <br>
    <p width="80%" style="text-align:center; margin-left:10%; margin-right:10%; padding-top: 20px; padding-bottom: 10px;">
        <i style="font-size: 0.9em;">
        Attention patterns of Trajectory Transformer, showing (left) a discovered <b><span style="color:#D62728;">Markovian stratetgy</span></b> and (right) an approach with <b><span style="color:#D62728;">action smoothing</span></b>.
        </i>
    </p>
</center>

<p>
    In the first, state and action predictions depend primarily on the immediately preceding transition, resembling a learned Markov property.
    In the second, state dimension predictions depend most strongly on the corresponding dimensions of all previous states, and action dimensions depend primarily on all prior actions.
    While the second dependency violates the usual intuition of actions being a function of the prior state in behavior-cloned policies, this is reminiscent of the action smoothing used in some <a href="https://arxiv.org/abs/1909.11652">trajectory optimization algorithms</a> to enforce slowly varying control sequences.
</p>


<!-- begin section II: planning -->

<h3 id="planning">Beam search as trajectory optimizer</h3>

<p>
    The simplest model-predictive control routine is composed of three steps: <b><span style="color:#D62728;">(1)</span></b> using a model to search for a sequence of actions that lead to a desired outcome; <b><span style="color:#D62728;">(2)</span></b> enacting the first<sup id="fnref:mpc"><a href="#fn:mpc" class="footnote"><font size="-2">2</font></a></sup> of these actions in the actual environment; and <b><span style="color:#D62728;">(3)</span></b> estimating the new state of the environment to begin step (1) again.
    Once a model has been chosen (or trained), most of the important design decisions lie in the first step of that loop, with differences in action search strategies leading to a wide array of trajectory optimization algorithms.
</p>

<p>
    Continuing with the theme of pulling from the sequence modeling toolkit to tackle reinforcement learning problems, we ask whether the go-to technique for decoding neural language models can also serve as an effective trajectory optimizer.
    This technique, known as <a href="https://kilthub.cmu.edu/articles/journal_contribution/Speech_understanding_systems_summary_of_results_of_the_five-year_research_effort_at_Carnegie-Mellon_University_/6609821/1">beam search</a>, is a pruned breadth-first search algorithm that has found remarkably consistent use since the earliest days of computational linguistics.
    We explore variations of beam search and instantiate its use a model-based planner in three different settings:
</p>

<div>
  <ol>
    <li><p>
        <b><span style="color:#D62728;">Imitation:</span></b> If we use beam search without modification, we sample trajectories that are probable under the distribution of the training data. Enacting the first action in the generated plans gives us a long-horizon model-based variant of imitation learning.
    </p></li>
    <li><p>
       <b><span style="color:#D62728;">Goal-conditioned RL:</span></b> Conditioning the Transformer on <i>future</i> desired context alongside previous states, actions, and rewards yields a goal-reaching method. This works by recontextualizing past data as optimal for some task, in the same spirit as <a href="https://arxiv.org/abs/1707.01495">hindsight relabeling</a>.
    </p>
        <center>
          <img width="28%" src="https://bair.berkeley.edu/static/blog/trajectory_transformer/0.png">
          &nbsp;
          &nbsp;
          <img width="28%" src="https://bair.berkeley.edu/static/blog/trajectory_transformer/1.png">
          &nbsp;
          &nbsp;
          <img width="28%" src="https://bair.berkeley.edu/static/blog/trajectory_transformer/2.png">
          <br>
          <img width="2%" src="https://bair.berkeley.edu/static/blog/trajectory_transformer/rolloutblack-1.png">
          &nbsp;
          Start
          &nbsp;
          &nbsp;
          <img width="2%" src="https://bair.berkeley.edu/static/blog/trajectory_transformer/rolloutblue-1.png">
          &nbsp;
          Goal
        </center>
        <p width="90%" style="text-align:center; padding-top: 20px; padding-bottom: 10px;">
        <i style="font-size: 0.9em;">
            Paths taken by the goal-conditioned beam-search planner in a four-rooms environment.
        </i>
    </p>
    </li>
    <li><p>
        <b><span style="color:#D62728;">Offline RL:</span></b> If we replace transitions' log probabilities with their rewards (their <a href="https://arxiv.org/abs/1805.00909">log probability of optimality</a>), we can use the same beam search framework to optimize for reward-maximizing behavior.
        We find that this simple combination of a trajectory-level sequence model and beam search decoding performs on par with the best prior offline reinforcement learning algorithms <i>without</i> the usual ingredients of standard offline reinforcement learning algorithm: <a href="https://arxiv.org/abs/1911.11361">behavior policy regularization</a> or explicit <a href="https://arxiv.org/abs/2006.04779">pessimism</a> in the case of model-free algorithms, or <a href="https://arxiv.org/abs/2005.05951">ensembles</a> or other <a href="https://arxiv.org/abs/2005.13239">epistemic uncertainty estimators</a> in the case of model-based algorithms. All of these roles are fulfilled by the same Transformer model and fall out for free from maximum likelihood training and beam-search decoding.
    </p></li>
  </ol>
</div>

<center>
    <img width="80%" style="padding-top: 0px;" src="https://bair.berkeley.edu/static/blog/trajectory_transformer/d4rl.png">
    &nbsp;
    <br>
    <img width="80%" src="https://bair.berkeley.edu/static/blog/trajectory_transformer/bar.png">
    <br>
    <p style="text-align:center; margin-left:10%; margin-right:10%; padding-top: 20px; padding-bottom: 10px;">
    <i style="font-size: 0.9em;">
        Performance on the locomotion environments in the <a href="https://arxiv.org/abs/2004.07219">D4RL offline benchmark suite.</a> We compare two variants of the Trajectory Transformer (TT) &mdash; differing in how they discretize continuous inputs &mdash; with model-based, value-based, and recently proposed sequence-modeling algorithms.
    </i>
    </p>
</center>
<br>

<!-- begin section II: outlook -->

<h3 id="model">What does this mean for reinforcement learning?</h3>

<p>
    The Trajectory Transformer is something of an exercise in minimalism.
    Despite lacking most of the common ingredients of a reinforcement learning algorithm, it performs on par with approaches that have been the result of much collective effort and tuning.
    Taken together with the concurrent <a href="https://arxiv.org/abs/2106.01345">Decision Transformer</a>, this result highlights that scalable architectures and stable training objectives can sidestep some of the difficulties of reinforcement learning in practice.
</p>

<p>
    However, the simplicity of the proposed approach gives it predictable weaknesses.
    Because the Transformer is trained with a maximum likelihood objective, it is more dependent on the training distribution than a conventional dynamic programming algorithm.
    Though there is value in studying the most streamlined approaches that can tackle reinforcement learning problems, it is possible that the most effective instantiation of this framework will come from combinations of the sequence modeling and reinforcement learning toolboxes.
</p>

<p>
    We can get a preview of how this would work with a fairly straightforward combination: plan using the Trajectory Transformer as before, but use a $Q$-function trained via dynamic programming as a search heuristic to guide the beam search planning procedure.
    We would expect this to be important in sparse-reward, long-horizon tasks, since these pose particularly difficult search problems.
    To instantiate this idea, we use the $Q$-function from the <a href="https://arxiv.org/abs/2110.06169">implicit $Q$-learning</a> (IQL) algorithm and leave the Trajectory Transformer otherwise unmodified.
    We denote the combination <b>TT</b>$_{\color{#999999}{(+Q)}}$:
</p>

<center>
    <img width="80%" style="padding-top: 20px;" src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/antmaze.png">
    <p style="text-align:center; margin-left:10%; margin-right:10%; padding-top: 20px; padding-bottom: 10px;">
    <i style="font-size: 0.9em;">
        Guiding the Trajectory Transformer's plans with a $Q$-function trained via dynamic programming (TT$_{\color{#999999}{(+Q)}}$) is a straightforward way of improving empirical performance compared to model-free (CQL, IQL) and return-conditioning (DT) approaches.
        We evaluate this effect in the sparse-reward, long-horizon <a href="https://arxiv.org/abs/2004.07219">AntMaze goal-reaching tasks</a>.
    </i>
    </p>
</center>
<br>

<p>
    Because the planning procedure only uses the $Q$-function as a way to filter promising sequences, it is not as prone to local inaccuracies in value predictions as policy-extraction-based methods like <a href="https://arxiv.org/abs/2006.04779">CQL</a> and <a href="https://arxiv.org/abs/2110.06169">IQL</a>.
    However, it still benefits from the temporal compositionality of dynamic programming and planning, so outperforms return-conditioning approaches that rely more on complete demonstrations.
</p>

<p>
    Planning with a terminal value function is a time-tested strategy, so $Q$-guided beam search is arguably the simplest way of combining sequence modeling with conventional reinforcement learning.
    This result is encouraging not because it is new algorithmically, but because it demonstrates the empirical benefits even straightforward combinations can bring.
    It is possible that designing a sequence model from the ground-up for this purpose, so as to retain the scalability of Transformers while incorporating the principles of dynamic programming, would be an even more effective way of leveraging the strengths of each toolkit.
</p>

<hr>

<p>
    This post is based on the following paper:
</p>

<ul>
    <li>
        <a href="https://arxiv.org/abs/2106.02039"><strong>Offline Reinforcement Learning as One Big Sequence Modeling Problem</strong></a>
        <br>
        <a href="http://michaeljanner.com/">Michael Janner</a>, <a href="https://scholar.google.com/citations?user=qlwwdfEAAAAJ&hl=en">Qiyang Li</a>, and <a href="https://people.eecs.berkeley.edu/~svlevine/">Sergey Levine</a>
        <br>
        <em>Neural Information Processing Systems (NeurIPS), 2021.</em>
        <br>
        <a href="https://github.com/JannerM/trajectory-transformer">Open-source code</a>
    </li>
</ul>

<hr>
<div class="footnotes">
  <ol>
    <li id="fn:anderson">
      <p>
        Though qualitative capabilities advances from scale alone might seem surprising, physicists have long known that <a href="https://cse-robotics.engr.tamu.edu/dshell/cs689/papers/anderson72more_is_different.pdf">more is different</a>.
        <a href="#fnref:anderson" class="reversefootnote">↩</a>
      </p>
    </li>
    <li id="fn:mpc">
      <p>
        You could also enact multiple actions from the sequence, or act according to a closed-loop controller until there has been enough time to generate a new plan.
        <a href="#fnref:mpc" class="reversefootnote">↩</a>
      </p>
    </li>
  </ol>
</div>
<hr>
