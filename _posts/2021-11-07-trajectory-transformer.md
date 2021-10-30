---
layout:             post
title:              "Reinforcement Learning as Sequence Modeling"
date:               2021-11-07  9:00:00
author:             <a href="https://people.eecs.berkeley.edu/~janner/">Michael Janner</a>
img:                assets/successor/gamma-teaser.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<!-- twitter -->
<meta name="twitter:title" content="Sequence Modeling Solutions for Reinforcement Learning Problems">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/successor/twitter-card-0.98-01.png">

<meta name="keywords" content="trajectory, transformer, reinforcement, learning, RL">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Michael Janner">

<title>Reinforcement Learning as Sequence Modeling</title>


<!-- begin section I: introduction -->

<p style="text-align:center; margin-top:-40px;">
    <br>
    <!-- <b><font size="3">Trajectory Transformer</font></b> -->
    <video width="100%" autoplay playsinline muted>
        <source src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/rollout_transformer.mp4" type="video/mp4">
    </video>
    <!-- <b><font size="3">Single-Step Model</font></b> -->
    <video width="100%" autoplay playsinline muted>
        <source src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/rollout_single.mp4" type="video/mp4">
    </video>
    <p width="80%" style="text-align:center; margin-left:10%; margin-right:10%; padding-bottom: -10px;">
        <i style="font-size: 18px;">
        Long-horizon predictions of the (top) <b><span style="color:#D62728;">Trajectory Transformer</span></b> compared <br>to those of a (bottom) <b><span style="color:#D62728;">single-step</span></b> dynamics model.
        </i>
    </p>
<br>

<p>
    Modern <a href="https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html">machine</a> <a href="https://arxiv.org/abs/1807.03748">learning</a> <a href="https://www.nature.com/articles/s41586-021-03819-2">success</a> <a href="https://arxiv.org/abs/2002.05709">stories</a> often have one thing in common: they use methods that scale gracefully with ever-increasing amounts of data.
    This is particularly clear from recent advances in sequence modeling, where simply increasing the size of a stable architecture and its training set leads to <a href="https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf">qualitatively</a> <a href="https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">different</a> <a href="https://arxiv.org/abs/2005.14165">capabilites</a>.<sup id="fnref:anderson"><a href="#fn:anderson" class="footnote"><font size="-2">1</font></a></sup>
</p>

<p>
    Meanwhile, the situation in reinforcement learning has proven more complicated.
    While it has been possible to apply reinforcement learning algorithms to <a href="https://journals.sagepub.com/doi/full/10.1177/0278364917710318">large</a>-<a hrfe="https://www.science.org/doi/10.1126/science.aar6404">scale</a> <a href="https://arxiv.org/abs/1912.06680">problems</a>, generally there has been much more friction in doing so.
    In this post, we explore whether we can alleviate these difficulties by tackling the reinforcement learning problem with the toolbox of sequence modeling.
    The end result is a generative model of trajectories that looks like a <a href="https://arxiv.org/abs/1706.03762">large language model</a> and a planning algorithm that looks like <a href="https://kilthub.cmu.edu/articles/journal_contribution/Speech_understanding_systems_summary_of_results_of_the_five-year_research_effort_at_Carnegie-Mellon_University_/6609821/1">beam search</a>.
    Code for the approach can be found <a href="https://github.com/JannerM/trajectory-transformer">here</a>.
</p>

<h3 id="models">The Trajectory Transformer</h3>

<p>
    The standard framing of reinforcement learning focuses on decomposing a complicated long-horizon problem into smaller, more tractable subproblems, leading to the class of dynamic programming methods like $Q$-learning and an emphasis on Markovian dynamics models.
    However, we can also view reinforcement learning as analogous to a sequence generation problem, with the goal being to produce a sequence of actions that, when enacted in an environment, will yield a sequence of high rewards.
</p>

<p>
    Taking this view to its logical conclusion, we begin by modeling the trajectory data provided to reinforcement learning algorithms with a Transformer architecture, the current tool of choice for modeling long-horizon dependencies.
    We treat these trajectories as unstructured sequences of (autoregressively discretized) states, actions, and rewards, and train the Transformer architecture using the standard cross-entropy loss.
    Modeling all trajectory data with a single high-capacity model and scalable training objective, as opposed to separate procedures for dynamics models, policies, and $Q$-functions, allows for a more streamlined approach that removes much of the usual complexity.
</p>

<center>
    <img width="80%" style="padding-top: 20px; padding-bottom: 20px" src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/architecture.png">
    <br>
    <p width="80%" style="text-align:center; margin-left:10%; margin-right:10%; padding-bottom: 10px;">
        <i style="font-size: 18px;">
        We model the distribution over $N$-dimensional states $\mathbf{s}_t$, $M$-dimensional actions $\mathbf{a}_t$, and scalar rewards $r_t$ using a Transformer architecture.
        </i>
    </p>
</center>

<!-- <p>
    The Trajectory Transformer depicted above models all parts of the trajectory jointly.
    Conventionally, state predictions and action proposals are handled by independently-trained dynamics models and policies.
    It will later prove useful to augment these trajectories with Monte-Carlo estimates of return-to-go, causing the Transformer to additionally serve the role of a $Q$-function.
    Handling these separate concerns with a single high-capacity model and scalable training objective allows for a more streamlined approach that removes much of the complexity normally associated with each individual modeling problem.
</p> -->
<!--more-->

<!-- begin section II: models -->

<h3 id="models">Transformers as dynamics models</h3>

<p>
    We find that the expressiveness and capacity of Transformers in language modeling also transfers to modeling dynamical systems.
    While compounding errors in the rollouts of single-step models often make them unreliable for control, the long-horizon predictions of the Trajectory Transformer remain accurate:
</p>

<br>
<center>
<div style="width: 90%;">
    <div style="width: 50%; float: left;">
        <img src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/error_blog.png" width="100%">
    </div>
    <div style="width: 47.5%; float: right;">
        <p>
            <br><br>
            <i>The Transformer's long horizon predictions are substantially more accurate than those from a single-step model.
            For a qualitative view of the same data, see the video at the top of this page.</i>
            <br><br>
        </p>
    </div>
    <br>
</div>
</center>
<br clear="left"/>
<br>

<p>
    We can also inspect the Transformer's predictions as one would a standard language model.
    A common strategy in machine translation, for example, is to <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">visualize the intermediate token weights</a> as a proxy for token dependencies.
    The same visualization applied to Trajectory Transformer reveals two salient patterns:
</p>

<center>
    <img width="35%" style="padding-top: 10px; padding-right: 60px;" src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/markov.png">
    <img width="35%" src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/striated.png">
    <br>
    <p width="80%" style="text-align:center; margin-left:10%; margin-right:10%; padding-top: 20px; padding-bottom: 10px;">
        <i style="font-size: 18px;">
        Attention patterns of Trajectory Transformer, showing (left) a discovered <br>Markovian stratetgy and (right) an approach with action smoothing.
        </i>
    </p>
</center>

<p>
    In the first, state and action predictions depend primarily on the immediately preceding transition, resembling a learned Markov property.
    In the second, state dimension predictions depend most strongly on the corresponding dimensions of all previous states, and action dimensions depend primarily on all prior actions.
    While the second dependency violates the usual intuition of actions depending on only the prior state, this is reminiscient of the action smoothing used in some <a href="https://arxiv.org/abs/1909.11652">trajectory optimization algorithms</a> to enforce slowly varying control sequences.
</p>


<!-- begin section II: planning -->

<h3 id="planning">Beam search as trajectory optimizer</h3>

<p>
    While the architectural workhorse of contemporary language modeling is a recent innovation, the procedure for decoding these models has remained remarkably consistent since the earliest days of computational linguistics.
    We show that this pruned breadth-first search known as <a href="https://kilthub.cmu.edu/articles/journal_contribution/Speech_understanding_systems_summary_of_results_of_the_five-year_research_effort_at_Carnegie-Mellon_University_/6609821/1">beam search</a> can also be a surprisingly effective planning algorithm.
    We study this in three different settings:
</p>
<div>
  <ol>
    <li><p>
        <b><span style="color:#D62728;">Imitation:</span></b> If we use beam search without modification, we sample trajectories that are probable under the distribution of the training data, giving us a long-horizon model-based variant of imitation learning.
    </p></li>
    <li><p>
       <b><span style="color:#D62728;">Goal-conditioned RL:</span></b> Conditioning the Transformer on <i>future</i> desired context alongside previous states, actions, and rewards yields a goal-reaching method. This works by recontextualizing past data as optimal for some task, in the same spirit as <a href="https://arxiv.org/abs/1707.01495">hindsight relabeling</a>.
    </p>
        <center>
          <img width="28%" src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/0.png">
          &nbsp;
          &nbsp;
          <img width="28%" src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/1.png">
          &nbsp;
          &nbsp;
          <img width="28%" src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/2.png">
          <br>
          <img width="2%" src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/rolloutblack-1.png">
          &nbsp;
          Start
          &nbsp;
          &nbsp;
          <img width="2%" src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/rolloutblue-1.png">
          &nbsp;
          Goal
        </center>
        <p width="90%" style="text-align:center; padding-top: 20px; padding-bottom: 10px;">
        <i style="font-size: 18px;">
            Paths taken by the goal-conditioned beam-search planner in a four-rooms environment.
        </i>
    </p>
    </li>
    <li><p>
        <b><span style="color:#D62728;">Offline RL:</span></b> If we replace transitions' log probabilities with their rewards (their <a href="https://arxiv.org/abs/1805.00909">log probability of optimality</a>), we can use the same beam search framework to optimize for reward-maximizing behavior.
        We find that this simple combination of a trajectory-level sequence model and beam search decoding performs on par with the best prior offline reinforcement learning algorithms:
    </p></li>
  </ol>
</div>

<br>
<center>
    <img width="80%" src="https://people.eecs.berkeley.edu/~janner/trajectory-transformer/blog/bar.png">
    <br>
    <p width="80%" style="text-align:center; margin-left:0%; margin-right:0%; padding-top: 20px; padding-bottom: 10px;">
    <i style="font-size: 18px;">
        Performance on the locomotion environments in the <a href="https://arxiv.org/abs/2004.07219">D4RL offline benchmark suite</a>.
    </i>
    </p>
</center>
<br>


<!-- begin section II: outlook -->

<h3 id="model">What does this mean for reinforcement learning?</h3>

<p>
    The Trajectory Transformer is something of an exercise in minimalism.
    Despite lacking most of the common ingredients of a reinforcement learning algorithm, it performs on par with approaches that have been the result of much collective effort and tuning.
    Taken together with the concurrent <a href="https://arxiv.org/abs/2106.01345">Decision Transformer</a>, this result highlights that scalable architectures and stable training objectives can iron out many of the difficulties of reinforcement learning in practice.
</p>

<p>
    However, the simplicity of the proposed approach gives it some predictable weaknesses.
    Because the Transformer is trained with a maximum likelihood objective, it is more dependent on the training distribution than a conventional dynamic programming algorithm.
    Though there is value in studying the most streamlined approaches that can tackle the reinforcement learning problem, it is possible that the most effective instantiation of this framework will come from combinations of the sequence modeling and reinforcement learning toolboxes.
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

<!-- <p>
    <em>I would like to thank Michael Chang and Sergey Levine for their valuable feedback.</em>
</p> -->

<hr>
<div class="footnotes">
  <ol>
    <li id="fn:anderson">
      <p>
        Though qualitative capabilities advances from scale alone might have been surprising, physicists have long known that <a href="https://cse-robotics.engr.tamu.edu/dshell/cs689/papers/anderson72more_is_different.pdf">more is different</a>.
        <a href="#fnref:anderson" class="reversefootnote">↩</a>
    </p>
    </li>
  </ol>
</div>
<hr>

<!-- <p>
<font size="-1">
<strong>References</strong>
<ol style="margin-top:-15px">
    <li>A Barreto, W Dabney, R Munos, JJ Hunt, T Schaul, HP van Hasselt, and D Silver. <a href="https://arxiv.org/abs/1606.05312">Successor features for transfer in reinforcement learning.</a> <i>NeurIPS</i> 2017.</li>
</ol>
</font>
</p>
 -->

