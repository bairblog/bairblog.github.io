---
layout:             post
title:              "The Successor Representation, $\\gamma$-Models,<br> and Infinite-Horizon Prediction"
date:               2021-01-04  9:00:00
author:             <a href="https://people.eecs.berkeley.edu/~janner/">Michael Janner</a> <br />
img:                assets/successor/gamma-teaser.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<!-- twitter -->
<meta name="twitter:title" content="The Successor Representation, γ-Models and Infinite-Horizon Prediction">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/successor/twitter-card-0.98-01.png">

<meta name="keywords" content="successor, representation, SR, gamma-models, gamma, models, reinforcement, learning, generative, temporal, difference">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Michael Janner">

<title>The Successor Representation, Gamma-Models, and Infinite-Horizon Prediction</title>


<script>
function increment_img(id) {
    element = document.getElementById(id)
    src = element.src
    length = src.length

    ind_string = src.substring(length-6, length-4)
    ind = parseInt(ind_string, 10)

    next_ind = (ind+1) % 10
    next_ind_string = next_ind.toString().padStart(2, '0')

    next_src = src.replace(ind_string, next_ind_string)
    element.src = next_src
}

setInterval(function() {
    increment_img('rollout')
    // console.log(src, next_src)
}, 1 * 1000);
</script>

<!-- begin section I: introduction -->

<br>
<p style="text-align:center;">
    <img src="https://bair.berkeley.edu/static/blog/successor/gamma-teaser.png" width="100%">
    <br>
    <i>Standard single-step models have a horizon of one. This post describes a method for training predictive dynamics models in continuous state spaces with an infinite, probabilistic horizon.</i>
</p>
<br>

<p>
    Reinforcement learning algorithms are frequently categorized by whether they predict future states at any point in their decision-making process. Those that do are called <i>model-based</i>, and those that do not are dubbed <i>model-free</i>. This classification is so common that we mostly take it for granted these days; I am <a href="https://bair.berkeley.edu/blog/2019/12/12/mbpo/">guilty of using it myself</a>. However, this distinction is not as clear-cut as it may initially seem.
</p>

<p>
     In this post, I will talk about an alternative view that emphases the mechanism of prediction instead of the content of prediction. This shift in focus brings into relief a space between model-based and model-free methods that contains exciting directions for reinforcement learning. The first half of this post describes some of the classic tools in this space, including
     <a href="https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/horde1.pdf">generalized value functions</a> and the <a href="http://www.gatsby.ucl.ac.uk/~dayan/papers/d93b.pdf">successor representation</a>. The latter half is based on our recent paper about <a href="https://arxiv.org/abs/2010.14496">infinite-horizon predictive models</a>, for which code is available <a href="https://github.com/JannerM/gamma-models">here</a>.
 </p>

<!--more-->

<!-- begin section II: what-versus-how -->

<h2 id="what-how">The <i>what</i> versus <i>how</i> of prediction</h2>

<p>
    The dichotomy between model-based and model-free algorithms focuses on what is predicted directly: states or values. Instead, I want to focus on how these predictions are made, and specifically how these approaches deal with the complexities arising from long horizons.
</p>

<p>
    Dynamics models, for instance, approximate a single-step transition distribution, meaning that they are trained on a prediction problem with a horizon of one. In order to make a short-horizon model useful for long-horizon queries, its single-step predictions are composed in the form of sequential model-based rollouts. We could say that the “testing” horizon of a model-based method is that of the rollout.
</p>

<p>
    In contrast, value functions themselves are long-horizon predictors; they need not be used in the context of rollouts because they already contain information about the extended future. In order to amortize this long-horizon prediction, value functions are trained with either Monte Carlo estimates of expected cumulative reward or with dynamic programming. The important distinction is now that the long-horizon nature of the prediction task is dealt with during training instead of during testing.
</p>

<br>
<center>
<div style="width: 90%;">
    <div style="width: 45%; float: left;">
        <img src="https://bair.berkeley.edu/static/blog/successor/mb-mf.png" width="100%">
    </div>
    <div style="width: 46%; float: right;">
        <p>
            <br>
            <i>We can organize reinforcement learning algorithms in terms of when they deal with long-horizon complexity. Dynamics models train for a short-horizon prediction task but are deployed using long-horizon rollouts. In contrast, value functions amortize the work of long-horizon prediction at training, so a single-step prediction (and informally, a shorter "horizon") is sufficient during testing.</i>
            <br><br>
        </p>
    </div>
    <br>
</div>
</center>
<br clear="left"/>
<br>

<p>
    Taking this view, the fact that models predict states and value functions predict cumulative rewards is almost a detail. What really matters is that models predict <i>immediate</i> next states and value functions predict <i>long-term sums</i> of rewards. This idea is nicely summarized in a line of work on <a href="https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/horde1.pdf">generalized</a> <a href="https://sites.ualberta.ca/~amw8/phd.pdf">value</a> <a href="http://incompleteideas.net/papers/maei-sutton-10.pdf">functions</a>, describing how temporal difference learning may be used to make long-horizon predictions about any kind of cumulant, of which a reward function is simply one example.
</p>

<p>
    This framing also suggests that some phenomena we currently think of as distinct, like <a href="https://arxiv.org/abs/1906.08253">compounding model prediction errors</a> and <a href="https://arxiv.org/abs/1906.00949">bootstrap error accumulation</a>, might actually be different lenses on the same problem. The former describes the growth in error over the course of a model-based rollout, and the latter describes the propagation of error via the Bellman backup in model-free reinforcement learning. If models and value functions differ primarily in when they deal with horizon-based difficulties, then it should come as no surprise that the testing-time error compounding of models has a direct training-time analogue in value functions.
</p>

<p>
    A final reason to be interested in this alternative categorization is that it allows us to think about hybrids that do not make sense under the standard dichotomy. For example, if a model were to make long-horizon state predictions by virtue of training-time amortization, it would avoid the need for sequential model-based rollouts and circumvent testing-time compounding errors. The remaining sections describe how we can build such a model, beginning with the foundation of the successor representation and then introducing new work for making this form of prediction compatible with continuous spaces and neural samplers.
</p>


<!-- begin section III: the successor representation -->

<h2 id="model-based-techniques">The successor representation</h2>

<p>
    The <a href="http://www.gatsby.ucl.ac.uk/~dayan/papers/d93b.pdf">successor representation</a> (SR), an idea influential in both <a href="https://www.nature.com/articles/s41562-017-0180-8">cognitive</a> <a href="https://www.jneurosci.org/content/38/33/7193">science</a> and <a href="https://arxiv.org/abs/1606.02396">machine</a> <a href="https://arxiv.org/abs/1606.05312">learning</a>, is a long-horizon, policy-dependent dynamics model. It leverages the insight that the same type of recurrence relation used to train \(Q\)-functions:

    \[
        Q(\mathbf{s}_t, \mathbf{a}_t) \leftarrow
        \mathbb{E}_{\mathbf{s}_{t+1}}
            [r(\mathbf{s}_{t}, \mathbf{a}_t, \mathbf{s}_{t+1}) + \gamma V(\mathbf{s}_{t+1})]
    \]

    may also be used to train a model that predicts states instead of values:

    \[
        M(\mathbf{s}_{t}, \mathbf{a}_t) \leftarrow
        \mathbb{E}_{\mathbf{s}_{t+1}}
            [\mathbf{1}(\mathbf{s}_{t+1}) + \gamma M(\mathbf{s}_{t+1})] \tag*{(1)}
    \]
</p>

<p>
    The key difference between the two is that the scalar rewards \(r(\mathbf{s}_t, \mathbf{a}_t, \mathbf{s}_{t+1})\) from the \(Q\)-function recurrence are now replaced with one-hot indicator vectors \(\mathbf{1}(\mathbf{s}_{t+1})\) denoting states. As such, SR training may be thought of as vector-valued \(Q\)-learning. The size of the “reward” vector, as well as the successor predictions \(M(\mathbf{s}_t, \mathbf{a}_t)\) and \(M(\mathbf{s}_t)\), is equal to the number of states in the MDP.
</p>


<p>
    In contrast to standard dynamics models, which approximate a single-step transition distribution, SR approximates what is known as the discounted occupancy:

    \[
        \mu(\mathbf{s}_e \mid \mathbf{s}_t, \mathbf{s}_t) = (1 - \gamma)
        \sum_{\Delta t=1}^{\infty} \gamma^{\Delta t - 1}
        p(
            \mathbf{s}_{t+\Delta t} = \mathbf{s}_e \mid
            \mathbf{s}_t, \mathbf{a}_t, \pi
        )
    \]
</p>


<p>
    This occupancy is a weighted mixture over an infinite series of multi-step models, with the mixture weights being controlled by a discount factor \(\gamma\).<sup id="fnref:exit-state"><a href="#fn:exit-state" class="footnote"><font size="-2">1</font></a></sup> <sup id="fnref:options"><a href="#fn:options" class="footnote"><font size="-2">2</font></a></sup> Setting  \(\gamma=0\) recovers a standard single-step model, and any \(\gamma \in (0,1)\) induces a model with an infinite, probabilistic horizon. The predictive lookahead of the model qualitatively increases with larger \(\gamma\).
</p>

<p style="text-align:center;">
    <img src="https://bair.berkeley.edu/static/blog/successor/tolman.gif" width="60%">
    <br>
    <i>The successor representation of a(n optimal) rat in a maze<sup id="fnref:maze"><a href="#fn:maze" class="footnote"><font size="-2">3</font></a></sup>, showing the rat’s path with a probabilistic horizon determined by discount factor \(\gamma\).
</i>
</p>

<!-- begin section III: gamma-models -->

<h2 id="gamma-models">Generative models in continuous spaces: from SR to \(\boldsymbol{\gamma}\)-models</h2>

<p>
    <a href="https://arxiv.org/abs/1606.02396">Continuous</a> <a href="https://arxiv.org/abs/1606.05312">adaptations</a> of SR replace the one-hot state indicator \(\mathbf{1}(\mathbf{s}_t)\) in Equation 1 with a learned state featurization \(\phi(\mathbf{s}_t, \mathbf{a}_t)\), giving a recurrence of the form:

    \[
        \psi(\mathbf{s}_t, \mathbf{a}_t) \leftarrow \phi(\mathbf{s}_t, \mathbf{a}_t) + \gamma
        \mathbb{E}_{\mathbf{s}_{t+1}} [\psi(\mathbf{s}_{t+1})]
    \]
</p>

<p>
    This is not a generative model in the usual sense, but is instead known as an expectation model: \(\psi\) denotes the expected feature vector \(\phi\). The advantage to this approach is that an expectation model is easier to train than a generative model. Moreover, if rewards are linear in the features, an expectation model is sufficient for value estimation.
</p>

<p>
    However, the limitation of an expectation model is that it cannot be employed in some of the most common use-cases of predictive dynamics models. Because \(\psi(\mathbf{s}_t, \mathbf{a}_t)\) only predicts a first moment, we cannot use it to sample future states or perform model-based rollouts.
</p>

<p>
    To overcome this limitation, we can turn the discriminative update used in SR and its continuous variants into one suitable for training a generative model \({\color{#D62728}\mu}\):

    \[
        \max_{\color{#D62728}\mu} \mathbb{E}_{\mathbf{s}_t, \mathbf{a}_t, \mathbf{s}_{t+1} \sim \mathcal{D}} [ \mathbb{E}_{
            \mathbf{s}_e \sim (1-\gamma) p(\cdot \mid \mathbf{s}_t, \mathbf{a}_t) + \gamma
            {\color{#D62728}\mu}(\cdot \mid \mathbf{s}_{t+1})
        }
        [\log {\color{#D62728}\mu}(\mathbf{s}_e \mid \mathbf{s}_t, \mathbf{a}_t)] ]
    \]
</p>

<p>
    On first glance, this looks like a standard maximum likelihood objective. The important difference is that the distribution over which the inner expectation is evaluated depends on the model \({\color{#D62728}\mu}\) itself. Instead of a bootstrapped target value like those commonly used in model-free algorithms, we now have a bootstrapped target distribution.

    \[
        \underset{
            \vphantom{\Huge\Sigma}
            \Large \text{target value}
        }{
            r + \gamma V
        } 
        ~~~~~~~~ \Longleftrightarrow ~~~~~~~~
        \underset{
            \vphantom{\Huge\Sigma}
            \Large \text{target }\color{#D62728}{\text{distribution}}
        }{
            (1-\gamma) p + \gamma {\color{#D62728}\mu}
        }
    \]

    Varying the discount factor \(\gamma\) in the target distribution yields models that predict increasingly far into the future.
</p>

<br>
<p style="text-align:center;">
    <img src="https://bair.berkeley.edu/static/blog/successor/gamma-flow-acrobot.png" width="100%">
    <br>
    <br>
    <i>Predictions of a \(\gamma\)-model for varying discounts \(\gamma\). The rightmost column shows Monte Carlo estimates of the discounted occupancy corresponding to \(\gamma=0.95\) for reference. The conditioning state is denoted by \(\circ\).
</i>
</p>
<br>

<p>
    In the spirit of infinite-horizon model-free control, we refer to this formulation as infinite-horizon prediction and the corresponding model as a \(\gamma\)-model. Because the bootstrapped maximum likelihood objective circumvents the need for reward vectors the size of the state space, \(\gamma\)-model training is suitable for continuous spaces while retaining an interpretation as a generative model. In our paper we show how to instantiate \(\gamma\)-models as both <a href="https://arxiv.org/abs/1505.05770">normalizing flows</a> and <a href="https://arxiv.org/abs/1406.2661">generative adversarial networks</a>.
</p>

<!-- begin section IV: model-based control -->

<h2 id="model-based-control">Generalizing model-based control with \(\boldsymbol{\gamma}\)-models</h2>

<p>
    Replacing single-step dynamics models with \(\gamma\)-models leads to generalizations of some of the staples of model-based control:
</p>

<p>
    <strong>Rollouts:</strong>&nbsp;  \(\gamma\)-models divorce timestep from model step. As opposed to incrementing one timestep into the future with every prediction, \(\gamma\)-model rollout steps have a negative binomial distribution over time. It is possible to reweight these \(\gamma\)-model steps to simulate the predictions of a model trained with higher discount.
</p>

<p style="text-align:center;">
    <img id="rollout" src="https://bair.berkeley.edu/static/blog/successor/rollout_00.png" width="100%">
    <br>
    <i>Whereas conventional dynamics models predict a single step into the future, \(\gamma\)-model rollout steps have a negative binomial distribution over time. The first step of a \(\gamma\)-model has a geometric distribution from the special case of \(~\text{NegBinom}(1, p) = \text{Geom}(1-p)\).</i>
</p>
<br>

<p>
    <strong>Value estimation:</strong>&nbsp;  Single-step models estimate values using long model-based rollouts, often between tens and hundreds of steps long. In contrast, values are expectations over a single feedforward pass of a \(\gamma\)-model. This is similar to a decomposition of value as an inner product, as seen in <a href="https://arxiv.org/abs/1606.05312">successor features</a> and <a href="https://arxiv.org/abs/1606.02396">deep SR</a>. In tabular spaces with indicator rewards, the inner product and expectation are the same!
</p>


<div>
    <center>
    <figure class="video_container">
    <video width="80%" height="auto" autoplay loop playsinline muted>
      <source src="https://bair.berkeley.edu/static/blog/successor/value_estimation.mp4" type="video/mp4">
    </video>
    </figure>
    </center>
    <p style="text-align:center;">
    <i>Because values are expectations of reward over a single step of a \(\gamma\)-model, we can perform value estimation without sequential model-based rollouts.</i>
    </p>
</div>
<br>

<p>
    <strong>Terminal value functions:</strong>&nbsp;  To account for truncation error in single-step model-based rollouts, it is common to augment the rollout with a terminal value function. This strategy, sometimes referred to as <a href="https://arxiv.org/abs/1803.00101">model-based value expansion</a> (MVE), has an abrupt handoff between the model-based rollout and the model-free value function. We can derive an analogous strategy with a \(\gamma\)-model, called \(\gamma\)-MVE, that features a gradual transition between model-based and model-free value estimation. This value estimation strategy can be incorporated into a model-based reinforcement learning algorithm for improved sample-efficiency.
</p>

<br>
<p style="text-align:center;">
    <img src="https://bair.berkeley.edu/static/blog/successor/consolidated.png" width="100%">
    <br>
    <i>
        \(\gamma\)-MVE features a gradual transition between model-based and model-free value estimation.
    </i>
</p>

<hr>

<p>
    This post is based on the following paper:
</p>

<ul>
    <li>
        <a href="https://arxiv.org/abs/2010.14496"><strong>\(\gamma\)-Models: Generative Temporal Difference Learning for Infinite-Horizon Prediction</strong></a>
        <br>
        <a href="https://people.eecs.berkeley.edu/~janner/">Michael Janner</a>, <a href="https://scholar.google.com/citations?user=Vzr1RukAAAAJ&hl=en">Igor Mordatch</a>, and <a href="https://people.eecs.berkeley.edu/~svlevine/">Sergey Levine</a>
        <br>
        <em>Neural Information Processing Systems (NeurIPS), 2020.</em>
        <br>
        <a href="https://github.com/JannerM/gamma-models">Open-source code</a> (runs in your browser!)
    </li>
</ul>

<hr>

<div class="footnotes">
  <ol>
    <li id="fn:exit-state">
      <p>
        The \(e\) subscript in \(\mathbf{s}_e\) is short for "exit", which comes from an interpretation of the discounted occupancy as the exit state in a modified MDP in which there is a constant \(1-\gamma\) probability of termination at each timestep.<a href="#fnref:exit-state" class="reversefootnote">↩</a>
    </p>
    </li>
    <li id="fn:options">
      <p>
        Because the discounted occupancy plays such a central role in reinforcement learning, its approximation by Bellman equations has been a focus in multiple lines of research. <a href="https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf">Option models</a> and <a href="http://www.incompleteideas.net/papers/sutton-95.pdf">\(\beta\)-models</a> describe generalizations of this idea that allow for state-dependent termination conditions and arbitrary timestep mixtures.<a href="#fnref:options" class="reversefootnote">↩</a>
     </p>
    </li>
    <li id="fn:maze">
      <p>
        If this particular maze looks familiar, you might have seen it in Tolman’s <a href="https://personal.utdallas.edu/~tres/spatial/tolman.pdf">Cognitive Maps in Rats and Men</a>. (Our web version has been stretched slightly horizontally.)<a href="#fnref:maze" class="reversefootnote">↩</a>
     </p>
    </li>
  </ol>
</div>

<hr>

<p>
<font size="-1">
<strong>References</strong>
<ol style="margin-top:-15px">
    <li>A Barreto, W Dabney, R Munos, JJ Hunt, T Schaul, HP van Hasselt, and D Silver. <a href="https://arxiv.org/abs/1606.05312">Successor features for transfer in reinforcement learning.</a> <i>NeurIPS</i> 2017.</li>
    <li>P Dayan. <a href="http://www.gatsby.ucl.ac.uk/~dayan/papers/d93b.pdf">Improving generalization for temporal difference learning: The successor representation.</a> <i>Neural Computation</i> 1993.</li>
    <li>Vladimir Feinberg, Alvin Wan, Ion Stoica, Michael I. Jordan, Joseph E. Gonzalez, Sergey Levine. <a href="https://arxiv.org/abs/1803.00101">Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning.</a> <i>ICML</i> 2018.</li>
    <li>SJ Gershman. <a href="https://www.jneurosci.org/content/38/33/7193">The successor representation: Its computational logic and neural substrates.</a> <i>Journal of Neuroscience</i> 2018.</li>
    <li>IJ Goodfellow, J Pouget-Abadie, M Mirza, B Xu, D Warde-Farley, S Ozair, A Courville, Y Bengio. <a href="https://arxiv.org/abs/1406.2661">Generative Adversarial Networks.</a> <i>NeurIPS</i> 2014.</li>
    <li>M Janner, J Fu, M Zhang, S Levine. <a href="https://arxiv.org/abs/1906.08253">When to Trust Your Model: Model-Based Policy Optimization.</a> <i>NeurIPS</i> 2019.</li>
    <li>TD Kulkarni, A Saeedi, S Gautam, and SJ Gershman. <a href="https://arxiv.org/abs/1606.02396">Deep successor reinforcement learning.</a> 2016.</li>
    <li>A Kumar, J Fu, G Tucker, S Levine. <a href="https://arxiv.org/abs/1906.00949">Stabilizing Off-Policy \(Q\)-Learning via Bootstrapping Error Reduction.</a> <i>NeurIPS</i> 2019.</li>
    <li>HR Maei and RS Sutton. <a href="http://incompleteideas.net/papers/maei-sutton-10.pdf">GQ(\(\lambda\)): A general gradient algorithm for temporal-difference prediction learning with eligibility traces.</a> <i>AGI</i> 2010.</li>
    <li>I Momennejad, EM Russek, JH Cheong, MM Botvinick, ND Daw, and SJ Gershman. <a href="https://www.nature.com/articles/s41562-017-0180-8">The successor representation in human reinforcement learning.</a> <i>Nature Human Behaviour</i> 2017.</li>
    <li>DJ Rezende and S Mohamed. <a href="https://arxiv.org/abs/1505.05770">Variational Inference with Normalizing Flows.</a> <i>ICML</i> 2015.</li>
    <li>RS Sutton. <a href="http://www.incompleteideas.net/papers/sutton-95.pdf">TD Models: Modeling the World at a Mixture of Time Scales.</a> <i>ICML</i> 1995.</li>
    <li>RS Sutton, J Modayil, M Delp, T Degris, PM Pilarski, A White, and D Precup. <a href="https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/horde1.pdf">Horde:  A  scalable  real-time  architecture  for  learning  knowledge  from  unsupervised sensorimotor interaction.</a> <i>AAMAS</i> 2011.</li>
    <li>RS Sutton, D Precup, and S Singh. <a href="https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf">Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.</a> <i>Artificial Intelligence</i> 1999.</li>
    <li>E Tolman. <a href="https://personal.utdallas.edu/~tres/spatial/tolman.pdf">Cognitive Maps in Rats and Men.</a> <i>Psychological Review</i> 1948.</li>
    <li>A White. <a href="https://sites.ualberta.ca/~amw8/phd.pdf">Developing a predictive approach to knowledge.</a> PhD thesis, 2015.</li>
</ol>
</font>
</p>


