---
layout:             post
title:              "Functional RL with Keras and Tensorflow Eager"
date:               2019-10-14 9:00:00
author:             <a href="https://www.linkedin.com/in/eric-liang-31308019/">Eric Liang</a> and
                    <a href="http://people.eecs.berkeley.edu/~rliaw/">Richard Liaw</a> and
                    <a href="http://people.csail.mit.edu/gehring/">Clement Gehring</a>
img:                assets/functional/rl_lib.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

In this blog post, we explore a functional paradigm for implementing
[reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
(RL) algorithms. The paradigm will be that developers write the numerics of
their algorithm as independent, pure functions, and then use a library to
compile them into _policies_ that can be trained at scale. We share how these
ideas were implemented in [RLlib](https://rllib.io)’s [policy builder
API](https://ray.readthedocs.io/en/latest/rllib-concepts.html#building-policies-in-tensorflow),
eliminating thousands of lines of “glue” code and bringing support for
[Keras](https://ray.readthedocs.io/en/latest/rllib-models.html#tensorflow-models)
and [TensorFlow
2.0](https://ray.readthedocs.io/en/latest/rllib-concepts.html#building-policies-in-tensorflow-eager).

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/1*1EwDu6skRrPkNPx_fzpVbg.png">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img01.png">
<br>
</p>

<!--more-->

### Why Functional Programming?

One of the key ideas behind functional programming is that programs can be
composed largely of pure functions, i.e., functions whose outputs are entirely
determined by their inputs. Here less is more: by imposing restrictions on what
functions can do, we gain the ability to more easily reason about and
manipulate their execution.

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*D11A8FaF53k76olV">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img02.png">
<br>
</p>


In TensorFlow, such functions of tensors can be executed either
**symbolically** with placeholder inputs or **eagerly** with real tensor
values. Since such functions have no side-effects, they have the same effect on
inputs whether they are called once symbolically or many times eagerly.

### Functional Reinforcement Learning

Consider the following loss function over agent rollout data, with current
state $s$, actions $a$, returns $r$, and policy $\pi$:

$$L(s, a, r) = -[\log \pi(s, a)] \cdot r$$

If you’re not familiar with RL, all this function is saying is that we should
try to _improve the probability of good actions_ (i.e., actions that increase
the future returns). Such a loss is at the core of [policy
gradient](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
algorithms. As we will see, defining the loss is almost all you need to start
training a RL policy in RLlib.

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*C410k6WuEQY9ChMF">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img03.png">
<br>
<i>
Given a set of rollouts, the <a href="https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html">policy gradient</a>
loss seeks to improve the probability of good actions (i.e., those that lead to
a win in this Pong example above).
</i>
</p>

A straightforward translation into Python is as follows. Here, the loss
function takes $(\pi, s, a, r)$, computes $\pi(s, a)$ as a discrete action
distribution, and returns the log probability of the actions multiplied by the
returns:

```python
def loss(model, s: Tensor, a:  Tensor, r: Tensor) -> Tensor:
    logits = model.forward(s)
    action_dist = Categorical(logits)
    return -tf.reduce_mean(action_dist.logp(a) * r)
```

There are multiple benefits to this functional definition. First, notice that
loss reads quite naturally — **there are no placeholders, control loops, access
of external variables, or class members** as commonly seen in RL
implementations. Second, since it doesn’t mutate external state, it is
compatible with both TF graph and eager mode execution.

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*GTmsc5AQZbE4f9qY">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img04.png">
<br>
<i>
In contrast to a class-based API, in which class methods can access arbitrary
parts of the class state, a functional API builds policies from loosely coupled
pure functions.
</i>
</p>

In this blog we explore defining RL algorithms as collections of such pure
functions. The paradigm will be that developers write the numerics of their
algorithm as independent, pure functions, and then use a RLlib helper function
to compile them into _policies_ that can be trained at scale. This proposal is
implemented concretely in the RLlib library.

### Functional RL with RLlib

[RLlib](https://ray.readthedocs.io/en/latest/rllib.html) is an open-source
library for reinforcement learning that offers both high scalability and a
unified API for a variety of applications. It offers a [wide range of scalable
RL algorithms](https://ray.readthedocs.io/en/latest/rllib-algorithms.html).

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*2wpxKQ_TBBQW7Lhe">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img05.png">
<br>
<i>
Example of how RLlib scales algorithms, in this case with distributed
synchronous sampling.
</i>
</p>

Given the increasing popularity of PyTorch (i.e., imperative execution) and the
imminent release of TensorFlow 2.0, we saw the opportunity to improve RLlib’s
developer experience with a functional rewrite of RLlib’s algorithms. The major
goals were to:

**Improve the RL debugging experience**

*   Allow eager execution to be used for any algorithm with just an — eager
    flag, enabling easy `print()` debugging.

**Simplify new algorithm development**

*   Make algorithms easier to customize and understand by replacing monolithic
    “Agent” classes with policies built from collections of pure functions
    (e.g., primitives provided by [TRFL](https://github.com/deepmind/trfl)).
*   Remove the need to manually declare tensor placeholders for TF.
*   Unify the way TF and PyTorch policies are defined.

### Policy Builder API

The RLlib policy builder API for functional RL (stable in RLlib 0.7.4) involves
just two key functions:

*   [build\_tf\_policy](https://ray.readthedocs.io/en/latest/rllib-concepts.html#building-policies-in-tensorflow)()
*   [build\_torch\_policy](https://ray.readthedocs.io/en/latest/rllib-concepts.html#building-policies-in-pytorch)()

At a high level, these builders take a number of **function objects** as input,
including a `loss_fn` similar to what you saw earlier, a `model_fn` to return a
neural network model given the algorithm config, and an `action_fn` to generate
action samples given model outputs. The actual API takes quite a few more
arguments, but these are the main ones. The builder compiles these functions
into a
[policy](https://ray.readthedocs.io/en/latest/rllib-concepts.html#policies)
that can be queried for actions and improved over time given experiences:

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*IFYYtLJg-FyGEI77">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img06.png">
<br>
</p>

These policies can be leveraged for single-agent, vector, and multi-agent
training in RLlib, which calls on them to determine how to interact with
environments:

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*nVuy28pbOkgNygaD">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img07.png">
<br>
</p>

We’ve found the policy builder pattern general enough to port almost all of RLlib’s reference algorithms, including [A2C](https://github.com/ray-project/ray/blob/master/rllib/agents/a3c/a3c_tf_policy.py), [APPO](https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/appo_policy.py), [DDPG](https://github.com/ray-project/ray/blob/b520f6141ecdd54496b0c26106f3df4442a5f91e/rllib/agents/ddpg/ddpg_policy.py), [DQN](https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn_policy.py), [PG](https://github.com/ray-project/ray/blob/master/rllib/agents/pg/pg_policy.py), [PPO](https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/appo_policy.py), [SAC](https://github.com/ray-project/ray/blob/master/rllib/agents/sac/sac_policy.py), and [IMPALA](https://github.com/ray-project/ray/blob/master/rllib/agents/impala/vtrace_policy.py) in TensorFlow, and [PG](https://github.com/ray-project/ray/blob/master/rllib/agents/pg/torch_pg_policy.py) / [A2C](https://github.com/ray-project/ray/blob/master/rllib/agents/a3c/a3c_torch_policy.py) in PyTorch. While code readability is somewhat subjective, users have reported that the builder pattern makes it much easier to customize algorithms, especially in environments such as Jupyter notebooks. In addition, these refactorings have reduced the size of the algorithms by up to hundreds of lines of code _each_.

### Vanilla Policy Gradients Example

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*I7SuZh-u1rl3Emfb">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img08.png">
<br>
<i>
Visualization of the vanilla policy gradient loss function in RLlib.
</i>
</p>

Let’s take a look at how the earlier loss example can be implemented concretely
using the builder pattern. We define `policy_gradient_loss`, which requires a
couple of tweaks for generality: (1) RLlib supplies the proper
`distribution_class` so the algorithm can work with any type of action space
(e.g., continuous or categorical), and (2) the experience data is held in a
`train_batch` dict that contains state, action, etc. tensors:

```python
def policy_gradient_loss(
        policy, model, distribution_cls, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = distribution_cls(logits, model)
    return -tf.reduce_mean(
        action_dist.logp(train_batch[“actions”]) *
        train_batch[“returns”])
```

To add the “returns” array to the batch, we need to define a postprocessing
function that calculates it as the [temporally discounted
reward](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#reward-and-return)
over the trajectory:

$$R(\tau) = \sum_{t=0}^{\infty}{\gamma^tr_t}$$

We set $\gamma = 0.99$ when computing $R(T)$ below in code:

```python
from ray.rllib.evaluation.postprocessing import discount

# Run for each trajectory collected from the environment
def calculate_returns(policy,
                      batch,
                      other_agent_batches=None,
                      episode=None):
   batch[“returns”] = discount(batch[“rewards”], 0.99)
   return batch
```

Given these functions, we can then build the RLlib policy and
[trainer](https://ray.readthedocs.io/en/latest/rllib-concepts.html#trainers)
(which coordinates the overall training workflow). The model and action
distribution are automatically supplied by RLlib if not specified:

```python
MyTFPolicy = build_tf_policy(
   name="MyTFPolicy",
   loss_fn=policy_gradient_loss,
   postprocess_fn=calculate_returns)

MyTrainer = build_trainer(
   name="MyCustomTrainer", default_policy=MyTFPolicy)
```

Now we can run this at the desired scale using
[Tune](https://ray.readthedocs.io/en/latest/tune.html), in this example showing
a configuration using 128 CPUs and 1 GPU in a cluster:

```python
tune.run(MyTrainer,
    config={“env”: “CartPole-v0”,
            “num_workers”: 128,
            “num_gpus”: 1})
```

While this example [(runnable
code)](https://github.com/ray-project/ray/blob/master/rllib/examples/custom_tf_policy.py)
is only a basic algorithm, it demonstrates how a functional API can be concise,
readable, and highly scalable. When compared against the previous way to define
policies in RLlib using TF placeholders, **the** **functional API uses ~3x
fewer lines of code (23 vs 81 lines),** and also works in eager:

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/1*hrzYi0u3I6uARF-XvfcLTQ.png">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img09.png">
<br>
<i>
Comparing the <a href="https://github.com/ray-project/ray/blob/75ac016e2bd39060d14a302292546d9dbc49f6a2/python/ray/rllib/agents/pg/pg_policy_graph.py">legacy class-based API</a>
with the new <a href="https://github.com/ray-project/ray/blob/b520f6141ecdd54496b0c26106f3df4442a5f91e/rllib/agents/pg/pg_policy.py">functional policy builder API</a>
Both policies implement the same behaviour, but the functional definition is
much shorter.
</i>
</p>


### How the Policy Builder works

Under the hood, `build_tf_policy` takes the supplied building blocks
(`model_fn`, `action_fn`, `loss_fn`, etc.) and compiles them into either a
[DynamicTFPolicy](https://github.com/ray-project/ray/blob/03a1b758526b2699a21e44a932bb2abdfe636f2b/rllib/policy/dynamic_tf_policy.py)
or
[EagerTFPolicy](https://github.com/ray-project/ray/blob/03a1b758526b2699a21e44a932bb2abdfe636f2b/rllib/policy/eager_tf_policy.py),
depending on if TF eager execution is enabled. The former implements graph-mode
execution (auto-defining placeholders dynamically), the latter eager execution.

The main difference between `DynamicTFPolicy` and `EagerTFPolicy` is how many
times they call the functions passed in. In either case, a `model_fn` is
invoked once to create a Model class. However, functions that involve tensor
operations are either called once in graph mode to build a symbolic computation
graph, or multiple times in eager mode on actual tensors. In the following
figures we show how these operations work together in blue and orange:


<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/1*4RKNH6Zt4-P82ZeoETKVog.png">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img10.png">
<br>
</p>

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*1vTJnByqwRvPoyBV">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img11.png">
<br>
<i>
Overview of a generated EagerTFPolicy. The policy passes the environment state
through model.forward(), which emits output logits. The model output
parameterizes a probability distribution over actions (“ActionDistribution”),
which can be used when sampling actions or training. The loss function operates
over batches of experiences. The model can provide additional methods such as a
value function (light orange) or other methods for computing Q values, etc.
(not shown) as needed by the loss function.
</i>
</p>

This policy object is all RLlib needs to launch and scale RL training.
Intuitively, this is because it encapsulates how to compute actions and improve
the policy. External state such as that of the environment and RNN hidden state
is managed externally by RLlib, and does not need to be part of the policy
definition. The policy object is used in one of two ways depending on whether
we are computing rollouts or trying to improve the policy given a batch of
rollout data:

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*-35jGBA7Gha9WnOA">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img12.gif">
<br>
<i>
<b>Inference:</b> Forward pass to compute a single action. This only involves
querying the model, generating an action distribution, and sampling an action
from that distribution. In eager mode, this involves calling action_fn
<a href="https://github.com/ray-project/ray/blob/03a1b758526b2699a21e44a932bb2abdfe636f2b/rllib/agents/dqn/simple_q_policy.py#L111">
DQN example of an action sampler</a>,
which creates an action distribution / action sampler as relevant that is then
sampled from.
</i>
</p>


<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*w4cJ0KPTM8QPX5Ex">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img13.gif">
<br>
<i>
<b>Training:</b> Forward and backward pass to learn on a batch of experiences.
In this mode, we call the loss function to generate a scalar output which can
be used to optimize the model variables via SGD. In eager mode, both action_fn
and loss_fn are called to generate the action distribution and policy loss
respectively. Note that here we don’t show differentiation through action_fn,
but this does happen in algorithms such as DQN.
</i>
</p>


### Loose Ends: State Management

RL training inherently involves a lot of state. If algorithms are defined using
pure functions, where is the state held? In most cases it can be managed
automatically by the framework. There are three types of state that need to be
managed in RLlib:

1.  **Environment state**: this includes the current state of the environment
and any recurrent state passed between policy steps. RLlib manages this
internally in its [rollout
worker](https://github.com/ray-project/ray/blob/master/rllib/evaluation/rollout_worker.py)
implementation.
2.  **Model state**: these are the policy parameters we are trying to learn via
an RL loss. These variables must be accessible and optimized in the same way
for both graph and eager mode. Fortunately,
[Keras](https://www.tensorflow.org/guide/keras) models can be used in either
mode. RLlib provides a [customizable model class
(TFModelV2)](https://ray.readthedocs.io/en/latest/rllib-models.html#tensorflow-models)
based on the object-oriented Keras style to hold policy parameters.
3.  **Training workflow state**: state for managing training, e.g., the
annealing schedule for various hyperparameters, steps since last update, and so
on. RLlib lets algorithm authors add [mixin
classes](https://github.com/ray-project/ray/blob/b520f6141ecdd54496b0c26106f3df4442a5f91e/rllib/agents/ppo/ppo_policy.py#L284)
to policies that can hold any such extra variables.

### Loose ends: Eager Overhead

Next we investigate RLlib’s eager mode performance with [eager
tracing](https://www.tensorflow.org/beta/tutorials/eager/tf_function) on or
off. As shown in the below figure, tracing greatly improves performance.
However, the tradeoff is that Python operations such as print may not be called
each time. For this reason, tracing is off by default in RLlib, but can be
enabled with “eager\_tracing”: True. In addition, you can also set
“no\_eager\_on\_workers” to enable eager only for learning but disable it for
inference:

<p style="text-align:center;">
<!--
<img src="https://cdn-images-1.medium.com/max/800/0*YvplJKQocQXjclg5">
-->
<img src="https://bair.berkeley.edu/static/blog/functional/img14.png">
<br>
</p>

Eager inference and gradient overheads measured using `rllib train --run=PG
--env=<env> [ --eager [ --trace]]` on a laptop processor. With tracing off, eager
imposes a significant overhead for small batch operations. However it is often
as fast or faster than graph mode when tracing is enabled.


### Conclusion

To recap, in this blog post we propose using ideas from functional programming
to simplify the development of RL algorithms. We implement and validate these
ideas in RLlib. Beyond making it easy to support new features such as eager
execution, we also find the functional paradigm leads to substantially more
concise and understandable code. Try it out yourself with `pip install
ray[rllib]` or by checking out the
[docs](https://ray.readthedocs.io/en/latest/rllib.html) and [source
code](https://github.com/ray-project/ray/tree/master/rllib).

If you’re interested in helping improve RLlib, we’re also [hiring](https://jobs.lever.co/anyscale).
