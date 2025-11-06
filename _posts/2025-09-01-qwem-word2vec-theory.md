---
layout:             post
title:              "What exactly does word2vec learn?"
date:               2025-09-01 09:00:00
author:             <a href="https://dkarkada.xyz/">Dhruva Karkada</a>, <a href="https://james-simon.github.io/">Jamie Simon</a>, <a href="https://research.google.com/pubs/YasamanBahri.html">Yasaman Bahri</a>, <a href="https://physics.berkeley.edu/people/faculty/michael-deweese">Mike DeWeese</a>
img:                /assets/qwem-word2vec-theory/fig1.c8u1a3E7_Z23iPso.webp
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<!-- twitter -->
<meta name="twitter:title" content="What exactly does word2vec learn? A complete theory">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/qwem-word2vec-theory/fig1.c8u1a3E7_Z23iPso.webp">

<meta name="keywords" content="">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Dhruva Karkada, Jamie Simon, Yasaman Bahri, Mike DeWeese">



What exactly does `word2vec` learn, and how? Answering this question amounts to understanding representation learning in a minimal yet interesting language modeling task. Despite the fact that `word2vec` is a well-known precursor to modern language models, for many years researchers lacked a quantitative and predictive theory describing its learning process. In our new [paper](https://arxiv.org/abs/2502.09863), we finally provide such a theory. We prove that there are realistic, practical regimes in which the learning problem reduces to *unweighted least-squares matrix factorization*. We solve the gradient flow dynamics in closed form; the final learned representations are simply given by PCA.

<div style="width: 100%; margin: 0 auto; text-align: center;">
<p style="text-align:center;">
<img src="{{ site.baseurl }}/assets/qwem-word2vec-theory/fig1.c8u1a3E7_Z23iPso.webp" width="100%">
<br>
<i style="font-size: 0.9em;"><a href="https://arxiv.org/abs/2502.09863" target="_blank"><strong>Learning dynamics of word2vec</strong></a>. When trained from small initialization, word2vec learns in discrete, sequential steps. Left: rank-incrementing learning steps in the weight matrix, each decreasing the loss. Right: three time slices of the latent embedding space showing how embedding vectors expand into subspaces of increasing dimension at each learning step, continuing until model capacity is saturated.</i>
</p>
</div>


<!--more-->

Before elaborating on this result, let's motivate the problem. `word2vec` is a well-known algorithm for learning dense vector representations of words. These embedding vectors are trained using a contrastive algorithm; at the end of training, the semantic relation between any two words is captured by the angle between the corresponding embeddings. In fact, the learned embeddings empirically exhibit striking linear structure in their geometry: linear subspaces in the latent space often encode interpretable concepts such as gender, verb tense, or dialect. This so-called *linear representation hypothesis* has recently garnered a lot of attention since [LLMs exhibit this behavior as well](https://arxiv.org/abs/2311.03658), enabling [semantic inspection of internal representations](https://arxiv.org/abs/2309.00941) and providing for [novel model steering techniques](https://arxiv.org/abs/2310.01405). In `word2vec`, it is precisely these linear directions that enable the learned embeddings to complete analogies (e.g., "man : woman :: king : queen") via embedding vector addition.

Maybe this shouldn't be too surprising: after all, the `word2vec` algorithm simply iterates through a text corpus and trains a two-layer linear network to model statistical regularities in natural language using self-supervised gradient descent. In this framing, it's clear that `word2vec` is a minimal neural language model. Understanding `word2vec` is thus a prerequisite to understanding feature learning in more sophisticated language modeling tasks.

## The Result

With this motivation in mind, let's describe the main result. Concretely, suppose we initialize all the embedding vectors randomly and very close to the origin, so that they're effectively zero-dimensional. Then (under some mild approximations) the embeddings collectively learn one "concept" (i.e., orthogonal linear subspace) at a time in a sequence of discrete learning steps. 

It's like when diving head-first into learning a new branch of math. At first, all the jargon is muddled — what's the difference between a function and a functional? What about a linear operator vs. a matrix? Slowly, through exposure to new settings of interest, the words separate from each other in the mind and their true meanings become clearer.

As a consequence, each new realized linear concept effectively increments the rank of the embedding matrix, giving each word embedding more space to better express itself and its meaning. Since these linear subspaces do not rotate once they're learned, these are effectively the model's learned features. Our theory allows us to compute each of these features a priori in *closed form* – they are simply the eigenvectors of a particular target matrix which is defined solely in terms of measurable corpus statistics and algorithmic hyperparameters.

### What are the features?

The answer is remarkably straightforward: the latent features are simply the top eigenvectors of the following matrix:

$$M^{\star}_{ij} = \frac{P(i,j) - P(i)P(j)}{\frac{1}{2}(P(i,j) + P(i)P(j))}$$

where $i$ and $j$ index the words in the vocabulary, $P(i,j)$ is the co-occurrence probability for words $i$ and $j$, and $P(i)$ is the unigram probability for word $i$ (i.e., the marginal of $P(i,j)$). 

Constructing and diagonalizing this matrix from the Wikipedia statistics, one finds that the top eigenvector selects words associated with celebrity biographies, the second eigenvector selects words associated with government and municipal administration, the third is associated with geographical and cartographical descriptors, and so on.

The takeaway is this: during training, `word2vec` finds a sequence of optimal low-rank approximations of $M^{\star}$. It's effectively equivalent to running PCA on $M^{\star}$.

The following plots illustrate this behavior.

<div style="width: 100%; margin: 20px auto; text-align: center;">
<p style="text-align:center;">
<img src="{{ site.baseurl }}/assets/qwem-word2vec-theory/fig2.C4kWlUSu_ZJTCeE.webp" width="100%">
<br>
<i style="font-size: 0.9em;">Learning dynamics comparison showing discrete, sequential learning steps.</i>
</p>
</div>

On the left, the key empirical observation is that `word2vec` (plus our mild approximations) learns in a sequence of essentially discrete steps. Each step increments the effective rank of the embeddings, resulting in a stepwise decrease in the loss. On the right, we show three time slices of the latent embedding space, demonstrating how the embeddings expand along a new orthogonal direction at each learning step. Furthermore, by inspecting the words that most strongly align with these singular directions, we observe that each discrete "piece of knowledge" corresponds to an interpretable topic-level concept. These learning dynamics are solvable in closed form, and we see an excellent match between the theory and numerical experiment.

What are the mild approximations? They are: 1) quartic approximation of the objective function around the origin; 2) a particular constraint on the algorithmic hyperparameters; 3) sufficiently small initial embedding weights; and 4) vanishingly small gradient descent steps. Thankfully, these conditions are not too strong, and in fact they're quite similar to the setting described in the original `word2vec` paper.

Importantly, none of the approximations involve the data distribution! Indeed, a huge strength of the theory is that it makes no distributional assumptions. As a result, the theory predicts exactly what features are learned in terms of the corpus statistics and the algorithmic hyperparameters. This is particularly useful, since fine-grained descriptions of learning dynamics in the distribution-agnostic setting are rare and hard to obtain; to our knowledge, this is the first one for a practical natural language task.

As for the approximations we do make, we empirically show that our theoretical result still provides a faithful description of the original `word2vec`. As a coarse indicator of the agreement between our approximate setting and true `word2vec`, we can compare the empirical scores on the standard analogy completion benchmark: `word2vec` achieves 68% accuracy, the approximate model we study achieves 66%, and the standard classical alternative (known as PPMI) only gets 51%. Check out our paper to see plots with detailed comparisons.

To demonstrate the usefulness of the result, we apply our theory to study the emergence of abstract linear representations (corresponding to binary concepts such as masculine/feminine or past/future). We find that over the course of learning, `word2vec` builds these linear representations in a sequence of noisy learning steps, and their geometry is well-described by a spiked random matrix model. Early in training, semantic signal dominates; however, later in training, noise may begin to dominate, causing a degradation of the model's ability to resolve the linear representation. See our paper for more details.

All in all, this result gives one of the first complete closed-form theories of feature learning in a minimal yet relevant natural language task. In this sense, we believe our work is an important step forward in the broader project of obtaining realistic analytical solutions describing the performance of practical machine learning algorithms.

**Learn more about our work: [Link to full paper](https://arxiv.org/abs/2502.09863)**

---

*This post originally appeared on [Dhruva Karkada's blog](https://dkarkada.xyz/posts/qwem/).*