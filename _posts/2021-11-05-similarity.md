---
layout:             post
title:              "How should we compare neural network representations?"
date:               2021-11-05  9:00:00
author:             <a href="https://francesding.github.io/">Frances Ding</a> and <a href="https://jsteinhardt.stat.berkeley.edu/">Jacob Steinhardt</a>
img:                assets/similarity/PWCCA_avg_double_heatmap.png
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
<meta name="twitter:title" content="Example Post Title">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/example_post/image1.png">

<meta name="keywords" content="similarity metrics, representation learning, interpretability">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Frances Ding, Jacob Steinhardt">

<!--
The actual text for the post content appears below.  Text will appear on the
homepage, i.e., https://bair.berkeley.edu/blog/ but we only show part of the
posts on the homepage. The rest is accessed via clicking 'Continue'. This is
enforced with the `more` excerpt separator.
-->

To understand neural networks, researchers often use **similarity metrics** to measure how similar or different two neural networks are to each other. For instance, they are used to compare vision transformers to convnets \[1\], to understand transfer learning \[2\], and to explain the success of standard training practices for deep models \[3\]. Below is an example visualization using similarity metrics; specifically we use the popular CKA similarity metric (introduced in \[4\]) to compare two transformer models across different layers: 

<p style="text-align:center;">
<img src="https://bounded-regret.ghost.io/content/images/2021/10/CKA_avg_double_heatmap.png" width="75%">
<br>
<i><b>Figure 1.</b> CKA (Centered Kernel Alignment) similarity between two networks trained identically except for random initialization. Lower values (darker colors) are more similar. CKA suggests that the two networks have similar representations.</i>
</p>


Unfortunately, there isn't much agreement on which particular similarity metric to use. Here's the exact same figure, but produced using the Canonical Correlation Analysis (CCA) metric instead of CKA:

<p style="text-align:center;">
<img src="https://bounded-regret.ghost.io/content/images/2021/10/PWCCA_avg_double_heatmap.png" width="75%">
<br>
<i><b>Figure 2.</b> CCA (Canonical Correlation Analysis) similarity between the same two networks. CCA distances suggest that the two networks learn somewhat different representations, especially at later layerss.</i>
</p>


<!--more-->

In the literature, researchers often propose new metrics and justify them based on intuitive desiderata that were missing from previous metrics. For example, Morcos et al. motivate CCA by arguing that similarity metrics should be invariant to invertible linear transformations \[5\]. Kornblith et al. argue that similarity metrics should be invariant to orthogonal transformation and isotropic scaling, but not invertible linear transformation; this motivates their proposed metric, CKA \[4\]. Kornblith et al. then propose an intuitive test for similarity metrics to pass - given two trained networks with the same architecture but different initialization, layers at the same depth should be most similar to each other - and CKA performs the best on their test. 

Our paper, [Grounding Representation Similarity with Statistical Testing](https://arxiv.org/abs/2108.01661), argues against this practice. To start, we show that by choosing different intuitive tests, we can make different methods look good. CKA does well on a "specificity test" similar to the one proposed by Kornblith et al., but it does poorly on a "sensitivity test" that CCA shines on.

To move beyond intuitive tests, our paper provides a carefully-designed quantitative benchmark for evaluting similarity metrics. The basic idea is that a good similarity metric should correlate with the actual **functionality** of a neural network, which we operationalize as accuracy on a task (but other axes of functionality for future work could be modularity, or meta-learning capability, etc.). Why? Accuracy differences between models are a signal that the models are processing data differently, so intermediate representations must be different, and similarity metrics should notice this. 

Thus, for a given pair of neural network representations, we measure both their (dis)similarity and the difference between their accuracies on some task. If these are well-correlated across many pairs of representations, we have a good similarity metric. Of course, a perfect correlation with accuracy on a particular task also isn’t what we’re hoping for, since metrics should capture many important differences between models, not just one. A good similarity metric is one that gets generally high correlations across a couple of functionalities.

We assess functionality with a range of tasks. For a concrete example, one subtask in our benchmark leverages the finding that BERT language models finetuned with different random seeds can have in-distribution accuracy all within a percentage point of each other, but out-of-distribution accuracy ranging from 0 to 60% [6]. We hope that given two robust models, a similarity metric would rate them as similar, and given one robust and one non-robust model, a metric would rate them as dissimilar. Thus we take 100 such BERT models and evaluate whether (dis)similarity between each pair of model representations correlates with their difference in OOD accuracy. 


Our benchmark is composed of many of these subtasks, where we collect model representations that vary along axes such as training seeds or layer depth, and evaluate the models' functionalities. We include the following subtasks:
1. **Varying seeds and layer depths, and assessing functionality through linear probes** (linear classifiers trained on top of a frozen model's intermediate layer)
2. **Varying seeds, layer depths, and principal component deletion, and assessing functionality through linear probes**
3. **Varying finetuning seeds and assessing functionality through OOD test sets** (described above) 
4. **Varying pretraining and finetuning seeds and assessing functionality through OOD test sets** 

You can find the code for our benchmarks [here](https://github.com/js-d/sim_metric).

The table below shows our results with BERT language models (vision model results can be found in the paper). In addition to the popular CKA and (PW)CCA metrics, we considered a classical baseline called the Procrustes distance. Both CKA and PWCCA dominate certain benchmarks and fall behind on others, while Procrustes is more consistent and often close to the leader. In addition, our last subtask is challenging, with no similarity measure achieving high correlation. We present it as a challenge task to motivate further progress for similarity metrics.

![results_table_lang-1](https://bounded-regret.ghost.io/content/images/2021/11/results_table_lang-1.png)
<figcaption> </figcaption>

In the end, we were surprised to see Procrustes do so well since the recent CKA and CCA methods have gotten more attention, and we originally included Procrustes as a baseline for the sake of thoroughness. Building these benchmarks across many different tasks was essential for highlighting Procrustes as a good all-around method, and it would be great to see the creation of more benchmarks that evaluate the capabilities and limitations of other tools for understanding and interpreting neural networks.

For more details, please see our [full paper](https://arxiv.org/abs/2108.01661)!


References

\[1\] [Raghu, Maithra, et al. "Do Vision Transformers See Like Convolutional Neural Networks?."](https://arxiv.org/abs/2108.08810) arXiv preprint arXiv:2108.08810 (2021).

\[2\][Neyshabur, Behnam, Hanie Sedghi, and Chiyuan Zhang. "What is being transferred in transfer learning?."](https://arxiv.org/abs/2008.11687) NeurIPS. 2020. 

\[3\] [Gotmare, Akhilesh, et al. "A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and Distillation."](https://arxiv.org/abs/1810.13243) International Conference on Learning Representations. 2018.

\[4\] [Kornblith, Simon, et al. "Similarity of neural network representations revisited."](https://arxiv.org/abs/1905.00414) International Conference on Machine Learning. PMLR, 2019.
\[5\] [Morcos, Ari S., Maithra Raghu, and Samy Bengio. "Insights on representational similarity in neural networks with canonical correlation."](https://arxiv.org/abs/1806.05759) Proceedings of the 32nd International Conference on Neural Information Processing Systems. 2018.
\[6\] [R. T. McCoy, J. Min, and T. Linzen. Berts of a feather do not generalize together: Large variability in generalization across models with similar test set performance.](https://arxiv.org/abs/1911.02969) Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP, 2020.





<!--The content here after the excerpt separator will not appear on the front page
of the BAIR blog but will show in the post.-->

<!-- # Text formatting

Markdown provides text formatting such as **bold** and *italic*.

LaTeX is also supported, such as $y = \beta x + \alpha$ inline, or as a separate
line

$$y = \beta x + \alpha.$$

URLs can be inserted through square brackets, such as [this][1].

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/example_post/image2.png" width="30%">
<br>
<i><b>Figure title.</b> Figure caption. This image is centered and set to 30%
page width.</i>
</p>
-->

<hr>

<i>This post is based on the paper “Grounding Representation Similarity with Statistical Testing”, to be presented at NeurIPS 2021. You
can see full results [in our paper][1], and we [provide code][2] to to reproduce
our experiments. We thank Juanky Perdomo and John Miller for their valuable feedback on this blog
post.</i>

[1]:https://arxiv.org/abs/2108.01661
[2]:https://github.com/js-d/sim_metric
