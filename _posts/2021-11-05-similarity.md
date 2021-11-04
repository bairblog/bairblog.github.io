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

To understand neural networks, researchers often use **similarity metrics** to measure how similar or different two neural networks are to each other. For instance, they are used to compare vision transformers to convnets [1], to understand transfer learning [2], and to explain the success of standard training practices for deep models [3]. Below is an example visualization using similarity metrics; specifically we use the popular CKA similarity metric (introduced in [4]) to compare two transformer models across different layers: 

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


This is a template for [BAIR blog][1] posts. Here is an example image.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/example_post/image1.png" width="50%">
<br>
<i><b>Figure title.</b> Figure caption. This image is centered and set to 50%
page width.</i>
</p>

<!--more-->

The content here after the excerpt separator will not appear on the front page
of the BAIR blog but will show in the post.

# Text formatting

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

<hr>

<i>This post is based on the paper “TODO”, to be presented at CONFERENCE 2021. You
can see results [on our website][2], and we [provide code][3] to to reproduce
our experiments. We thank XXX and YYY for their valuable feedback on this blog
post.</i>

[1]:https://bair.berkeley.edu/blog/
[2]:https://www.google.com/
[3]:https://github.com/
