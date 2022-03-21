---
layout:             post
title:              "All You Need is LUV: Unsupervised Collection of Labeled Images Using UV-Fluorescent Markings"
date:               2022-02-23  12:00:00
author:             <a href="https://bthananjeyan.github.io">Brijen Thananjeyan*</a> and <a href="https://kerrj.github.io/">Justin Kerr*</a> 
img:                assets/luv/img1.png
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
<meta name="twitter:title" content="All You Need is LUV: Unsupervised Collection of Labeled Images Using UV-Fluorescent Markings">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/luv/splash.png">

<meta name="keywords" content="semantic segmentation, robot perception, self-supervised learning">
<meta name="description" content="Blog post about labels from ultraviolet">
<meta name="author" content="Brijen Thananjeyan, Justin Kerr">

<!--
The actual text for the post content appears below.  Text will appear on the
homepage, i.e., https://bair.berkeley.edu/blog/ but we only show part of the
posts on the homepage. The rest is accessed via clicking 'Continue'. This is
enforced with the `more` excerpt separator.
-->

<!-- ![Main Image](https://bair.berkeley.edu/static/blog/luv/splash.png) -->
<center><img src="https://bair.berkeley.edu/static/blog/luv/splash.png" alt="Main Image" width="400"/></center>



Large-scale semantic image annotation is a significant challenge for learning-based perception systems in robotics.
<!-- Current approaches often rely on human labelers, which can be expensive, or simulation data, which can visually or physically differ from real data. -->
Supervised learning requires labeled data, and a common approach is for humans to hand-label images with segmentation masks, keypoints, and class labels.
However this is time-consuming, error-prone, and expensive, especially when dense or 3-D annotations are required.
An alternative approach is to use simulated data, where data annotation can be densely and procedurally generated at scale at relatively low cost.
However, simulation data can visually or physically differ from real data. In this blog post, we present Labels from UltraViolet (LUV), a novel framework that enables rapid, labeled data collection in real manipulation environments without human labeling.

<!--more-->

LUV uses an array of ultraviolet lights placed around a manipulation workspace that can be switched automatically. We mark objects or keypoints in the scene with transparent, ultraviolet fluorescent paints that are nearly invisible in visible light but highly visible under ultraviolet radiation. For each physical configuration, LUV takes two images: one with standard lighting and one with the ultraviolet lights turned on. LUV provides precise labels for the standard image by performing color segmentation on the paired ultraviolet image. LUV trains a network on the resulting dataset to make predictions on subsequent scenes under standard lighting.

## UV-Fluorescent Paint and Lighting
LUV relies on paint that is nearly transparent under visible light, but fluoresces under ultraviolet radiation. We leverage this property by painting relevant objects and keypoints. For example, in a cable segmentation perception task, we paint the entire cable with UV-fluorescent paint. In a towel corner detection task, we paint the corners of each towel with UV-fluorescent paint.

![UV Visibility](https://bair.berkeley.edu/static/blog/luv/visibility.png)

We observe that different types of paints have better transparency properties on different materials. Further discussion on different types of paints and their interaction with different material types can be found at [https://sites.google.com/berkeley.edu/luv](https://sites.google.com/berkeley.edu/luv).

![Paints](https://bair.berkeley.edu/static/blog/luv/paints.png)

## Mask Generation

To generate masks, the UV lights are turned on, and if available the ambient white lights turned off. The camera exposure for each sample is found by manually sweeping exposures and selecting the exposure yielding clearest label colors. For scenes with both dark and light painted materials, multiple exposures can be captured and post-processed with HDR to retrieve well exposed labels for all colors. We perform HSV color filtering on the UV images to extract the training labels.

![LUV Labels](https://bair.berkeley.edu/static/blog/luv/luvlabels.png)

## Benefits of UV Labels

Here, we discuss the benefits of the training labels generated by LUV.

### Accurate Labels
A key benefit of LUV is that the ultraviolet fluorescence can improve the quality of labels, especially in visually challenging scenes. For example, labeling the below images manually with towel corners is challenging for human labelers.

![Standard Corners](https://bair.berkeley.edu/static/blog/luv/normal.png)

However, UV fluorescence can accurately identify all of the corners in the image (below).

![UV Corners](https://bair.berkeley.edu/static/blog/luv/uv.png)

### Faster Labels
LUV generates each UV label in less than 200ms on commodity desktop hardware, even when the labels are complex. The cable segmentation task is particularly challenging for human labelers, because the ground-truth cable masks are very complex. We find that the cable images in our cable segmentation dataset take an average of 446 seconds for humans to label, over 2500 times longer. Labeling our entire dataset of 486 training images takes less than two minutes in a single-threaded program, but would take over 60 hours for a human.

### Lower Cost Labels
Data annotation services are a popular solution for labeling images. The total one-time cost of our setup is 282 dollars. Based on Amazon's recommended price of 0.82 dollars per semantic segmentation label on Amazon Mechanical Turk, and using 2 labels per image based on their recommendation for quality, this breaks even with Turk at 167 labeled images.

## Network Training Results
We train a fully-convolutional neural network that predicts the training labels from only the standard images. We present results from three tasks commonly considered in robot learning literature: towel corner detection, cable segmentation, and needle segmentation. We observe that the networks are able to accurately predict these segmentation masks and keypoints on test images.

<!-- ![Network Predictions](https://bair.berkeley.edu/static/blog/luv/preds.png) -->
<center><img src="https://bair.berkeley.edu/static/blog/luv/preds.png" alt="Network Predictions" width="400"/></center>


## Links

**Paper:** [All You Need is LUV: Unsupervised Collection of Labeled Images Using UV-Fluorescent Markings](https://arxiv.org/abs/2203.04566)
Brijen Thananjeyan\*, Justin Kerr\*, Huang Huang, Kishore Srinivas, Joseph E. Gonzalez, Ken Goldberg.

\**these authors contributed equally* 

**Supplementary Material:** [https://sites.google.com/berkeley.edu/luv](https://sites.google.com/berkeley.edu/luv)


