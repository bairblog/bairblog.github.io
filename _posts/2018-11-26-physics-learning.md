---
layout:             post
title:              "Physics-Based Learned Design: Teaching a Microscope How to Image"
date:               2018-11-26 9:00:00
author:             <a href="http://people.eecs.berkeley.edu/~kellman/">Michael Kellman</a>,
                    <a href="https://emrahbostan.com/">Emrah Bostan</a>, and
                    <a href="http://www.laurawaller.com/">Laura Waller</a><br>
img:                /assets/physics/figure4.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/physics/figure1_left.JPG" height="270" style="margin: 10px;">
    <img src="http://bair.berkeley.edu/static/blog/physics/figure1_middle.JPG" height="270" style="margin: 10px;">
    <img src="http://bair.berkeley.edu/static/blog/physics/figure1_right.gif" height="180" style="margin: 10px;">
    <br/>
<i>
<b>Figure 1</b>: (left) LED Array Microscope constructed using a standard
commercial microscope and an LED array. (middle) Close up on the LED array dome
mounted on the microscope. (right) LED array displaying patterns at 100Hz.
</i>
</p>

Computational imaging systems marry the design of hardware and image
reconstruction. For example, in optical microscopy, [tomographic][5],
[super-resolution][3], and [phase imaging][4] systems can be constructed from
simple hardware modifications to a commercial hardware (Fig. 1) and
computational reconstruction. For live cell imaging applications, where a high
temporal resolution is required, we are limited in the number of measurements we
can acquire. Naturally, we want to know what are the best measurements to
acquire. In this post, we highlight our latest work that pushes away from
classic optimal design and towards learned design for a
non-linear computational imaging system.

<!--more-->

## Motivation

Standard microscopes usually image the absorption contrast of the sample;
however, most biological cells are weakly absorbing. Stains or dyes can be used
to observe contrast, but this may be prohibitively invasive for live cell
biology. With a computational imaging system it is possible to image other
intrinsic optical properties of the sample such as phase (ie. refractive index)
to provide that strong mechanism for contrast (Fig. 2) and quantitative
information about the sample.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/physics/figure2.png"
         width="600"
         alt="..."/>
    <br/>
<i>
<b>Figure 2</b>: Mouse Fibroblast Cells: (left) absorption contrast image shows poor
contrast, (right) quantitative phase image has strong structural contrast.
</i>
</p>

Using a standard microscope, the illumination source can be replaced with a
programmable LED array to construct an LED array microscope (Fig 1.). Images
taken under different LED’s illumination will encode information about the
sample’s phase information (specifically, different parts of the sample’s
spatial spectrum) into the intensity measurements. A large set of measurements (10s to
100s) can then be combined by solving an inverse problem to quantitatively
reconstruct the phase of the sample, possibly with higher resolution than the
native resolution of the microscope, or in 3D.

Due to the multiple measurement requirement, obtaining high quality
reconstructions is traded off with poor temporal resolution. So when sample
motion is present, poor temporal resolution will degrade image quality due to
motion blur. To improve upon this tradeoff, designing how to capture a smaller
set of measurements would enable high quality reconstruction at an improved
temporal resolution. But what are the best patterns to display on the LED array?
What is the most efficient way to encode information?

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/physics/figure3.gif"
         width="700"
         alt="..."/>
    <br/>
<i>
<b>Figure 3</b>: LED Array Microscope System: (top) Measurement formation
processing using LED array microscope. (bottom) Regularized inverse problem
solved for image reconstruction.
</i>
</p>

Designing the LED array patterns for a set of measurements is complicated. Many
computational imaging systems, including this one, are non-linear in how the
hardware encodes information, how the computation reconstructs information, or
both. Traditional design methods, (e.g. [optimal design][10], 
[spectral analysis][4]) almost all consider design for linear models and linear reconstructions, thus
will not necessarily improve performance for systems when the processes are non-linear. 

In this post, we consider a new computational imaging framework, [Physics-based Learned Design][7] [1], 
that optimizes the hardware system design for the overall
performance of the system given a system model, a computational reconstruction,
and a dataset. Here, we consider using the LED array microscope as the hardware system
and regularized inverse problems as the computational reconstruction for our
computational imaging system. In this setting, the hardware and software of this
system are both non-linear and thus are difficult to design using traditional
criteria. In this work, we learn a measurement design to
reconstruct Quantitative Phase Imaging using fewer measurements than traditional
methods and using very few training examples.

## Traditional Design and Phase Retrieval

Traditional optimal design methods consider minimizing the variance (mean
square error) of an unbiased estimator or equivalently maximizing its
information (the reciprocal of the variance). Usually, methods optimize 
real-valued summary statistic (e.g. trace, determinant) of the variance, especially when estimating several parameters. However, such design methods are only optimal for linear unbiased estimators (e.g.
least squares) and will not necessarily be optimal for non-linear biased
estimators. In our setting, we want to estimate the phase image by solving a regularized
phase retrieval problem (both non-linear and biased).

We want to retrieve the sample’s optical properties, $\mathbf{x}$, from the
non-linear measurements, $\mathbf{y}_k$, by minimizing several data consistency
terms and a prior term, $\mathcal{P}$, (e.g. sparsity, total-variation).

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/physics/equation1.png"
         width="450"
         alt="..."/>
    <br/>
</p>

This type of cost function can be efficiently minimized via gradient-based
iterative optimization (eg. accelerated proximal gradient descent or
alternating direction method of multipliers). Here we choose accelerated
proximal gradient descent.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/physics/algorithm1.png"
         width="450"
         alt="..."/>
    <br/>
<i>
<b>Algorithm 1</b>: Accelerated Proximal Gradient Descent, $\mathbf{x}$ is the
current estimate, $\mu$ is the acceleration term, $\mathbf{w},\mathbf{z}$ are
intermediate variables, and $N$ is the number of iterations. Each iteration
consists of a gradient update (green), proximal update (purple), and an
acceleration update (orange).
</i>
</p>


## Physics-based Network

Our goal is to learn the design (ie. how to best encode information) for the
non-linear system outlined in figure 3. However, as we discussed traditional
design methods will not work, so let us rethink the problem. Consider unrolling
the iterations of the optimizer in Alg. 1 into a network where each layer of the network
is an iteration of the optimizer. Within each layer we have operations which
perform a gradient update, a proximal update, and an acceleration update. Now, 
we have a network, which includes specific operations that
incorporate the non-linear image formation process as well as the sparsity-based
non-linearities in the reconstruction.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/physics/figure4.png"
         alt="..."/>
    <br/>
<i>
<b>Figure 4</b>: Unrolled Physics-based Network: (left) takes several camera
measurements parameterized by the hardware design, $C$, as input and (right)
outputs the reconstructed phase image, $\mathbf{x}^*$, which is compared to the
ground through, $\mathbf{x}'$. Each layer of the network corresponds to an
iteration of the gradient-based image reconstruction. Within each layer, there is a
gradient update (green), a proximal update (purple), and an acceleration update (orange).
</i>
</p>

We are not the first to consider unrolled physics-based networks. A work in
another imaging field considers unrolling the iterations of their image
reconstruction to form a network and then learns quantities in the
reconstruction which replace the proximal update [2]. The past year, the
mechanics of unrolling the iterations of an image reconstruction pipeline 
has rapidly grown in popularity.

## Physics-based Learned Design

For our context, we want to learn how to turn on the LEDs during each
measurement. Specifically, we learn the brightness of each LED, such that the
image reconstruction error is minimized. Now, the design problem can be posed
as a supervised learning problem,

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/physics/equation2.png"
         width="450"
         alt="..."/>
    <br/>
</p>

Where, our loss function is the $L_2$ distance between the reconstructed image,
$\mathbf{x}^*$, with the ground truth, $\mathbf{x}'$, for a dataset of images
over positive LED brightnesses, $C \in \mathbb{R}^{M,N}$ for $N$ LED
brightnesses for $M$ measurements.  Because we incorporate the system physics
model and sparsifying nonlinearities, we only have to learn a few parameters. This enables us to efficiently learn the design using a relatively
small dataset ($100$ examples)! We consider minimizing the loss function using standard
gradient-based methods with a non-negativity constraint and backpropagation to
compute the gradient with respect to the design parameters.

## Experiments

Our experimental context is cell imaging, so we use 100 cell images (100px by
100px) as our dataset (90 training examples / 10 testing examples). We learn the
design parameters for acquiring only two measurements. In figure 5, we compare
the phase image reconstructions using our learned design and the [traditional design][4]
against phase image reconstructed using a [validation method][3]. Using only a fraction of the measurements,
our learned designs can reconstruct a phase image with a similar quality to that of the
validation method, while the traditional design's phase image is degraded in quality.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/physics/figure5.png"
         alt="..."/>
    <br/>
<i>
<b>Figure 5</b>: Result Comparison: Quantitative Phase Images of Mouse
Fibroblast Cells reconstructed using (left) traditional design, (middle) our
Physics-based Learned Design, (right) validation using many measurements. Below,
a cross section comparison of the three reconstructions through the nucleus of a
cell.
</i>
</p>


## Conclusion

Optimal design methods are useful when the system is linear, however, their
design will not necessarily improve the performance when the system is non-linear. We
have shown we can learn the design for a non-linear computational imaging system using supervised learning 
by unrolling the image reconstruction to make a network. Looking forward, we will consider
analyzing larger more complex systems with different measurement constraints, as
well as, structured ways to learn parts of the image reconstruction.

## References

[1] Kellman, M. R., Bostan, E., Repina, N., & Waller, L. (2018). Physics-based
Learned Design: Optimized Coded-Illumination for Quantitative Phase Imaging.
arXiv preprint arXiv:1808.03571.

[2] Hammernik, K., Klatzer, T., Kobler, E., Recht, M. P., Sodickson, D. K.,
Pock, T., & Knoll, F. (2018). Learning a variational network for reconstruction
of accelerated MRI data. Magnetic resonance in medicine, 79(6), 3055-3071.


[3]:https://www.nature.com/articles/nphoton.2013.187
[4]:https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-9-11394
[5]:https://www.osapublishing.org/oe/abstract.cfm?uri=oe-17-1-266
[7]:https://arxiv.org/abs/1808.03571
[10]:https://en.wikipedia.org/wiki/Optimal_design
