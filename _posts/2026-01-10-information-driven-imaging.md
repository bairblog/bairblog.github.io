---
layout:             post
title:              "Information-Driven Design of Imaging Systems"
date:               2026-01-10  09:00:00
author:             <a href="https://henrypinkard.github.io/">Henry Pinkard</a>, <a href="https://Lakabuli.github.io/">Leyla Kabuli</a>, <a href="https://emarkley.github.io/">Eric Markley</a>, Tiffany Chien, <a href="https://people.eecs.berkeley.edu/~jiantao/">Jiantao Jiao</a>, <a href="https://www2.eecs.berkeley.edu/Faculty/Homepages/waller.html">Laura Waller</a><br>
img:                /assets/information-driven-imaging/info_estimation_overview.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      True
---

<!--
These are comments in HTML. The above header text is needed to format the
title, authors, etc. The "information-driven-imaging" is the representative image
that we use for each post for tweeting (see below as well) and for the
emails to subscribers.

The `static/blog` directory is a location on the blog server which permanently
stores the images/GIFs in BAIR Blog posts. Each post has a subdirectory under
this for its images (titled `information-driven-imaging` here).

Keeping the post visibility as False will mean the post is only accessible if
you know the exact URL.
-->

<!-- twitter -->
<meta name="twitter:title" content="Measuring What Matters: Information-Driven Design of Imaging Systems">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="/assets/information-driven-imaging/info_estimation_overview.png">

<meta name="keywords" content="information theory, computational imaging, optical design, imaging systems, machine learning, optimization">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Henry Pinkard, Leyla Kabuli, Eric Markley, Tiffany Chien, Jiantao Jiao, Laura Waller">

Many imaging systems produce measurements that humans never see or cannot interpret directly. Your smartphone processes raw sensor data through algorithms before producing the final photo. MRI scanners collect frequency-space measurements that require reconstruction before doctors can view them. Self-driving cars process camera and LiDAR data directly with neural networks.

What matters in these systems is not how measurements look, but how much useful information they contain. AI can extract this information even when it is encoded in ways that humans cannot interpret.

And yet we rarely evaluate information content directly. Traditional metrics like resolution and signal-to-noise ratio assess individual aspects of quality separately, making it difficult to compare systems that trade off between these factors. The common alternative, training neural networks to reconstruct or classify images, conflates the quality of the imaging hardware with the quality of the algorithm.

<!--more-->

We developed a framework that enables direct evaluation and optimization of imaging systems based on their information content. In our [NeurIPS 2025 paper][paper], we show that this information metric predicts system performance across four imaging domains, and that optimizing it produces designs that match state-of-the-art end-to-end methods while requiring less memory, less compute, and no task-specific decoder design.

<p style="text-align:center;">
<img src="/assets/information-driven-imaging/info_estimation_overview.png" width="100%">
<br>
<i>An encoder (optical system) maps objects to noiseless images, which noise corrupts into measurements. Our information estimator uses only these noisy measurements and a noise model to quantify how well measurements distinguish objects.</i>
</p>

## Why mutual information?

Mutual information quantifies how much a measurement reduces uncertainty about the object that produced it. Two systems with the same mutual information are equivalent in their ability to distinguish objects, even if their measurements look completely different.

This single number captures the combined effect of resolution, noise, sampling, and all other factors that affect measurement quality. A blurry, noisy image that preserves the features needed to distinguish objects can contain more information than a sharp, clean image that loses those features.

<p style="text-align:center;">
<img src="/assets/information-driven-imaging/noise_res_spectrum.png" width="90%">
<br>
<i>Information unifies traditionally separate quality metrics. It accounts for noise, resolution, and spectral sensitivity together rather than treating them as independent factors.</i>
</p>

Previous attempts to apply information theory to imaging faced two problems. The first approach treated imaging systems as unconstrained communication channels, ignoring the physical limitations of lenses and sensors. This produced wildly inaccurate estimates. The second approach required explicit models of the objects being imaged, limiting generality.

Our method avoids both problems by estimating information directly from measurements.

## Estimating information from measurements

Estimating mutual information between high-dimensional variables is notoriously difficult. Sample requirements grow exponentially with dimensionality, and estimates suffer from high bias and variance.

However, imaging systems have properties that enable decomposing this hard problem into simpler subproblems. Mutual information can be written as:

$$I(X; Y) = H(Y) - H(Y \mid X)$$

The first term, $H(Y)$, measures total variation in measurements from both object differences and noise. The second term, $H(Y \mid X)$, measures variation from noise alone.

<p style="text-align:center;">
<img src="/assets/information-driven-imaging/entropies_decomposition.png" width="70%">
<br>
<i>Mutual information equals the difference between total measurement variation and noise-only variation.</i>
</p>

Imaging systems have well-characterized noise. Photon shot noise follows a Poisson distribution. Electronic readout noise is Gaussian. This known noise physics means we can compute $H(Y \mid X)$ directly, leaving only $H(Y)$ to be learned from data.

For $H(Y)$, we fit a probabilistic model (e.g. a transformer or other autoregressive model) to a dataset of measurements. The model learns the distribution of all possible measurements. We tested three models spanning efficiency-accuracy tradeoffs: a stationary Gaussian process (fastest), a full Gaussian (intermediate), and an autoregressive PixelCNN (most accurate). The approach provides an upper bound on true information; any modeling error can only overestimate, never underestimate.

## Validation across four imaging domains

Information estimates should predict decoder performance if they capture what limits real systems. We tested this relationship across four imaging applications.

<p style="text-align:center;">
<img src="/assets/information-driven-imaging/applications_figure.png" width="100%">
<br>
<i>Information estimates predict decoder performance across color photography, radio astronomy, lensless imaging, and microscopy. Higher information consistently produces better results on downstream tasks.</i>
</p>

**Color photography.** Digital cameras encode color using filter arrays that restrict each pixel to detect only certain wavelengths. We compared three filter designs: the traditional Bayer pattern, a random arrangement, and a learned arrangement. Information estimates correctly ranked which designs would produce better color reconstructions, matching the rankings from neural network demosaicing without requiring any reconstruction algorithm.

**Radio astronomy.** Telescope arrays achieve high angular resolution by combining signals from sites across the globe. Selecting optimal telescope locations is computationally intractable because each site's value depends on all others. Information estimates predicted reconstruction quality across telescope configurations, enabling site selection without expensive image reconstruction.

**Lensless imaging.** Lensless cameras replace traditional optics with light-modulating masks. Their measurements bear no visual resemblance to scenes. Information estimates predicted reconstruction accuracy across a lens, microlens array, and diffuser design at various noise levels.

**Microscopy.** LED array microscopes use programmable illumination to generate different contrast modes. Information estimates correlated with neural network accuracy at predicting protein expression from cell images, enabling evaluation without expensive protein labeling experiments.

In all cases, higher information meant better downstream performance.

## Designing systems with IDEAL

Information estimates can do more than evaluate existing systems. Our Information-Driven Encoder Analysis Learning (IDEAL) method uses gradient ascent on information estimates to optimize imaging system parameters.

<p style="text-align:center;">
<img src="/assets/information-driven-imaging/IDEAL_overview.png" width="100%">
<br>
<i>IDEAL optimizes imaging system parameters through gradient feedback on information estimates, without requiring a decoder network.</i>
</p>

The standard approach to computational imaging design, end-to-end optimization, jointly trains the imaging hardware and a neural network decoder. This requires backpropagating through the entire decoder, creating memory constraints and potential optimization difficulties.

IDEAL avoids these problems by optimizing the encoder alone. We tested it on color filter design. Starting from a random filter arrangement, IDEAL progressively improved the design. The final result matched end-to-end optimization in both information content and reconstruction quality.

<p style="text-align:center;">
<img src="/assets/information-driven-imaging/IDEAL_perf.png" width="50%">
<br>
<i>IDEAL matches end-to-end optimization performance while avoiding decoder complexity during training.</i>
</p>

## Implications

Information-based evaluation creates new possibilities for rigorous assessment of imaging systems in real-world conditions. Current approaches require either subjective visual assessment, ground truth data that is unavailable in deployment, or isolated metrics that miss overall capability. Our method provides an objective, unified metric from measurements alone.

The computational efficiency of IDEAL suggests possibilities for designing imaging systems that were previously intractable. By avoiding decoder backpropagation, the approach reduces memory requirements and training complexity. We explore these capabilities more extensively in [follow-on work][idealio].

The framework may extend beyond imaging to other sensing domains. Any system that can be modeled as deterministic encoding with known noise characteristics could benefit from information-based evaluation and design, including electronic, biological, and chemical sensors.

<hr>

*This post is based on our NeurIPS 2025 paper ["Information-driven design of imaging systems"][paper]. Code is available on [GitHub][code]. A video summary is available on the [project website][website].*

**BibTeX:**
```
@article{pinkard2024informationdrivendesignimagingsystems,
  title={Information-driven design of imaging systems},
  author={Henry Pinkard and Leyla Kabuli and Eric Markley and Tiffany Chien and Jiantao Jiao and Laura Waller},
  year={2024},
  eprint={2405.20559},
  archivePrefix={arXiv},
  primaryClass={physics.optics},
  url={https://arxiv.org/abs/2405.20559},
}
```

[paper]: https://arxiv.org/abs/2405.20559
[code]: https://github.com/Waller-Lab/EncodingInformation
[website]: https://waller-lab.github.io/EncodingInformationWebsite/
[idealio]: https://arxiv.org/abs/2507.07789
