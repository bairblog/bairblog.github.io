---
layout:             post
title:              "Accelerating Ukraine Intelligence Analysis with Computer Vision on Synthetic Aperture Radar Imagery"
date:               2022-03-21  12:00:00
author:             <a href="http://ritwikgupta.me/">Ritwik Gupta*</a>, <a href="https://people.eecs.berkeley.edu/~cjrd/">Colorado Reed*</a>, <a href="https://anna-rohrbach.net/">Anja Rohrbach</a>, and <a href="https://people.eecs.berkeley.edu/~trevor/">Trevor Darrell</a> 
img:                assets/maers/ukraine-clouds-optim.gif
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---
<p style="text-align:center;">
    <img src="http://localhost:8000/ukraine-clouds-optim.gif" width="120%">
    <br>
    <i><b>Figure 1:</b> Airmass measurements over Ukraine from February 18, 2022 - March 01, 2022 from the SEVIRI instrument. Data accessed via the <a href="https://view.eumetsat.int/productviewer?v=default">EUMETSAT Viewer</a>.</i>
</p>

Satellite imagery is a critical source of information during the current invasion of Ukraine. Military strategists, journalists, and researchers use this imagery to make decisions, unveil violations of international agreements, and inform the public of the stark realities of war. With Ukraine experiencing a large amount of cloud cover and attacks often occuring during night-time, many forms of satellite imagery are hindered from seeing the ground. Synthetic aperture radar imagery penetrates cloud cover, but requires special training to interpret. Automating this tedious task would enable real-time insights, but current computer vision methods developed on typical RGB imagery do not properly account for the phenomenology of SAR. This leads to suboptimal performance on this critical modality. Improving the access to and availability of SAR-specific methods, codebases, datasets, and pretrained models will benefit intelligence agencies, researchers, and journalists alike during this critical time for Ukraine. 

In this post, we present a baseline method and pretrained models that enable the interchangeable use of RGB and SAR for downstream classification, semantic segmentation, and change detection pipelines.

<!--more-->

## TL;DR

Satellite imagery is a critical source of information during the current invasion of Ukraine. Military strategists, journalists, and researchers use this imagery to make decisions, unveil violations of international agreements, and inform the public of the stark realities of war. With Ukraine experiencing a large amount of cloud cover and attacks often occuring during night-time, many forms of satellite imagery are hindered from seeing the ground.

As such, there has been an increased reliance on a form of satellite imagery known as [Synthetic Aperture Radar (SAR)](https://earthdata.nasa.gov/learn/backgrounders/what-is-sar) to provide visibility in all forms of adverse environmental conditions. With its non-standard and unintuitive phenomenology, hundreds of specially trained SAR analysts have been manually combing through billions of pixels trying to provide actionable intelligence in real-time.

Automating this tedious task enables real-time intelligence and democratizes SAR analysis. However, current computer vision methods developed on typical RGB imagery do not properly account for the phenomenology of SAR, leading to suboptimal performance on this critical modality. Improving the access to and availability of SAR-specific methods, codebases, datasets, and pretrained models will benefit intelligence agencies, researchers, and journalists alike during this critical time for Ukraine. 

In this post, we present a baseline method and pretrained models that enable the interchangeable use of RGB and SAR for downstream classification, semantic segmentation, and change detection pipelines. We are working on transitioning these models to our US government and NGO partners for use during this crisis, and we encourage others in the Computer Vision community to contribute to this critical research area.

## Introduction

We live in a rapidly changing world, one that experiences natural disasters, civic upheaval, war, and all sorts of chaotic events which leave unpredictable—and often permanent—marks on the face of the planet. Understanding this change has historically been difficult. Surveyors were sent out to explore our new reality, and their distributed findings were often noisily integrated into a source of reality. Maintaining a constant state of vigilance has been a goal of mankind since we were able to conceive such a thought, all the way from when [Nadar took the first aerial photograph](https://time.com/longform/aerial-photography-drones-history/) to when [Sputnik 1’s radio signals were used to analyze the ionosphere](https://www.sciencedirect.com/science/article/abs/pii/S0273117715001623).

Vigilance, or to the French, _surveillance_, has been a part of human history for millenia. As with any tool, it has been a double-edged sword. Historically, surveillance without checks and balances has been detrimental to society. Conversely, the proper and responsible surveillance has allowed us to learn deep truths about our world which have resulted in advances in the [scientific](https://www.nasa.gov/mission_pages/icebridge/instruments/index.html) and [humanitarian](https://web.archive.org/web/20211001071654/https://news.un.org/en/story/2006/04/176152-un-launches-new-enhanced-tool-use-satellite-data-fighting-hunger-poverty) domains. With the amount of satellites in orbit today, our understanding of the environment is updated almost daily. We have rapidly transitioned from having very little information to now having more data than we can meaningfully extract knowledge from. Storing this information, let alone understanding, is an engineering challenge that is of growing urgency.

## Machine Learning and Remote Sensing

With [hundreds of terabytes](https://datacenterfrontier.com/terabytes-from-space-satellite-imaging-is-filling-data-centers/) of data being downlinked from satellites to data centers every day, gaining knowledge and actionable insights from that data with manual processing has already become an impossible task. The most widely used form of remote sensing data is electro-optical (EO) satellite imagery. EO imagery is commonplace—anyone who has used Google Maps or similar mapping software has interacted with EO satellite imagery.

Machine learning (ML) on EO imagery is used in a wide variety of scientific and commercial applications. From [improving precipitation predictions](https://journals.ametsoc.org/view/journals/hydr/17/3/jhm-d-15-0075_1.xml), [analyzing human slavery by identifying brick kilns](https://www.sciencedirect.com/science/article/pii/S0924271618300479), to [classifying entire cities to improve traffic routing](https://blog.google/products/maps/google-maps-101-ai-power-new-features-io-2021/), the outputs of ML on EO imagery have been integrated into almost every facet of human society.

<p style="text-align:center;">
    <img src="http://localhost:8000/bridge-eo-kyiv.jpeg" width="100%">
    <br>
    <i><a href="https://www.cnn.com/europe/live-news/ukraine-russia-putin-news-03-03-22/h_ed1c79ce964585a1d044c2dd50e2997a"><b>Figure 2:</b> VHR EO imagery over the Kyiv region as acquired by Maxar on February 28, 2022</a>.</i>
</p>

Commonly used satellite constellations for EO imagery include the [Landsat](https://landsat.gsfc.nasa.gov/) series of satellites operated by the United States Geological Survey and the [Copernicus Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) constellation operated by the European Space Agency. These constellations provide imagery at resolutions between 10-60 meters which is good enough for many use cases, but preclude the observation of finer details.

## The Advent of Very High Resolution, Commercial Electro-Optical Satellite Imagery

Over the last few years, very high resolution (VHR) EO imagery has been made available through a variety of commercial sources. Ranging from between 0.3 - 2.0 meter resolution[^1], companies such as [Planet](https://www.planet.com/), [Maxar](https://www.maxar.com/), [Airbus](https://www.airbus.com/en/products-services/space/earth-observation), and others are providing extremely precise imagery with high revisit rates, [imaging the entire planet every day](https://www.fastcompany.com/40498033/every-day-this-satellite-company-takes-a-snapshot-of-the-entire-planet).

<p style="text-align:center;">
    <img src="http://localhost:8000/maxar-ships.jpeg" width="100%">
    <br>
    <i><a href="https://blog.maxar.com/earth-intelligence/2022/enhancing-maritime-domain-awareness-with-maxars-crows-nest-solution"><b>Figure 3:</b> An example of Maxar VHR EO imagery showing floating production, storage and off-loading units and a tanker</a>.</i>
</p>

The increased resolution provided by VHR imagery enables a litany of downstream use cases. [Erosion can be detected at finer scales](https://onlinelibrary.wiley.com/doi/full/10.1002/ldr.1094), and the [building damage can be classified after natural disasters](https://xview2.org/)**.**

Machine learning methods have had to adapt in response to VHR satellite imagery. With an increased acuity, the amount of pixels and the [amount of classes that can be discerned](http://xviewdataset.org/) has increased by orders of magnitude. Computer vision research has responded by [reducing the computational cost to learn efficient representation of satellite imagery](https://www.nature.com/articles/s41467-021-24638-z), creating [methods to alleviate the increased burden on labelers](https://arxiv.org/abs/2108.09186), and even [engineering large software frameworks](https://arxiv.org/abs/2111.08872) to allow computer vision practitioners to handle this abundant source of imagery.

In general, existing computer vision methods on other, non-aerial RGB imagery [transfer very well](https://arxiv.org/abs/1510.00098) to satellite imagery. This has allowed commercial VHR imagery to be immediately useful with highly accurate results.

## The Problem with Electro-Optical Imagery

For highly turbulent and risky situations such as war and natural disasters, having constant, reliable access to the Earth is paramount.  Unfortunately, EO imagery cannot solve all of our surveillance needs. EO can only detect light sources during daytime, and as it turns out, [nearly 2/3rds of the Earth is covered by clouds at any given time](https://earthobservatory.nasa.gov/images/85843/cloudy-earth). Unless you care about clouds, this blockage of the surface of the planet is problematic when understanding what happens on the ground is of critical importance. Machine learning methods attempt to sidestep this problem by [predicting what the world would look like without clouds](https://hal-enpc.archives-ouvertes.fr/hal-01832797/document). However, the loss of information is fundamentally irrecoverable.

## Synthetic Aperture Radar Imagery

Synthetic aperture radar (SAR) imagery is an active form of remote sensing in which a satellite transmits pulses of microwave radar waves down to the surface of the Earth. These radar waves reflect off the ground and any objects on it and are returned back to the satellite. By processing these pulses over time and space, a SAR image is formed where each pixel is the superposition of different radar scatters.

Radar waves penetrate clouds, and since the satellite is actively producing the radar waves, it illuminates the surface of the Earth even during the night. Synthetic aperture radar has a wide variety of uses, being used to [estimate the roughness of the Earth](https://ieeexplore.ieee.org/abstract/document/134087), [mapping the extent of flooding over large areas](https://unitar.org/about/news-stories/news/unosat-introduces-ai-its-flood-rapid-mapping-operations-benefit-national-disaster-management), and to [detect the presence of illegal fishing vessels in protected waters](https://iuu.xview.us/).

There are multiple SAR satellite constellations in operation at the moment. The [Copernicus Sentinel-1](https://sentinel.esa.int/web/sentinel/missions/sentinel-1) constellation provides imagery to the public at large with resolutions ranging from 10 - 80 meters (10 meter imagery being the most common. Most commercial SAR providers, such as [ICEYE](https://www.iceye.com/) and [Capella Space](https://www.capellaspace.com/), provide imagery down to 0.5 meter resolution. In upcoming launches, other commercial vendors aim to produce SAR imagery with sub-0.5 meter resolution with high revisit rates as satellite constellations grow and government regulations evolve.

<p style="text-align:center;">
    <img src="http://localhost:8000/sar-ukraine-belarus.webp" width="100%">
    <br>
    <i><a href="https://www.wired.co.uk/article/ukraine-russia-satellites"><b>Figure 4:</b> A VHR SAR image provided by Capella Space over the Ukraine-Belarus border</a>.</i>
</p>

## The Wacky World of Synthetic Aperture Radar Imagery

While SAR imagery, at a quick glance, may look very similar to EO imagery, the underlying physics is quite different, which leads to many interesting effects in the imagery product which can be counterintuitive and incompatible with modern computer vision. Three common effects are termed polarization, layover, and multi-path effects.

Radar antennas on SAR satellites often transmit polarized radar waves. The direction of polarization is the orientation of the wave’s electric field. Objects on the ground exhibit different responses to the different polarizations of radar waves. Therefore, SAR satellites often operate in dual or quad-polarization modes, broadcasting horizontally (H) or vertically (V) polarized waves and reading either polarization back, resulting in HH, HV, VH, and VV bands. You can contrast this with RGB bands in EO imagery, but the fundamental physics are different.

<p style="text-align:center;">
    <img src="http://localhost:8000/sentinel1-vv-vh.png" width="100%">
    <br>
    <i><b>Figure 5:</b> Difference between VH (left) and VV (right) polarizations over the same region in Dnipro, Ukraine from Sentinel-1 radiometric terrain corrected imagery. As seen, the radar returns in corresponding local regions are entirely different.</i>
</p>

Layover is an effect of foreshortening in which radar beams reach the top of a structure before they reach the bottom, resulting in the top of the object being presented as overlapping with the bottom. This happens when objects are particularly tall. Visually, tall buildings appear as if they are laying on their side, while mountains will have their peaks intersecting with their bases.

<p style="text-align:center;">
    <img src="http://localhost:8000/capella-layover.jpeg" width="100%">
    <br>
    <i><a href="https://twitter.com/capellaspace/status/1367865023587049474/photo/1"><b>Figure 6</b>: Example of layover in Capella’s VHR SAR imagery.</a> The upper portion of the stadium is intersecting, seemingly, with the parking lot behind it.</i>
</p>

Multi-path effects occur when radar waves reflect off of objects on the ground and incur multiple bounces before returning to the SAR sensor. Multi-path effects result in objects appearing in the imagery in various transformations in the resulting image. This effect can be seen everywhere in SAR imagery, but is particularly noticeable in urban areas, forests, and other dense environments.

<p style="text-align:center;">
    <img src="http://localhost:8000/multipath.png" width="100%">
    <br>
    <i><a href="https://discovery.ucl.ac.uk/id/eprint/10053908/"><b>Figure 7:</b> Example of a multi-path effect on a bridge from oblique SAR imagery</a>.</i>
</p>

Existing computer vision methods that are built on traditional RGB imagery are not built with these effects in mind. Object detectors trained on EO satellite imagery assume that a unique object will only appear once, or that the object will appear relatively similar in different contexts, rather than potentially mirrored or scattered or interwoven with surrounding objects. The very nature of occlusion and the vision principles underlying the assumptions of occlusion in EO imagery do not transfer to SAR. Taken together, existing computer vision techniques can transfer to SAR imagery, but with reduced performance and a set of systematic errors that can be addressed through SAR-specific methodology.

**Computer Vision on SAR Imagery for Ukraine**

Imagery analysts are currently relying on both EO and SAR imagery where available over Ukraine. When EO imagery is available, existing computer vision tooling built for that modality is used to expedite the process of intelligence gathering. However, when only SAR imagery is available, these toolchains cannot be used. Imagery analysts have to resort to manual analysis which is time consuming and can be prone to mistakes.

At Berkeley AI Research, we have created an initial set of methods and models that have learned robust representations for RGB, SAR, and co-registered RGB + SAR imagery from the publicly released [BigEarthNet-MM dataset](https://bigearth.net) and the data from [Capella’s Open Data](https://www.capellaspace.com/community/capella-open-data/), which consists of both RGB and SAR imagery. As such, using our models, imagery analysts are able to interchangeably use co-registered RGB or SAR imagery (or both, when available) for downstream tasks such as image classification, semantic segmentation, object detection, or change detection.

Given that SAR is a phenomenologically different data source than EO imagery, we have found that the Vision Transformer (ViT) is a particularly effective architecture for representation learning with SAR as it removes the scale and shift invariant inductive biases built into convolutional neural networks. Our top performing method, MAERS, for representation learning on RGB, SAR, and co-registered RGB + SAR builds upon the [Masked Autoencoder](https://arxiv.org/abs/2111.06377) (MAE) recently introduced by He et. al., where the network learns to encode the input data by taking a masked version of the data as input, encoding the data, and then learning to decode the data in such a way that it reconstructs the unmasked input data. Contrary to popular [classes of contrastive learning techniques](https://arxiv.org/abs/2002.05709), the MAE does not presuppose certain augmentation invariances in the data that may be incorrect for SAR features. Instead, it solely relies on reconstructing the original input, which is agnostic to RGB, SAR, or co-registered modalities.

<p style="text-align:center;">
    <img src="http://localhost:8000/maers.png" width="100%">
    <br>
    <i><b>Figure 8:</b> (top) A visualization of MAERS to learn an optionally joint representation and encoder that can be used for a (bottom) downstream task, such as object detection on either, or both, modalities.</i>
</p>

Learning representations for RGB, SAR, and co-registered modalities can benefit a range of downstream tasks, such as content-based image retrieval, classification, segmentation, and detection. To demonstrate the effectiveness of our learned representations, we perform experiments on the well-established benchmarks of 1) multi-label classification of co-registered EO and SAR scenes from the [BigEarth-MM](https://bigearth.net/) dataset, and 2) semantic segmentation on the VHR EO and SAR [SpaceNet 6](https://spacenet.ai/sn6-challenge/) dataset.

## Multi-Label Classification on BigEarth-MM

<p style="text-align:center;">
    <img src="http://localhost:8000/maers-benmm.png" width="100%">
    <br>
    <i><b>Figure 9:</b> (left) co-registered Sentinel-2 EO and Sentinel-1 SAR imagery are patchified and used to perform a multi-label classification task as specified by the BigEarth-MM challenge. A linear layer is added to our multi-modal encoder and then fine-tuned end-to-end.</i>
</p>

MAERS begins with a set of ImageNet weights for a ViT-Base encoder. We follow through with the MAERS training process on the BigEarthNet-MM dataset for 20 epochs, reconstructing masked RGB, SAR, and RGB+SAR imagery with the standard MAE masking ratio. We append a single Linear layer to the MAERS encoder and learn the multi-label classification task by 1) linear probing and 2) fine-tuning the entire model, each for 20 epochs. Our results are shown in Table 1. MAERS with linear probing convincingly beats transfer learning from ImageNet at a fraction of the training cost. MAERS with fine-tuning convincingly beats SOTA as presented in the BigEarthNet-MM paper.

<p style="text-align:center;">
    <img src="http://localhost:8000/maers-table.png" width="100%">
    <br>
    <i><b>Table 1:</b> Reported F2 scores on the validation set of BigEarthNet-MM.</i>
</p>

## Semantic segmentation on VHR EO and SAR SpaceNet 6

We then focus our attention on semantic segmentation of buildings as damaged/not-damaged as this particular task is of direct interest to researchers and reporting agencies aiming to understand the scope and severity of Russia’s attacks. We rely on the [SpaceNet6](https://spacenet.ai) dataset as an open and public benchmark to illustrate the effectiveness of our learned representations for building footprint detection with VHR SAR. We use this encoder in tandem with the [UperNet](https://arxiv.org/abs/1807.10221) architecture for semantic segmentation. We demonstrate that we are able to learn robust representations that allow a practitioner to use RGB or SAR imagery interchangeably to accomplish downstream tasks.

[[Plot showing and explaining the semantic segmentation of SpaceNet6]]

These results are preliminary, but compelling. We will follow up this effort with a thorough set of experiments and benchmarks as well as a paper detailing our work. We will aid in the transition of our models to our humanitarian partners to enable them to perform change detection over residential and other civilian areas to enable better tracking of war crimes being committed in Ukraine. 

These models are created with the goal of increasing the efficacy of organizations involved in humanitarian missions that are keeping a watchful eye on the war in Ukraine. However, as with any technology, it is our responsibility to understand how this technology could be misused. Therefore, we have designed these models with input from partners who perform intelligence and imagery analysis in humanitarian settings. By taking their thoughts, comments, and critiques into account, we are releasing a capability we are confident will be used for the good of humanity and with processes which dictate their safe and responsible use.

## Call to Action

As citizens of free democracies who develop technologies which help us make sense of the complicated, chaotic, and counter-intuitive world that we live in, we have a responsibility to act when acts of injustice occur. Our colleagues and friends in Ukraine are facing extreme uncertainties and danger. We possess skills in the cyber domain that can aid in the fight against Russian forces. By focusing our time and efforts, whether that be through targeted research or volunteering our time in [helping keep track of processing times at border crossings](https://ukrainenow.org/), we can make a small dent in an otherwise difficult situation.

We urge our fellow computer scientists to partner with government and humanitarian organizations and listen to their needs as difficult times persist. Simple things can make large differences.

## Model and Weights

The models are not being made publicly accessible at this time. We are releasing our models to qualified researchers and partners through this [form](https://forms.gle/8rB4wvzair1t8qqz9). Full distribution will follow once we have completed a thorough assessment of our models.

## Acknowledgements

Thank you to [Gen. Steve Butow](https://www.diu.mil/team/Steven-Butow) and  [Dr. Nirav Patel](https://scholar.google.com/citations?user=bJ51bBQAAAAJ&hl=en) at the Department of Defense’s [Defense Innovation Unit](https://diu.mil/) for reviewing this post and providing their expertise on the future of commercial SAR constellations.


<!-- Footnotes themselves at the bottom. -->
## Footnotes

[^1]:
     It’s interesting to note that the definition of VHR imagery has changed over time. In the 80s, [20 kilometer resolution was “VHR”](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JC093iC06p06735). Perhaps, in the future, 0.3m resolution imagery will no longer be VHR.