---
layout:             post
title:              "CVPR 2019 Challenges on Domain Adaptation in Autonomous Driving"
date:               2019-03-25 9:00:00
author:             <a href="https://www.yf.io/">Fisher Yu</a>
img:                assets/bair-commons/berkeley_way-west_web_1.jpg
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

We all dream of a future in which autonomous cars can drive us to every corner
of the world. Numerous researchers and companies are working day and night to
chase this dream by overcoming scientific and technological barriers. One of the
greatest challenges we still face is developing machine learning models that can
be trained in a local environment and also perform well in new, unseen
situations. For example, self-driving cars may utilize perception models to
recognize drivable areas from images. Companies in Silicon Valley can build and
perfect such a model using large local datasets from the Bay Area for training.
However, if the same model were deployed in a snowy area such as Boston, it
would likely perform miserably, because it has never seen snow before. Boston,
during winter, and Silicon Valley, during any time of the year, can be labeled
as separate domains for perception models, since they present clear differences
in climate and challenges in perception. In other cases, domains may be much
closer in nature, such as a city street and a nearby highway. The process of
transferring knowledge and models between different domains in machine learning
is called domain adaptation.

A large number of papers on domain adaptation of perception models have appeared
in top publishing venues for machine learning and computer vision. However, most
of these works focus on image classification and semantic segmentation. Hardly
any attention has been paid to instance-level tasks, such as object detection
and tracking, even though localization of nearby objects is arguably more
important for autonomous driving. To foster the study of domain adaptation of
perception models, Berkeley DeepDrive and Didi Chuxing are co-hosting two
competitions in <a href="https://sites.google.com/view/wad2019">CVPR 2019
Workshop on Autonomous Driving</a>. The challenges will focus on domain
adaptation of object detection and tracking based on the BDD100K, from Berkeley
DeepDrive, and D<sup>2</sup>-City, from Didi Chuxing, datasets. The domain of
BDD100K covers US scenes, while D<sup>2</sup>-City was collected on China’s
streets. The competitions ask participants to transfer object detectors from
BDD100K to D<sup>2</sup>-City and object trackers from D<sup>2</sup>-city to
BDD100K. More information about the challenges can be found on <a
href="https://bdd-data.berkeley.edu/wad-2019.html">our website<a> and <a
href="https://outreach.didichuxing.com/d2city">D<sup>2</sup>-City</a>.

<!--
<iframe width="100%" height="56%"
src="https://www.youtube.com/embed/X5RjE--TUGs" frameborder="0"
allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen>
</iframe>
-->

{% include youtubePlayer.html id="X5RjE--TUGs" %} <br>

Following our introduction of the BDD100K dataset, we have been busy working to
provide more temporal annotations. Above is an example of object tracking
annotation, created by our open-source annotation platform <a
href="https://www.scalabel.ai">Scalabel</a>. Some of the tracking labels are
used in the domain adaptation challenge for object tracking. More data will be
released this summer. Of course, we also have object tracking at night.

<!--
<iframe width="100%" height="56%"
src="https://www.youtube-nocookie.com/embed/gA7dvJW_il0" frameborder="0"
allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen>
</iframe>
-->

{% include youtubePlayer.html id="gA7dvJW_il0" %}
