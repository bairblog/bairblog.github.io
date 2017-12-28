---
layout:             post
title:              "Physical Adversarial Examples Against Deep Neural Networks"
date:               2017-12-29 9:00:00
author:             Ivan Evtimov, Kevin Eykholt, Earlence Fernandes, and Bo Li 
img:                /assets/yolo/image1.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

*This post is based on recent research by Ivan Evtimov, Kevin Eykholt, Earlence
Fernandes, Tadayoshi Kohno, Bo Li, Atul Prakash, Amir Rahmati, Dawn Song, and
Florian Tramèr.*

Deep neural networks (DNNs) have enabled great progress in a variety of
application areas, including image processing, text analysis, and speech
recognition. DNNs are also being incorporated as an important component in many
cyber-physical systems. For instance, the vision system of a self-driving car
can take advantage of DNNs to better recognize pedestrians, vehicles, and road
signs. However, recent research has shown that DNNs are vulnerable to
*adversarial examples*: Adding carefully crafted adversarial perturbations to the
inputs can mislead the target DNN into mislabeling them during run time. Such
adversarial examples raise security and safety concerns when applying DNNs in
the real world. For example, adversarially perturbed inputs could mislead the
perceptual systems of an autonomous vehicle into misclassifying road signs, with
potentially catastrophic consequences. 

There have been several techniques proposed to generate *adversarial examples*
and to defend against them. In this blog post we will briefly introduce
state-of-the-art algorithms to generate digital adversarial examples, and
discuss our algorithm to generate **physical** adversarial examples on real
objects under varying environmental conditions. We will also provide an update
on our efforts to generate physical adversarial examples for object detectors.

<!--more-->

# Digital Adversarial Examples

Different methods have been proposed to generate adversarial examples in the
white-box setting, where the adversary has full access to the DNN. The white-box
setting assumes a powerful adversary and thus can help set the foundation for
developing future fool-proof defenses. These methods contribute to understanding
digital adversarial examples.

Goodfellow et al. proposed the [fast gradient method][1] that applies a
first-order approximation of the loss function to construct adversarial samples.

[Optimization][2] based methods have also been proposed to create adversarial
perturbations for targeted attacks.  Specifically, these attacks formulate an
objective function whose solution seeks to maximize the difference between the
true labeling of an input, and the attacker’s desired target labeling, while
minimizing how different the inputs are, for some definition of input
similarity.  In computer vision classification problems, a common measure is the
L2-norm of the input vectors. Often, inputs with low L2 distances will be closer
to each other. Thus, it is possible to compute inputs that are visually very
similar to the human eye, but to a classifier, are very different.

Recent work has examined the [black-box][3] transferability of digital adversarial
examples, generating adversarial examples in black-box settings is also
possible. These techniques involve generating adversarial examples for another
known model in a white-box manner, and then running them against the target
unknown model.


# Physical Adversarial Examples

To better understand these vulnerabilities, there has been extensive research on
how *adversarial examples* may affect DNNs deployed in the *physical world*.

[Kurakin et al.][4] showed that printed adversarial examples can be
misclassified when viewed through a smartphone camera. [Sharif et al.][5]
attacked face recognition systems by printing adversarial perturbations on the
frames of eyeglasses. Their work demonstrated successful physical attacks in
relatively stable physical conditions with little variation in pose,
distance/angle from the camera, and lighting. This contributes an interesting
understanding of physical examples in stable environments.

Our recent work “[Robust physical-world attacks on deep learning models][6]” has
shown physical attacks on **classifiers**. (Check out the [videos][7]
[here][8].) As the next logical step, we show attacks on object **detectors**.
These computer vision algorithms identify relevant objects in a scene and
predict bounding boxes indicating objects’ position and kind. Compared with
classifiers, detectors are more challenging to fool as they process the entire
image and can use contextual information (e.g. the orientation and position of
the target object in the scene) in their predictions.
 
We demonstrate *physical* adversarial examples against the [YOLO][9] detector, a
popular state-of-the-art algorithm with good real-time performance. Our examples
take the form of sticker perturbations that we apply to a real STOP sign. The
following image shows our example physical adversarial perturbation.

<p style="text-align:center;">
<img src="{{site.url}}{{site.baseurl}}/assets/yolo/image1.png" height="450" style="margin: 10px;">
<img src="{{site.url}}{{site.baseurl}}/assets/yolo/image4.png" height="450" style="margin: 10px;">
<br>
</p>

We also perform dynamic tests by recording a video to test out the detection
performance.  As can be seen in the video, the YOLO network does not perceive
the STOP sign in almost all the frames. If a real autonomous vehicle were
driving down the road with such an adversarial STOP sign, it would not see the
STOP, possibly leading to a crash at an intersection. The perturbation we
created is robust to changing distances and angles -- the most commonly changing
factors in a self-driving scenario.

More interestingly, the physical adversarial examples generated for the YOLO
detector are also be able to fool standard [Faster-RCNN][10]. Our demo videos
contains a dynamic test of the physical adversarial example on Faster-RCNN. As
this is a black box attack on Faster-RCNN, the attack is not as successful as it
is in the YOLO case. This is expected behavior. We believe that with additional
techniques (such as ensemble training), the black box attack could be made more
effective.  Additionally, specially optimizing an attack for Faster-RCNN will
yield better results. We are currently working on a paper that explores these
attacks in more detail. The image below is an example of Faster-RCNN not
perceiving the STOP sign.

<p style="text-align:center;">
<img src="{{site.url}}{{site.baseurl}}/assets/yolo/image3.png" height="380" style="margin: 10px;">
<img src="{{site.url}}{{site.baseurl}}/assets/yolo/image2.png" height="380" style="margin: 10px;">
<br>
</p>

In both cases (YOLO and Faster-RCNN), a STOP sign is detected only when the
camera is very close to the sign (about 3 to 4 feet away). In real settings,
this distance is too close for a vehicle to take effective corrective action.
Stay tuned for our upcoming paper that contains more details about the algorithm
and results of physical perturbations against state-of-the-art object detectors.


# Attack Algorithm Overview

This algorithm is based off our earlier work on attacking classifiers.
Fundamentally, we take an optimization approach to generating adversarial
examples. However, our experimental experience indicates that generating robust
physical adversarial examples for detectors requires simulating a larger set of
varying physical conditions than what is needed to fool classifiers. This is
likely because a detector takes much more contextual information into account
while generating predictions. Key properties of the algorithm include the
ability to specify sequences of physical condition simulations, and the ability
to specify the translation invariance property. That is, a perturbation should
be effective no matter where the target object is situated within the scene. As
an object can move around freely in the scene depending on the viewer,
perturbations not optimized for this property will likely break when the object
moves. Our upcoming paper on this topic will contain more details on this
algorithm.


# Potential defenses

Given these adversarial examples in both digital and physical world, potential
defense methods have also been widely studied. Among them, different types of
adversarial training methods are the most effective. [Goodfellow et al.][1]
first proposed adversarial training as an effective way to improve the
robustness of DNNs, and [Tramèr et al.][11] extend it to ensemble adversarial
learning.  [Madry et al.][12] have also proposed robust networks via iterative
training with adversarial examples. To conduct an adversarial training based
defense, a large number of adversarial examples are required. In addition, these
adversarial examples can make the defense more robust if they come from
different models as suggested by work on [ensemble training][11]. The benefit of
ensemble adversarial training is to increase the diversity of adversarial
examples so that the model can fully explore the adversarial example space.
There are other types of defense methods as well, but [Carlini and Wagner][13]
have shown that none of these existing defense method is robust enough given
adaptive attack.

Overall, we are still a long way from finding the optimal defense strategy
against these adversarial examples, and we are looking forward to exploring this
exciting research area. 

<h4> Physical Adversarial Sticker Perturbations for YOLO </h4>
<iframe width="854" height="480" src="https://www.youtube.com/embed/gkKyBmULVvM?cc_load_policy=1" frameborder="0" gesture="media" allow="encrypted-media" allowfullscreen></iframe>

<h4> Physical Adversarial Examples for YOLO (2) </h4>
<iframe width="854" height="480" src="https://www.youtube.com/embed/zSFZyzHdTO0?cc_load_policy=1" frameborder="0" gesture="media" allow="encrypted-media" allowfullscreen></iframe>

<h4> Black box transfer to Faster RCNN of physical adversarial examples generated for YOLO </h4>
<iframe width="854" height="480" src="https://www.youtube.com/embed/_ynduxh4uww?cc_load_policy=1" frameborder="0" gesture="media" allow="encrypted-media" allowfullscreen></iframe>

[1]:https://arxiv.org/abs/1412.6572
[2]:https://arxiv.org/abs/1608.04644
[3]:https://arxiv.org/abs/1605.07277
[4]:https://arxiv.org/abs/1607.02533
[5]:https://www.cs.cmu.edu/~sbhagava/papers/face-rec-ccs16.pdf
[6]:https://arxiv.org/abs/1707.08945
[7]:https://www.youtube.com/watch?v=1mJMPqi2bSQ&feature=youtu.be
[8]:https://www.youtube.com/watch?v=xwKpX-5Q98o&feature=youtu.be
[9]:https://pjreddie.com/darknet/yolo/
[10]:https://github.com/endernewton/tf-faster-rcnn
[11]:https://arxiv.org/abs/1705.07204
[12]:https://pdfs.semanticscholar.org/bcf1/1c7b9f4e155c0437958332507b0eaa44a12a.pdf
[13]:http://nicholas.carlini.com/papers/2017_aisec_breakingdetection.pdf
