---
layout:     post
title:      "The Confluence of Geometry and Learning for 3D Understanding"
date:       2017-07-11 9:00:00
author:     Shubham Tulsiani and Tinghui Zhou
visible:    True
excerpt_separator: <!--more-->
---

Given only a single image, humans are able to infer the rich 3D structure present in the underlying scene. Since the task of inferring 3D-from-2D is mathematically under-constrained (see e.g. the left figure below), we must rely on learning from our past visual experiences. These visual experiences solely consist of 2D projections (as received on the retina) of the 3D world. Therefore, the learning signal for our 3D perception capability likely comes from making consistent connections among different perspectives of the world that only capture *partial* evidence of the 3D reality. In this post, we describe methods for building 3D predictions systems that can learn in a similar manner.

<!--
Interestingly, this problem is under-constrained, since an image could be the projection of an infinite number of 3D geometrical entities (see e.g. the famous "workshop metaphor" below).
 -->
 
<!--
![The workshop metaphor](https://i.imgur.com/3uXL86o.png) | ![MVS](https://i.imgur.com/P2b8u7v.png)
-->
![sinha](https://people.eecs.berkeley.edu/~tinghuiz/bair_blog/sinha.png =400x) | ![MVS](https://i.imgur.com/P2b8u7v.png)
:-------------------------:|:-------------------------:
An image could be the projection of infinitely many 3D structures [Sinha & Adelson 1993]. | Our visual experiences are solely comprised of 2D projections of the 3D world.

Building computational models for single image 3D inference is a long-standing problem in computer vision. Early attempts, such as the *Blocks World* [1] or 3D surface from line drawings [2], leveraged explicit reasoning over geometric cues to optimize for the 3D structure. Over the years, the incorporation of supervised learning allowed approaches to scale to more realistic settings and infer qualitative [3] or quantitative [4] 3D representations. The trend of obtaining impressive results in realistic settings has since continued to the current CNN-based incarnations [5,6], but at the cost of increasing reliance on direct 3D supervision, making this paradigm rather restrictive. It is costly and painstaking, if not impossible, to obtain such supervision at a large scale. Instead, akin to the human visual system, we want our computational systems to **learn 3D prediction without requiring 3D supervision**.

<!--Early attempts (such as the "blocks world" by Larry Roberts in the 1960s and 3D surface from 2D line drawings by Barrow and Tenenbaum in the 1980s) focus on optimization methods that heavily rely on hand-coded constraints and/or simplifying assumptions about the world that are mostly effective on toy/synthetic data only. Learning-based methods, with the potential of overcoming the above drawbacks, however, did not become prevalent until the early 2000s when impressive results on real images are demonstrated, including photo pop-up using qualitative geometry [~\cite{derek}] and
-->
<!--
Beyond being of an academic interest, this is practically desirable. 
"Conventional learning methods which predict qualitative [~\cite{derek}] or quantitative[~\cite{Saxena2005}] 3D representations, including the current CNN-based incarnations[~\cite{eigenFergus}] require direct supervision for ground-truth 3D. This is costly and painstaking, if not impossible, to obtain in large scale.”
Conventional learning methods (e.g. the seminal Make3D work from Saxena, Sun & Ng 2008) require direct supervision for ground-truth 3D, which is costly and painstaking, if not impossible, to obtain in large scale. Instead, we want to **learn 3D prediction without requiring 3D supervision**. 
-->

With this goal in mind, our work and several other recent approaches [7-11] explore another form of supervision: multi-view observations, for learning single-view 3D. Interesingly, not only do these different works share the goal of incorporating multi-view supervision, the methodologies used also follow common principles. A unifying foundation to these approaches is the interaction between learning and geometry, where predictions made by the learning system are encouraged to be 'geometrically consistent' with the multi-view observations. Therefore, geometry acts as a bridge between the learning system and the multi-view training data. 

<!-- Towards this, it is relatively easier to obtain "multi-view" observations i.e. how the world looks from different perspectives. Using such multi-view supervision for the 3D-from-2D tasks would allow learning 3D inference in numerous scenarios. --> 

In this blog post, we provide an overview of this recent trend and in particular, highlight the common principles across approaches. We then focus on two papers from the Berkeley Artificial Intelligence Research (BAIR) lab to appear in CVPR 2017 and discuss how these push the multi-view supervision paradigm further.

<!--more-->

<!--
So how do humans learn to perceive 3D from a single image? Noticing that our visual experience solely consists of 2D projections (as received on the retina) of the world, one hypothesis is that we learn to build mental 3D models that are *consistent* with our observations when moving around in the world. In other words, the learning signal for our 3D perception capability comes from making consistent connections among different views of the world that only capture *partial* evidence of the 3D reality.
-->
<!--
While recovering 3D structure from 2D images has been a long-standing problem in computer vision, it is only recently that approaches for learning single-view 3D with multi-vew supervision have started to emerge. In this blog post, we aim to provide an overview of this recent trend, and particularly focus on two papers from the Berkeley Artificial Intelligence Research (BAIR) lab to appear in CVPR 2017.
-->
<!--
## Multi-view Supervision for Single-view Prediction
The aim in this line of work is to learn a *Predictor* $P$ (typically a neural network) that can infer 3D from a single 2D image. At a first glance, it tempting to frame this as a standard supervised  learning task. Let's see how that would proceed -
-->
<!--
- Pick a random image $I$, with desired output $y_{gt}$ from a training dataset $T$.
- Predict $y = P(I)$. Update $P$, using gradient descent, to make $y$ more like $y_{gt}$.
- Repeat until convergence.
-->
<!--
And we are done!
Or are we?
-->
<!--
A crucial assumption in the steps above is that we have a training dataset with ground-truth 3D $y_{gt}$ for each image $I$. However, it is often impossible to obtain this supervision, making the setup described impractical. Instead, **we need to learn 3D prediction without requiring 3D supervision**.
In contrast to direct supervision for ground-truth 3D, it is relatively easy to obtain 'multi-view' observations i.e. how the world looks from different perspectives. It should be feasible to leverage this more easily available form of supervision to learn 3D prediction - after all, this is what we humans exploit to learn about 3D structure!
A recent drive in the computer vision community has been to leverage such multi-view supervision for diverse 3D-from-2D prediction tasks. A common foundation to these approaches is the interaction between learning and geometry. **While learning provides the means to succintly capture the information in training data, geometry provides the  language through which this information is communicated to the learning system**.
Let us now examine how this combination of geometry and learning allows us to leverage multi-view supervision for 3D-from-2D tasks.
-->

<!-- Example for direct vs multi-view supervision -->
<!--After all, this is the form of supervision that we, as humans, exploit to learn about 3D structure! -->


## Learning via Geometric Consistency
<!--a) Setup : Predictor P, multi-view training data T. Geometry allows us to define loss by measuring consistency.
b) Consider a game between P, V.
-->
Our aim is to to learn a *Predictor* $P$ (typically a neural network) that can infer 3D from a single 2D image. Under the supervision setting considered, the training data $T$ consists of multiple observations from different viewpoints. As alluded to earlier, geometry acts as a bridge to allow learning the *Predictor* $P$ using the training data $T$. This is because we know precisely, in the form of concise geometric equations, the relationship between a 3D representation and the corresponding 2D projections.  We can therefore train $P$ to predict 3D that is *geometrically consistent* with the associated 2D observations (from $T$).

<!--
Given a 3D shape and a 2D image, we can verify whether they adhere to these equations i.e. whether the 3D structure is *geometrically consistent* with the 2D observation. 
-->
![](https://i.imgur.com/KT2s75c.png)

To illustrate the training process, consider a simple game between the *Predictor* $P$ and a geometry expert, the *Verifier* $V$. We give $P$ a single image $I$, and it predicts a 3D representation $y$. $V$, who is then given the prediction $y$, and an observation $O$ of the world from a different camera viewpoint $C$, uses the geometric equations to validate if these are consistent. We ask $P$ to predict $y$ that would pass this consistency check performed by $V$. The key insight is that since $P$ does not know $(O, C)$ which will be used to verify its prediction, it will have to predict $y$ that is consistent will *all* the possible observations (similar to the unknown ground-truth $y_{gt}$). This allows us to define the following training algorithm to learn 3D-from-2D prediction using only multi-view supervision.

- From $T$, pick random image $I$, with associated observation $O$ from viewpoint $C$.
- Predict $y = P(I)$. Use $V$ to check consistency between $(y, O, C)$
- Update $P$, using gradient descent, to make $y$ more consistent with $(O, C)$.
- Repeat until convergence.

The recent approaches pursuing single-view prediction using multi-view supervision all adhere to this template, the differences being the form of 3D prediction being pursued (e.g. depth or shape), and the kinds of multi-view observations needed (e.g. color images or foreground masks). We now look at two papers which push the boundaries of the multi-view supervision paradigm. The first one leverages classical ray consistency formulations to introduce a generic *Verifier* which can measure consistency between a 3D shape and diverse kinds of observations $O$. The second one demonstrates that it is possible to even further relax the supervision required and presents a technique to learn 3D-from-2D without even requiring the camera viewpoints $C$ for training.

<!--The operationalization of the various multi-view supervised approaches can also be considered to have a common outline. The training set consists of tuples $(I, O, C)$ where $I$ is an input image, $O$ is an observation of the same object/scene as in $I$, but from a different camera viewpoint $C$. The central insight is that the 3D strucutre predicted using $I$, if correct, should be *geometrically consistent* with $O$ different when viewed from perspective $C$.-->


<!--
Let us consider two agents - a *Predictor* $(P)$ and a *Verifier* $(V)$. $P$ sees a single image $I$ of an object/scene and outputs a 3D representation $x$ from it. $V$ then looks at $x$, alongwith some extra training signal $T$, and intructs $P$ on how well it did, and also on how to improve its prediction $x$. The setup described actually corresponds to training the predictor $P$ (usually a Neural Network), by optimizing a loss function (implemented as $V$) using training signal $T$.
-->

<!--
In directly supervised settings, *V* has access to the correct answer $x_{gt}$ that *P* should predict and simply asks it to get the  predicted output $x$ closer to the correct output $x_{gt}$.
-->

<!--

- We know precisely, in the form of concise geometric equations, the relationship between a 3D representation and its 2D projections. This allows us to say something about what the 3D structure should be like given a corresponding 2D view i.e. the 3D structure should be *geometrically consistent* with the 2D views.

- If we have many such 2D views of a single 3D object/scene, the task of inferring the 3D structure can be addressed directly by optimizing for a 3D shape the is maximally consistent with the many available views. Numerous geometry-based techniques have addressed this task of reconstructing a single instance given multiple views, resulting in one of the early success stories in Computer Vision e.g. SfM, MVS, etc.

- The insight in the recent 
-->

## Differentiable Ray Consistency

<!--
<div style="text-align:center"><img src="https://i.imgur.com/4N0WVxE.png" width="450"></div>
-->

<!--Consider the 3D shape and the depth/foreground mask image shown above - how consistent are the two? -->
In our [recent paper](https://arxiv.org/pdf/1704.06254.pdf), we formulate a *Verifier* $V$ to measure the consistency between a 3D shape (represented as a probabilistic occupancy grid) and a 2D observation. Our generic formulation allows learning volumetric 3D prediction by leveraging different types of multi-view observations e.g. foreground masks, depth, color images, semantics etc. as supervision.

An insight which allows defining $V$ is that each pixel in the observation $O$ corresponds to a ray with some associated information. Then, instead of computing the geometric consistency between the observation $O$ and the shape $y$, we can consider, one ray at a time, the consistency between the shape $y$ and a ray $r$.

<div style="text-align:center"><img src="https://i.imgur.com/NQaMdTl.png"></div>

The figure above depicts the various aspects of formulating the ray consistency cost. a) The predicted 3D shape and a sample ray with which we measure consistency. b,c) We trace the ray through the 3D shape and compute *event probabilities* - the probabilities that the ray terminates at various points on its path. d) We can measure how inconsistent each ray termination event is with the information available for that ray. e) By defining the ray consistency cost as the expected event cost, we can compute gradients for how the prediction should be updated to increase the consistency. While in this example we visualize a depth observation $O$, an advantage of our formulation is that it allows incorporating diverse kinds of observations (color images, foreground masks etc.) by simply defining the corresponding event cost function.

 The results of 3D-from-2D prediction learned using our framework in different settings are shown below. Note that all the visualized predictions are obtained from a single RGB image by a *Predictor* $P$ trained *without using 3D supervision*.
<!--![DRC Overview](https://shubhtuls.github.io/drc/resources/images/formulation.png) -->

![](https://shubhtuls.github.io/drc/resources/images/sNetVis.png)  |  ![](https://shubhtuls.github.io/drc/resources/images/pascalVis.png)
:-------------------------:|:-------------------------:
Results on ShapeNet dataset using multiple depth images as supervision for training. a) Input image. b,c) Predicted 3D shape.  | Results on PASCAL VOC dataset using pose and foreground masks as supervision for training. a) Input image. b,c) Predicted 3D shape.

![](https://shubhtuls.github.io/drc/resources/images/csVis.png)  |  ![](https://shubhtuls.github.io/drc/resources/images/sNetColorVis.png)
:-------------------------:|:-------------------------:
Results on Cityscapes dataset using  depth, semantics as supervision. a) Input image. b,c) Predicted 3D shape rendered under simulated forward motion. | Results on ShapeNet dataset using multiple color images as supervision for training shape and per-voxel color prediction. a) Input image. b,c) Predicted 3D shape. 


## Learning Depth and Pose from Unlabeled Videos
Notice that in the above work, the input to the verifier $V$ is an observation with *known* camera viewpoint/pose. This is reasonable from the perspective of an agent with sensorimotor functionality (e.g. human or robots with odometers), but prevents its applications to more unstructured data sources (e.g. videos). In another [recent work](https://arxiv.org/abs/1704.07813), we show that the pose requirement can be relaxed, and in fact jointly learned with the single image 3D predictor $P$.

![Problem setup](https://people.eecs.berkeley.edu/~tinghuiz/bair_blog/teaser_h.jpg)


More specifically, our verifier $V$ in this case is based on a *differentiable depth-based view synthesizer* that outputs a target view of the scene using the predicted depth map and pixels from a source view (i.e. observation) seen under a different camera pose. Here both the depth map and the camera pose are predicted, and the consistency is defined by the pixel reconstruction error between the synthesized and the ground-truth target view. By jointly learning the scene geometry and the camera pose, we are able to train the system on unlabeled video clips without any direct supervision for either depth or pose. 


<!--
adopt the setup of having an agent (e.g. a car mounted with a monocular camera) move around and explore the world while capturing videos. From a large set of such unlabeled video clips, our goal is to jointly learn a single-view CNN predicting per-pixel depth and a multi-view CNN predicting the relative camera pose between input views. 
-->


<!--
Our approach is again based on the principle of learning via geometric consistency, and the verifier is defined using the task of *view synthesis*: given one input view of a scene, synthesize a new image of the scene seen from a different camera pose. It turns out the entire view synthesis pipeline can be formulated in a differentiable manner with depth and pose as the intermediate output predicted by deep networks. As a result, the view reconstruction loss can serve as a measure of the depth and pose prediction quality, and provide supervision signal for both networks directly. The figure below shows an overview of our learning pipeline. -->

![Training pipeline](https://people.eecs.berkeley.edu/~tinghuiz/bair_blog/pipeline.jpg =600x) |
:-------------------------:|:-------------------------:
Formulating the verifier as a depth-based view synthesizer and joint learning of depth and camera pose allows us to train the entire system from unlabeled videos without any direct supervision for either depth or pose. | 


<!--Our approach also bears resemblance to the *direct methods* in structure from motion literature, where both the camera parameters and scene geometry are estimated by minimizing a pixel-based error function. However, in contrast to the classic direct methods, our apporach is learning-based that allows the network to learn an implicit prior from a large corpus of related imagery.-->

We train and evaluate our model on the KITTI and Cityscapes datasets, which consist of videos captured by a car driving in urban cities. The video below shows frame-by-frame (i.e. no temporal smoothness) prediction made by our single-view depth network. 

<p style="text-align:center;">
<iframe  width="560" height="315" src="https://www.youtube.com/embed/UTlpYilJgrk" frameborder="0" allowfullscreen></iframe>
</p>

Surprisingly, despite being trained without any ground-truth labels, our single-view depth model performs on par with some of the supervised baselines, while the pose estimation model is also comparable with well-established SLAM systems (see the [paper](https://arxiv.org/pdf/1704.07813.pdf) for more details).

## Concluding Remarks

Learning single image 3D without 3D supervision is an exciting and thriving topic in computer vision. Using geometry as a bridge between the learning system and the multi-view training data allows us to bypass the tedious and expensive process of acquiring ground-truth 3D labels. More broadly, one could interpret the geometric consistency as a form of *meta supervision* on not *what* the prediction is but *how* it should behave. We believe that similar principles could be applied to other problem domains where obtaining direct labels is difficult or infeasible.

<hr />

This post is based on the following papers:
* Multi-view Supervision for Single-view Reconstruction via Differentiable Ray Consistency. CVPR 2017 ([pdf](https://arxiv.org/pdf/1704.06254.pdf), [code](https://github.com/shubhtuls/drc), [webpage](https://shubhtuls.github.io/drc/))
*Shubham Tulsiani, Tinghui Zhou, Alexei Efros, Jitendra Malik*
* Unsupervised Learning of Depth and Ego-Motion from Video. CVPR 2017 ([pdf](https://arxiv.org/pdf/1704.07813.pdf), [code](https://github.com/tinghuiz/SfMLearner), [webpage](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/))
*Tinghui Zhou, Matthew Brown, Noah Snavely, David Lowe*

<!--Other recent multi-view supervised 3D prediction papers :
-->
### References
[1] Lawrence G. Roberts. *Machine perception of three-dimensional solids.* PhD Dissertation, MIT, 1963.

[2] Harry G. Barrow, Jay M. Tenenbaum. *Interpreting line drawings as three-dimensional surfaces.* Artificial intelligence, 1981.

[3] Derek Hoiem, Alexei A. Efros, Martial Hebert. *Automatic photo pop-up.* In SIGGRAPH, 2005.

[4] Ashutosh Saxena, Sung H. Chung, Andrew Y. Ng. *Learning depth from single monocular images.* In NIPS, 2006.

[5] David Eigen, Christian Puhrsch, Rob Fergus. *Depth map prediction from a single image using a multi-scale deep network.* In NIPS, 2014.

[6] Xiaolong Wang, David Fouhey, Abhinav Gupta. *Designing deep networks for surface normal estimation.* In CVPR, 2015.

[7] Ravi Garg, Vijay Kumar BG, Gustavo Carneiro, Ian Reid. *Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue.* In ECCV, 2016.

[8] Xinchen Yan, Jimei Yang, Ersin Yumer, Yijie Guo, Honglak Lee. *Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction without 3D Supervision.* In NIPS, 2016.

[9] Danilo Jimenez Rezende, S. M. Ali Eslami, Shakir Mohamed, Peter Battaglia, Max Jaderberg, Nicolas Heess. *Unsupervised Learning of 3D Structure from Images.* In NIPS, 2016.

[10] Matheus Gadelha, Subhransu Maji, Rui Wang. *3D Shape Induction from 2D Views of Multiple Objects.* arXiv preprint, 2016.

[11] Clément Godard, Oisin Mac Aodha, Gabriel J. Brostow. *Unsupervised Monocular Depth Estimation with Left-Right Consistency.* In CVPR, 2017.


