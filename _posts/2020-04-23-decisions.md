---
layout:             post
title:              "Making Decision Trees Accurate Again: Explaining What Explainable AI Did Not"
date:               2020-04-23 9:00:00
author:             <a href="alvinwan.com">Alvin Wan</a>
img:                assets/decisions/decision_Trees.jpeg
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<!--
TODO TODO TODO
Be careful that these three lines are at the top, and that the title and image change for each blog post!
Edit: done
-->
<meta name="twitter:title" content="Making Decision Trees Accurate Again: Explaining what Explainable AI Did Not">
<meta name="twitter:card" content="summary_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/decisions/decision_Trees.jpeg">


![](https://paper-attachments.dropbox.com/s_BA6CF1E0C7002B59FE18648452A87A9807AAD32A0C81C7DB875B31DB25865507_1587079680844_Artboard+23x-100-min.jpg)

The interpretability of neural networks is becoming increasingly necessary, as
deep learning is being adopted in settings where accurate *and* justifiable
predictions are required. These applications range from finance to medical
imaging. However, deep neural networks are notorious for a lack of
justification. Explainable AI (XAI) attempts to bridge this divide between
accuracy and interpretability, but as we explain below, *XAI justifies
decisions without interpreting the model directly*.

# What is “Interpretable”?

Defining explainability or interpretability for computer vision is challenging:
What does it even *mean* to explain a classification for high-dimensional
inputs like images? As we discuss below, two popular definitions involve
*saliency maps* and *decision trees*, but both approaches have their
weaknesses.

# What Explainable AI Doesn’t Explain

## Saliency Maps[^saliency]

Many XAI methods produce saliency maps, but saliency maps focus on the input
and neglect to explain *how* the model makes decisions. For more on saliency
maps, see
[these](https://medium.com/datadriveninvestor/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4)
[saliency](https://medium.com/@thelastalias/saliency-maps-for-deep-learning-part-1-vanilla-gradient-1d0665de3284)
[tutorials](https://towardsdatascience.com/saliency-based-image-segmentation-473b4cb31774)
and [Github](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
[repositories](https://github.com/PAIR-code/saliency).

![Picturing the original image (left), saliency map using a method called Grad-CAM (middle), and another using Guided Backpropagation (right). The picture above is the canonical example for “class-discrimination”. The above saliency maps are taken from https://github.com/kazuto1011/grad-cam-pytorch.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586745832086_cat_dog.png)
![](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586745800570_0-resnet152-gradcam-layer4-boxer.png)
![](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586745848620_0-resnet152-guided-boxer.png)

## What Saliency Maps Fail to Explain

To illustrate why **saliency maps do not fully explain how the model
predicts**, here is an example: Below, the saliency maps are identical, but the
predictions differ. Why? Even though both saliency maps highlight the correct
object, one prediction is incorrect. How? Answering this could help us improve
the model, but as shown below, saliency maps fail to explain the model’s
decision process.

![(Left) The model predicts Eared Grebe. (Right) The model predicts Horned Grebe. These are Grad-CAM results for a ResNet18 model trained on Caltech-UCSD Birds-200–2011, or CUB 2011 for short. Although the saliency maps look extremely similar, the model predictions differ. As a result, saliency maps do not explain how the model reached its final prediction.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586474513136_92505169_559847128069778_9067506750264967168_n.png)
![](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586474509213_92523788_505242710353640_1499929015010459648_n.png)

## Decision Trees

Another approach is to **replace neural networks with interpretable models**.
Before deep learning, decision trees were the gold standard for accuracy and
interpretability. Below, we illustrate the interpretability of decision trees.

![Instead of only predicting “Super Burger” or “Waffle fries”, the above decision tree will output a sequence of decisions that lead up to a final prediction. These intermediate decisions can then be verified or challenged separately.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586766191312_decision_tree_example_blog.jpg)

For accuracy, however, **decision trees lag behind neural networks by up to 40%
accuracy** on image classification datasets[^data].
Neural-network-and-decision-tree hybrids also underperform, failing to match
neural networks on even the dataset CIFAR10, which features tiny 32x32 images
like the one below.

![](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586476074969_dog7.png)

As we show in our [paper](https://arxiv.org/abs/2004.00221) (Sec 5.2), this
accuracy gap damages interpretability: *high-accuracy, interpretable models are
needed to explain high-accuracy neural networks.*

# Enter Neural-Backed Decision Trees

We challenge this false dichotomy by building models that are both
interpretable and accurate. Our key insight is to combine neural networks with
decision trees, preserving high-level interpretability while using neural
networks for low-level decisions, as shown below. We call these models
[**Neural-Backed Decision Trees**](http://nbdt.alvinwan.com) (NBDTs) and show
they can **match neural network accuracy while preserving the interpretability
of a decision tree.**

![In this figure, each node contains a neural network. The figure only highlights one such node and the neural network inside. In a neural-backed decision tree, predictions are made via a decision tree, preserving high-level interpretability. However, each node in decision tree is a neural network making low-level decisions. The “low-level” decision made by the neural network above is “Has sausage” or “no sausage”.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586768596294_blog_nbdt.jpg)

**NBDTs** **are** **as i****nterpretable** **as** **decision trees****.**
Unlike neural networks today, NBDTs can output intermediate decisions for a
prediction. For example, given an image, a neural network may output *Dog*.
However, an NBDT can output both *Dog* and *Animal*, *Chordate*, *Carnivore*
(below).

![Instead of outputting only a prediction “Dog”, Neural-Backed Decision Trees (NBDT) output a series of decisions that lead up to the prediction. Pictured above, the demo NBDT outputs “Animal”, “Chordate”, “Carnivore”, and then “Dog”. The trajectory through the hierarchy is also visualized, to illustrate which possibilities were rejected. The photos above are taken from pexels.com, under the Pexels License.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586382243983_Screen+Shot+2020-04-08+at+2.43.59+PM.png)

**NBDTs achieve neural network accuracy.** Unlike any other decision-tree-based
method, NBDTs match neural network accuracy (< 1% difference) on CIFAR10,
CIFAR100, and TinyImageNet200. NBDTs also achieve accuracy within 2% of neural
networks on ImageNet, setting a new state-of-the-art accuracy for interpretable
models. The NBDT’s ImageNet accuracy of 75.30% outperforms the best competing
decision-tree-based method by a whole ~14%.

# How and what Neural-Backed Decision Trees Explain

## Justifications for Individual Predictions

The most insightful justifications are for objects the model has never seen
before. For example, consider an NBDT (below), and run inference on a *Zebra*.
Although this model has never seen *Zebra*, the intermediate decisions shown
below are correct — *Zebras* are both *Animals* and *Ungulates* (hoofed
animal). The ability to see justification for individual predictions is
quintessential for unseen objects.

![NBDTs make accurate intermediate decisions even for unseen objects. Here, the model was trained on CIFAR10 and has never seen zebras before. Despite that, the NBDT correctly identifies the Zebra as both an Animal and an Ungulate (hoofed animal). The photos above are taken from pexels.com, under the Pexels License.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586429228954_Screen+Shot+2020-04-09+at+3.47.02+AM.png)

## Justifications for Model Behavior

Furthermore, we find that with NBDTs, interpretability improves *with*
accuracy. This is contrary to the dichotomy in the introduction: NBDTs not only
have both accuracy and interpretability; they also make both accuracy and
interpretability the same objective.

![The ResNet10 hierarchy (above) makes less sense than the WideResNet hierarchy (right). In this hierarchy, Cat, Frog, and Airplane are placed under the same subtree.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586431141394_CIFAR10-induced-ResNet10.jpg)
![The WideResNet hierarchy (above) makes more sense than the ResNet10 hierarchy (left). The WideResNet hierarchy cleanly splits Animals and Vehicles, on each side of the hierarchy.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586473205530_CIFAR10-induced-wrn28_10_cifar10+1.jpg)

For example, ResNet10 achieves 4% lower accuracy than WideResNet28x10 on
CIFAR10. Correspondingly, the lower-accuracy ResNet^6 hierarchy (left) makes
less sense, grouping *Frog*, *Cat*, and *Airplane* together. This is “less
sensible,” as it is difficult to find an obvious visual feature shared by all
three classes. By contrast, the higher-accuracy WideResNet hierarchy (right)
makes more sense, cleanly separating *Animal* from *Vehicle*—thus, the higher
accuracy, the more interpretable the NBDT.

## Understanding Decision Rules

With low-dimensional tabular data, decision rules in a decision tree are simple
to interpret e.g., if the dish contains a bun, then pick the right child, as
shown below. However, decision rules are not as straightforward for inputs like
high-dimensional images.  As we *qualitatively* find in the
[paper](https://arxiv.org/abs/2004.00221) (Sec 5.3), the model’s decision rules
are based not only on object type but also on context, shape, and color.

![This example demonstrates how decision rules are easy to interpret with low-dimensional, tabular data. To the right is example tabular data for several items. To the left is a decision tree we trained on this data. In this case, the decision rule (blue) is “Has bun or not?” All items with a bun (orange) are sent to the top child, and all items without a bun (green) are sent to the bottom child.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1587072198036_blog_tabular+1.jpg)

To interpret decision rules *quantitatively*, we leverage an existing hierarchy
of nouns called WordNet[^wordnet]; with this hierarchy, we can find the most
specific shared meaning between classes. For example, given the classes *Cat*
and *Dog*, WordNet would provide *Mammal*. In our
[paper](https://arxiv.org/pdf/2004.00221.pdf) (Sec 5.2) and pictured below, we
quantitatively verify these WordNet hypotheses.

![The WordNet hypothesis for the left subtree (red arrow) is Vehicle. The WordNet hypothesis for the right (blue arrow) is Animal. To validate these meanings qualitatively, we tested the NBDT against unseen classes of objects: 1. Find images that were not seen during training. 2. Given the hypothesis, determine which child each image belongs to. For example, we know that Elephant is an Animal so is *supposed to go the right subtree. 3. We can now evaluate the hypothesis, by checking how many images are passed to the correct child. For example, check how many Elephant images are sent to the Animal subtree. These accuracies per-class are shown to the right, with unseen Animals (blue) and unseen Vehicles (red) both showing high accuracies.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1587067002530_CIFAR10_WRN_Tree-5.jpg)
![](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1587067006203_ood_both.jpg)

Note that in small datasets with 10 classes i.e., CIFAR10, we can find WordNet
hypotheses for all nodes. However, in large datasets with 1000 classes i.e.,
ImageNet, we can only find WordNet hypotheses for a subset of nodes.

# How it Works

The training and inference process for a Neural-Backed Decision Tree can be
broken down into four steps.

![Training an NBDT occurs in 2 phases: First, construct the hierarchy for the decision tree. Second, train the neural network with a special loss term. To run inference, pass the sample through the neural network backbone. Finally, run the final fully-connected layer as a sequence of decision rules.](https://paper-attachments.dropbox.com/s_BFFE4A09F6DC2B1D4283B10174F1A42E9AD08F8A70233F89DED078AA81384775_1586430067027_pipeline.jpg)

1. Construct a hierarchy for the decision tree, called the **Induced
Hierarchy**.
2. This hierarchy yields a particular loss function, which we call the **Tree
Supervision Loss**.
3. Start inference by passing the sample through the neural network backbone.
The backbone is all neural network layers before the final fully-connected
layer.
4. Finish inference by running the final fully-connected layer as a sequence of
decision rules, which we call **Embedded Decision Rules**. These decisions
culminate in the final prediction.

## Running Embedded Decision Rules

We first discuss inference. As explained above, our NBDT approach featurizes
each sample using the neural network backbone. To understand what happens next,
we will first construct a degenerate decision tree that is equivalent to a
fully-connected layer.

**Fully-Connected Layer:** Running inference with a featurized sample is a
matrix-vector product, as shown below.

![](https://paper-attachments.dropbox.com/s_BA6CF1E0C7002B59FE18648452A87A9807AAD32A0C81C7DB875B31DB25865507_1586509266298_Screen+Shot+2020-04-10+at+2.01.02+AM.png)

This yields a matrix-vector product yields a vector of inner products, which we
denote with $\hat{y}$. The index of the largest inner product is our class
prediction.

![](https://paper-attachments.dropbox.com/s_BA6CF1E0C7002B59FE18648452A87A9807AAD32A0C81C7DB875B31DB25865507_1586509813684_inference_modes.jpg)

**Naive Decision Tree**: We construct a basic decision tree with one root node
and a leaf for each class. This is pictured by “B - Naive” in the figure above.
Each leaf is directly connected to the root and has a representative vector,
namely a row vector from $W$ (Eqn. 1 above).

Also pictured above, running inference with a featurized sample $x$ means
taking inner products between $x$ and each child node’s representative vector.
Like the fully-connected layer, the index of the largest inner product is our
class prediction.

The direct equivalence between a fully-connected layer and a naive decision
tree motivates our particular inference method, using an inner-product decision
tree. In our work, we then extend this naive tree to deeper trees. However,
that discussion is beyond the scope of this article.  Our
[paper](https://arxiv.org/abs/2004.00221) (Sec. 3.1) discusses how this works,
in detail.

## Building Induced Hierarchies

This hierarchy determines which sets of classes the NBDT must decide between.
We refer to this hierarchy as an **Induced Hierarchy** because we build the
hierarchy using a pretrained neural network’s weights.

![](https://paper-attachments.dropbox.com/s_BA6CF1E0C7002B59FE18648452A87A9807AAD32A0C81C7DB875B31DB25865507_1586510110433_76388304-0e6aaa80-6326-11ea-8c9b-6d08cb89fafe.jpg)

In particular, we view each row vector in the fully-connected layer’s weight
matrix W as a point in d-dimensional space. This is illustrated by “Step B -
Set Leaf Vectors“. We then perform hierarchical agglomerative clustering on
these points. The successive clustering then determines the hierarchy, as
illustrated above. Our [paper](https://arxiv.org/abs/2004.00221) (Sec. 3.2)
discusses this in more detail.

## Training with Tree Supervision Loss

![](https://paper-attachments.dropbox.com/s_BA6CF1E0C7002B59FE18648452A87A9807AAD32A0C81C7DB875B31DB25865507_1586511030363_77226784-3208ce80-6b38-11ea-84bb-5128e3836665.jpg)

Consider “A - Hard” in the figure above. Say the green node corresponds to the
*Horse* class. This is just one class. However, it is also an *Animal*
(orange). As a result, we know that a sample arriving at the root node (blue)
*should* go to the right, to *Animal*. The sample arriving at the node *Animal*
also *should* go to the right again, towards *Horse*. We train each node to
predict the correct child node. We call the loss that enforces this the **Tree
Supervision Loss**, which is effectively a cross entropy loss for each node.

Our [paper](https://arxiv.org/abs/2004.00221) (Sec. 3.3) discusses this in more detail and further explains “B - Soft”.

# Trying NBDTs in under a minute

Interested in trying out an NBDT, *now*? Without installing anything, you can
[view more example outputs online](http://nbdt.alvinwan.com) and even [try out
our web demo](http://nbdt.alvinwan.com/demo/). Alternatively, use our
command-line utility to run inference (Install with `pip install nbdt`). Below,
we run inference on a [picture of a
cat](https://images.pexels.com/photos/126407/pexels-photo-126407.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=200).

```
nbdt https://images.pexels.com/photos/126407/pexels-photo-126407.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=32  # this can also be a path to local image
```

This outputs both the class prediction and all the intermediate decisions.

```
Prediction: cat // Decisions: animal (99.47%), chordate (99.20%), carnivore (99.42%), cat (99.86%)
```

You can load a pretrained NBDT in just a few lines of Python as well. Use the
following to get started. We support several WideResNet28x10, ResNet18 for
CIFAR100, CIFAR100, and TinyImageNet200.

```python
from nbdt.model import HardNBDT
from nbdt.models import wrn28_10_cifar10

model = wrn28_10_cifar10()
model = HardNBDT(
  pretrained=True,
  dataset='CIFAR10',
  arch='wrn28_10_cifar10',
  model=model)
```

For reference, see the [script for the command-line
tool](https://github.com/alvinwan/neural-backed-decision-trees/blob/master/nbdt/bin/nbdt)
we ran above; only ~20 lines are directly involved in transforming the input
and running inference. For more instructions on getting started and examples,
see our [Github
repository](https://github.com/alvinwan/neural-backed-decision-trees).

# Conclusion

Explainable AI does not *fully* explain how the neural network reaches a
prediction: Existing methods explain the image’s impact on model predictions
but do not explain the decision process. Decision trees address this, but
unfortunately, images[^images] are kryptonite for decision tree accuracy.

We thus combine neural networks and decision trees. Unlike predecessors that
arrived at the same hybrid design, our neural-backed decision trees (NBDTs)
simultaneously address the failures (1) of neural networks to provide
justification and (2) of decision trees to attain high accuracy. This primes a
new category of accurate, interpretable NBDTs for applications like medicine
and finance. To get started, see the [project page](http://nbdt.alvinwan.com).

By [Alvin Wan](http://alvinwan.com/), \*[Lisa
Dunlap](https://github.com/lisadunlap), \*[Daniel
Ho](https://github.com/daniel-ho), [Jihan
Yin](https://www.linkedin.com/in/jihanyin/), [Scott
Lee](https://www.linkedin.com/in/scottjlee98/), [Henry
Jin](https://www.linkedin.com/in/henryjin99/), [Suzanne
Petryk](https://spetryk.github.io/), [Sarah Adel
Bargal](https://cs-people.bu.edu/sbargal/), [Joseph E.
Gonzalez](https://people.eecs.berkeley.edu/~jegonzal/) where \* denotes
equal contribution.


<hr>

[^saliency]: There are two types of saliency maps: one is white-box, where the
    method has access to the model and its parameters. One popular white-box
    method is Grad-CAM, which uses both gradients and class activation maps to
    visualize attention. You can learn more from the paper, “[Grad-CAM: Visual
    Explanations from Deep Networks via Gradient-based
    Localization](http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)”.
    The other type of saliency map is black-box, where the model does not have
    access to the model parameters. RISE is one such saliency method. RISE
    masks random portions of the input image and passes this image through the
    model — the mask that damages accuracy the most is the most “important”
    portion. You can learn more from the paper “[RISE: Randomized Input
    Sampling for Explanation of Black-box
    Models](http://bmvc2018.org/contents/papers/1064.pdf)”.

[^data]: This 40% gap between decision tree and neural network accuracy shows
    up on TinyImageNet200.

[3] The three datasets in particular are CIFAR10, CIFAR100, TinyImageNet200.

[4] This ImageNet accuracy gain is significant: for *non-interpretable* neural networks, a similar 14% gain on ImageNet [took 3 years](https://paperswithcode.com/sota/image-classification-on-imagenet) of research. To make this comparison, we examine a similar accuracy gain which took 3 years, from AlexNet in 2013 (63.3%) to Inception V3 (78.8%). The NBDT improves on previously state-of-the-art results by ~14% at around the same range, from NofE (61.29%) to our NBDTs (75.30%). There are other factors at play, however: One obvious one is that compute and deep learning libraries were not as readily available in 2013. A fairer comparison may to be use the latest the latest 14%-gain on ImageNet. The latest 14% gain took 5 years, starting from VGG-19 in 2015 (74.5%) and leading up to FixEfficientNet-L2 in 2020 (88.5%). However, this technically isn’t comparable either since large gains are harder at higher accuracies. Despite this lack of perfectly comparable benchmark progress, we just took the minimum of the two ranges in time, to try and illustrate how large of a gap 14% is.

[^wordnet]: WordNet is a lexical hierarchy of various words. A large majority
    of words are nouns, but other parts of speech are included as well. For
    more information, see the [official
    website](https://wordnet.princeton.edu/).

[7] To understand the basic idea for a Tree Supervision Loss: *Horse* is just one class. However, it is also an *Ungulate* and an *Animal*. (See the figure in “Justifications for Individual Predictions”.) At the root node, the *Horse* sample thus needs to be passed to the child node *Animal*. Furthermore, the node *Animal* needs to pass the sample to *Ungulate*. Finally, the node *Ungulate* must pass the sample to *Horse*. Train each node to predict the correct child node. We call the loss that enforces this the Tree Supervision Loss.

[^images]: In general, decision trees perform best with low-dimensional data.
    Images are the antithesis of this best-case scenario, being extremely
    high-dimensional.
