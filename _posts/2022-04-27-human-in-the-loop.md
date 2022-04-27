---
layout:             post
title:              "Rethinking Human-in-the-Loop for Artificial Augmented Intelligence"
date:               2022-04-27  13:55:00
author:             <a href="https://github.com/zhmiao">Zhongqi Miao</a> and <a href="https://liuziwei7.github.io/">Ziwei Liu</a>
img:                /assets/human-in-the-loop/image3.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
---

<!-- twitter -->
<meta name="twitter:title" content="Rethinking Human-in-the-Loop for Artificial Augmented Intelligence">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/human-in-the-loop/image3.png">

<meta name="keywords" content="Human-in-the-loop, Artificial Augmented Intelligence, Real-world applications">
<meta name="description" content="It is time to rethink human-in-the-loop.">
<meta name="author" content="Zhongqi Miao, Ziwei Liu">

<!-- body -->

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/human-in-the-loop/image3.png"
    width="90%">
    <br>
<i>
Figure 1: In real-world applications, we think there exist a human-machine loop where humans and machines are mutually augmenting each other. We call it Artificial Augmented Intelligence.
</i>
</p>


How do we build and evaluate an AI system for real-world applications? In most AI research, the evaluation of AI methods involves a training-validation-testing process. The experiments usually stop when the models have good testing performance on the reported datasets because real-world data distribution is assumed to be modeled by the validation and testing data. However, real-world applications are usually more complicated than a single training-validation-testing process. The biggest difference is the ever-changing data. For example, wildlife datasets change in class composition all the time because of animal invasion, re-introduction, re-colonization, and seasonal animal movements. A model trained, validated, and tested on existing datasets can easily be broken when newly collected data contain novel species. Fortunately, we have out-of-distribution detection methods that can help us detect samples of novel species. However, when we want to expand the recognition capacity (i.e., being able to recognize novel species in the future), the best we can do is fine-tuning the models with new ground-truthed annotations. In other words, we need to incorporate human effort/annotations regardless of how the models perform on previous testing sets.

<!--more-->

# Inevitable human-in-the-loop

When human annotations are inevitable, real-world recognition systems become a never-ending loop of **data collection &rarr; annotation &rarr; model fine-tuning** (Figure 2). As a result, the performance of one single step of model evaluation does not represent the actual generalization of the whole recognition system because the model will be updated with new data annotations, and a new round of evaluation will be conducted. With this loop in mind, we think that instead of building a model with ***better testing performance***, focusing on ***how much human effort can be saved*** is a more generalized and practical goal in real-world applications.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/human-in-the-loop/image1.png"
    height="">
    <br>
<i>
Figure 2: In the loop of data collection, annotation, and model update, the goal of optimization becomes minimizing the requirement of human annotation rather than single-step recognition performance.
</i>
</p>


# A case study on wildlife recognition

In the paper we published last year in Nature-Machine Intelligence [1], we discussed the incorporation of human-in-the-loop into wildlife recognition and proposed to examine human effort efficiency in model updates instead of simple testing performance. For demonstration, we designed a recognition framework that was a combination of active learning, semi-supervised learning, and human-in-the-loop (Figure 3). We also incorporated a time component into this framework to indicate that the recognition models did not stop at any single time step. Generally speaking, in the framework, at each time step, when new data are collected, a recognition model actively selects which data should be annotated based on a prediction confidence metric. Low-confidence predictions are sent for human annotation, and high-confidence predictions are trusted for downstream tasks or pseudo-labels for model updates.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/human-in-the-loop/image2.png"
    height="">
    <br>
<i>
Figure 3: Here, we present an iterative recognition framework that can both maximize the utility of modern image recognition methods and minimize the dependence on manual annotations for model updating.  
</i>
</p>

In terms of human annotation efficiency for model updates, we split the evaluation into 1) the percentage of high-confidence predictions on validation (i.e., saved human effort for annotation); 2) the accuracy of high-confidence predictions (i.e., reliability); and 3) the percentage of novel categories that are detected as low-confidence predictions (i.e., sensitivity to novelty). With these three metrics, the optimization of the framework becomes minimizing human efforts (i.e., to maximize high-confidence percentage) and maximizing model update performance and high-confidence accuracy.

We reported a two-step experiment on a large-scale wildlife camera trap dataset collected from Mozambique National Park for demonstration purposes. The first step was an initialization step to initialize a model with only part of the dataset. In the second step, a new set of data with known and novel classes was applied to the initialized model. Following the framework, the model made predictions on the new dataset with confidence, where high-confidence predictions were trusted as pseudo-labels, and low-confidence predictions were provided with human annotations. Then, the model was updated with both pseudo-labels and annotations and ready for the future time steps. As a result, the percentage of high-confidence predictions on second step validation was 72.2%, the accuracy of high-confidence predictions was 90.2%, and the percentage of novel classes detected as low-confidence was 82.6%. In other words, our framework saved 72% of human effort on annotating all the second step data. As long as the model was confident, 90% of the predictions were correct. In addition, 82% of novel samples were successfully detected. Details of the framework and experiments can be found in the original paper.


# Artificial Augmented Intelligence (A<sup>2</sup>I)

By taking a closer look at Figure 3, besides the **data collection - human annotation - model update** loop, there is another **human-machine** loop hidden in the framework (Figure 1). This is a loop where both humans and machines are constantly improving each other through model updates and human intervention. For example, when AI models cannot recognize novel classes, human intervention can provide information to expand the model’s recognition capacity. On the other hand, when AI models get more and more generalized, the requirement for human effort gets less. In other words, the use of human effort gets more efficient.

In addition, the confidence-based human-in-the-loop framework we proposed is not limited to novel class detection but can also help with issues like long-tailed distribution and multi-domain discrepancies. As long as AI models feel less confident, human intervention comes in to help improve the model. Similarly, human effort is saved as long as AI models feel confident, and sometimes human errors can even be corrected (Figure 4). In this case, the relationship between humans and machines becomes synergistic. Thus, the goal of AI development changes from replacing human intelligence to mutually augmenting both human and machine intelligence. We call this type of AI: **Artificial Augmented Intelligence (A<sup>2</sup>I)**.


Ever since we started working on artificial intelligence, we have been asking ourselves, what do we create AI for? At first, we believed that, ideally, AI should fully replace human effort in simple and tedious tasks such as large-scale image recognition and car driving. Thus, we have been pushing our models to an idea called “human-level performance” for a long time. However, this goal of replacing human effort is intrinsically building up opposition or a mutually exclusive relationship between humans and machines. In real-world applications, the performance of AI methods is just limited by so many affecting factors like long-tailed distribution, multi-domain discrepancies, label noise, weak supervision, out-of-distribution detection, etc. Most of these problems can be somehow relieved with proper human intervention. The framework we proposed is just one example of how these separate problems can be summarized into high- versus low-confidence prediction problems and how human effort can be introduced into the whole AI system. We think it is not cheating or surrendering to hard problems. It is a more human-centric way of AI development, where the focus is on how much human effort is saved rather than how many testing images a model can recognize. Before the realization of Artificial General Intelligence (AGI), we think it is worthwhile to further explore the direction of machine-human interactions and A<sup>2</sup>I such that AI can start making more impacts in various practical fields.

<p style="text-align:center;">
    <img src="http://bair.berkeley.edu/static/blog/human-in-the-loop/image4.png"
    height="">
    <br>
<i>
Figure 4: Examples of high-confidence predictions that did not match the original annotations. Many high-confidence predictions that were flagged as incorrect based on validation labels (provided by students and citizen scientists) were in fact correct upon closer inspection by wildlife experts.  
</i>
</p>

*Acknowledgements: We thank all co-authors of the paper “Iterative Human and Automated Identification of Wildlife Images” for their contributions and discussions in preparing this blog. The views and opinions expressed in this blog are solely of the authors of this paper.*

This blog post is based on the following paper which is published at Nature - Machine Intelligence:\\
[1] Miao, Zhongqi, Ziwei Liu, Kaitlyn M. Gaynor, Meredith S. Palmer, Stella X. Yu, and Wayne M. Getz. "Iterative human and automated identification of wildlife images." Nature Machine Intelligence 3, no. 10 (2021): 885-895.(Link to <a href="https://arxiv.org/pdf/2105.02320.pdf">Pre-print</a>)
