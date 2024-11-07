---
layout:             post
title:              "Virtual Personas for Language Models via an Anthology of Backstories"
date:               2024-11-08  09:00:00
author:             <a href="https://scholar.google.com/citations?user=k4TJRc0AAAAJ&hl=en&oi=ao">Suhong Moon</a>, <a href="https://abdulhaim.github.io/">Marwa Abdulhai</a>, <a href="https://joshuaminwookang.github.io/">Minwoo Kang</a>, <a href="https://josephjeesungsuh.github.io/">Joseph Suh</a>, <a href="https://openreview.net/profile?id=~Widyadewi_Soedarmadji1">Widyadewi Soedarmadji</a>, <a href="https://www.linkedin.com/in/erankohenbehar/">Eran Kohen Behar</a>, <a href="https://dchan.cc/">David. M Chan</a>, and <a href="https://people.eecs.berkeley.edu/~jfc/">John Canny</a>
img:                /assets/virtual_personas/image1.png
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
<meta name="twitter:title" content="Virtual Personas for Language Models via an Anthology of Backstories">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/virtual_personas/header.png">

<meta name="keywords" content="large language models, computational social science, virtual personas">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Suhong Moon, Marwa Abdulhai, Minwoo Kang, Joseph Suh, Widyadewi Soedarmadji, Eran Kohen Behar, David. M Chan, John Canny">

<!--
The actual text for the post content appears below.  Text will appear on the
homepage, i.e., https://bair.berkeley.edu/blog/ but we only show part of the
posts on the homepage. The rest is accessed via clicking 'Continue'. This is
enforced with the `more` excerpt separator.
-->

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/virtual_personas/header.png" width="70%">
<br>
<i style="font-size: 0.9em;">We introduce <b>Anthology</b>, a method for conditioning LLMs to representative, consistent, and diverse virtual personas by generating and utilizing naturalistic backstories with rich details of individual values and experience.</i>
</p>
What does it mean for large language models (LLMs) to be trained on massive text corpora, collectively produced by millions and billions of distinctive human authors? 

In [“Language Models as Agent Models”][1], compelling evidence suggests that recent language models could be considered models of *agents*: provided with a textual context, LLMs are capable of generating conditional text that represents the characteristics of an agent likely to have produced that context. This suggests that, with appropriate conditioning, LLMs could be guided to approximate the responses of a particular human voice, rather than the *mixture of voices* that otherwise emerges. If realized, this capability of LLMs would have significant implications for user research and social sciences—conditioned language models as **virtual personas** of human subjects could serve as cost-effective pilot studies and supporting best practices in human studies, e.g. the Belmont principles of justice and beneficence.

In this work, we introduce **Anthology**, an approach for steering LLMs to representative, consistent, and diverse virtual personas by providing richly detailed life narratives of individuals as conditioning context to models.
<!--more-->
In doing so, we also present methods to generate backstories from LLMs themselves as a means to efficiently produce massive sets covering a wide range of human demographics.
By grounding language models in naturalistic backstories, Anthology allows LLMs to simulate individual human samples with increased fidelity, measured in terms of matching the distributions and consistencies of human responses.

## Our Approach: *Anthology*
### Conditioning Language Model Generation with Individual Life Narratives
A significant limitation of earlier methods in steering LLMs to virtual personas has been the inability to reliably approximate **individual** human samples. [Prior][2] [approaches][3] prompt LLMs with broad demographic information, e.g., "I am a 25-year-old from California. My highest level of education is less than high school," which are essentially bodies of text generated from a tuple of demographic variables. 
With these methods, we are only able to approximate human samples at a *population level*, not at the individual level, which results in:
* Responses prone to LLMs defaulting to stereotypical and/or prototypical portrayals, as they are only conditioned on demographic variables (e.g., race and gender)
* Inability to provide important metrics of interest such as covariance and statistical significance, as individual responses are required for such compuatations

Anthology enables the approximation of individual subjects by conditioning with richly detailed backstories. Through these backstories, the model captures implicit and explicit markers of personal identity, including demographic traits and spontaneous references to cultural, socioeconomic backgrounds, and life philosophies. Our approach involves generating a vast set of backstories representing a wide range of demographic attributes via language models queried with unrestricted, open-ended prompts such as, "Tell me about yourself." We then match virtual personas conditioned by each backstory to real-world survey samples.

### Results: Closer Approximation of Public Opinion Polls
For evaluation, we compare the effectiveness of different methods for conditioning virtual personas in the context of approximating three Pew Research Center ATP surveys: Waves 34, 92, and 99.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/virtual_personas/results.png" width="100%">
<br>
<i>Results on approximating human responses for Pew Research Center ATP surveys. Boldface and underlined results indicate values closest and the second closest to those of humans, respectively.</i>
</p>

As measures of success in approximating human samples with virtual personas, we consider the following metrics:
* Average Wasserstein distance (WD) between response distributions as a measure of representativeness
* Frobenius norm (Fro.) between correlation matrices as a measure of consistency
* Cronbach's alpha as an additional measure of internal consistency

Prior to analyzing virtual subjects, we estimate the lower bounds of each evaluation metric by repeatedly dividing the human population into two equal-sized groups at random and calculating these metrics between the subgroups. 
We take averaged values from 100 iterations to represent the lower-bound estimates.

We consistently observe that *Anthology* outperforms other conditioning methods with respect to all metrics, for both the Llama-3-70B and the Mixtral-8x22B. 
When comparing two matching methods, the greedy matching method tends to show better performance on the average Wasserstein distance across all Waves. We attribute differences in matching methods to the one-to-one correspondence condition of maximum weight matching and the limited number of virtual users available. Specifically, the weights assigned to matched virtual subjects in maximum weight matching are inevitably lower than those in greedy matching, as the latter relaxes the constraints on one-to-one correspondence. This discrepancy can result in a lower demographic similarity between matched human and virtual users compared to the counterpart from greedy matching. These results suggest that the richness of the generated backstories in our approach elicits more nuanced responses compared to baselines. 

## Final Thoughts
Anthology marks a promising new direction in conditioning virtual personas in LLMs that could potentially reshape how we conduct user research, public opinion surveys, and other social science applications by offering a scalable, and at times, ethical alternative to traditional human surveys.
However, the use of Anthology, as in any other application of language models in the social sciences, also brings several considerations to the forefront: although the generated backstories help create more representative personas, there remains a risk of perpetuating biases or infringing on privacy, so results should be used and interpreted with caution.

In terms of future steps, we envision our approach benefiting from a more expansive and diverse set of backstories, each representing a consistent life narrative of individuals.
Additionally, a valuable extension of the work would be to consider free-form response generation, enabling more natural and nuanced persona simulations beyond structured survey formats such as multiple-choice. 
Finally, an exciting next dimension in applying LLMs in behavioral studies would involve simulating longer-term effects, allowing virtual personas to model and retrospectively examine changes over time.

All of these directions present multitudes of technical challenges; please let us know if you are interested in collaborating or want to discuss our work further!

**Learn more about our work: [ link to full paper ](https://arxiv.org/abs/2407.06576)**

[1]:https://arxiv.org/abs/2212.01681
[2]:https://arxiv.org/abs/2303.17548
[3]:https://arxiv.org/abs/2209.06899

```
@article{moon2024virtual,
  title={Virtual personas for language models via an anthology of backstories},
  author={Moon, Suhong and Abdulhai, Marwa and Kang, Minwoo and Suh, Joseph and Soedarmadji, Widyadewi and Behar, Eran Kohen and Chan, David M},
  journal={arXiv preprint arXiv:2407.06576},
  year={2024}
}
```

[1]:https://arxiv.org/abs/2212.01681
[2]:https://arxiv.org/abs/2303.17548
[3]:https://arxiv.org/abs/2209.06899