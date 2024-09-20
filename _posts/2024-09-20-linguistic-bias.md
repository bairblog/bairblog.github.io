---
layout:             post
title:              "Linguistic Bias in ChatGPT: Language Models Reinforce Dialect Discrimination"
date:               2024-09-20  09:00:00
author:             <a href="https://www.efleisig.com/">Eve Fleisig</a>, <a href="https://haas.berkeley.edu/faculty/genevieve-smith/">Genevieve Smith</a>, <a href="https://sites.google.com/view/madeline-bossi">Madeline Bossi</a>, <a href="https://www.linkedin.com/in/ishitar/">Ishita Rustagi</a>, <a href="https://scholar.google.com/citations?user=WPCAOQQAAAAJ&hl=en">Xavier Yin</a>, and <a href="https://people.eecs.berkeley.edu/~klein/">Dan Klein</a>
img:                /assets/linguistic-bias/image1.png
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
<meta name="twitter:title" content="Linguistic Bias in ChatGPT: Language Models Reinforce Dialect
Discrimination">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/linguistic-bias/image1.png">

<meta name="keywords" content="language models, AI bias, ChatGPT">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="Eve Fleisig, Genevieve Smith, Madeline Bossi, Ishita Rustagi, Xavier Yin, Dan Klein">

<!--
The actual text for the post content appears below.  Text will appear on the
homepage, i.e., https://bair.berkeley.edu/blog/ but we only show part of the
posts on the homepage. The rest is accessed via clicking 'Continue'. This is
enforced with the `more` excerpt separator.
-->

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/linguistic-bias/image1.png" width="70%">
<br>
<i style="font-size: 0.9em;">Sample language model responses to different varieties of English and native speaker reactions.</i>
</p>

ChatGPT does amazingly well at communicating with people in English. But whose English?

[Only 15%][1] of ChatGPT users are from the US, where Standard American English is the default. But the model is also commonly used in countries and communities where people speak other varieties of English. Over 1 billion people around the world speak varieties such as Indian English, Nigerian English, Irish English, and African-American English.

Speakers of these non-"standard" varieties often face discrimination in the real world. They’ve been told that the way they speak is [unprofessional][2] or [incorrect][3], [discredited as witnesses][4], and [denied housing][5]–despite [extensive][6] [research][7] indicating that all language varieties are equally complex and legitimate. Discriminating against the way someone speaks is often a proxy for discriminating against their race, ethnicity, or nationality. What if ChatGPT exacerbates this discrimination?

To answer this question, [our recent paper][8] examines how ChatGPT's behavior changes in response to text in different varieties of English. We found that ChatGPT responses exhibit consistent and pervasive biases against non-“standard” varieties, including increased stereotyping and demeaning content, poorer comprehension, and condescending responses.

<!--more-->

## Our Study

We prompted both GPT-3.5 Turbo and GPT-4 with text in ten varieties of English: two "standard" varieties, Standard American English (SAE) and Standard British English (SBE); and eight non-"standard" varieties, African-American, Indian, Irish, Jamaican, Kenyan, Nigerian, Scottish, and Singaporean English. Then, we compared the language model responses to the "standard" varieties and the non-"standard" varieties.

First, we wanted to know whether linguistic features of a variety that are present in the prompt would be retained in GPT-3.5 Turbo responses to that prompt. We annotated the prompts and model responses for linguistic features of each variety and whether they used American or British spelling (e.g., "colour" or "practise"). This helps us understand when ChatGPT imitates or doesn’t imitate a variety, and what factors might influence the degree of imitation.

Then, we had native speakers of each of the varieties rate model responses for different qualities, both positive (like warmth, comprehension, and naturalness) and negative (like stereotyping, demeaning content, or condescension). Here, we included the original GPT-3.5 responses, plus responses from GPT-3.5 and GPT-4 where the models were told to imitate the style of the input.

## Results

We expected ChatGPT to produce Standard American English by default: the model was developed in the US, and Standard American English is likely the best-represented variety in its training data. We indeed found that model responses retain features of SAE far more than any non-"standard" dialect (by a margin of over 60%). But surprisingly, the model *does* imitate other varieties of English, though not consistently. In fact, it imitates varieties with more speakers (such as Nigerian and Indian English) more often than varieties with fewer speakers (such as Jamaican English). That suggests that the training data composition influences responses to non-"standard" dialects.

ChatGPT also defaults to American conventions in ways that could frustrate non-American users. For example, model responses to inputs with British spelling (the default in most non-US countries) almost universally revert to American spelling. That’s a substantial fraction of ChatGPT’s userbase likely hindered by ChatGPT’s refusal to accommodate local writing conventions.

**Model responses are consistently biased against non-"standard" varieties.** Default GPT-3.5 responses to non-"standard" varieties consistently exhibit a range of issues: stereotyping (19% worse than for "standard" varieties), demeaning content (25% worse), lack of comprehension (9% worse), and condescending responses (15% worse).

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/linguistic-bias/image2.png" width="90%">
<br>
<i>Native speaker ratings of model responses. Responses to non-”standard” varieties (blue) were rated as worse than responses to “standard” varieties (orange) in terms of stereotyping (19% worse), demeaning content (25% worse), comprehension (9% worse), naturalness (8% worse), and condescension (15% worse).</i>
</p>

When GPT-3.5 is prompted to imitate the input dialect, the responses exacerbate stereotyping content (9% worse) and lack of comprehension (6% worse). GPT-4 is a newer, more powerful model than GPT-3.5, so we’d hope that it would improve over GPT-3.5. But although GPT-4 responses imitating the input improve on GPT-3.5 in terms of warmth, comprehension, and friendliness, they exacerbate stereotyping (14% worse than GPT-3.5 for minoritized varieties). That suggests that larger, newer models don’t automatically solve dialect discrimination: in fact, they might make it worse.

## Implications

ChatGPT can perpetuate linguistic discrimination toward speakers of non-“standard” varieties. If these users have trouble getting ChatGPT to understand them, it’s harder for them to use these tools. That can reinforce barriers against speakers of non-“standard” varieties as AI models become increasingly used in daily life.

Moreover, stereotyping and demeaning responses perpetuate ideas that speakers of non-“standard” varieties speak less correctly and are less deserving of respect. As language model usage increases globally, these tools risk reinforcing power dynamics and amplifying inequalities that harm minoritized language communities.

**Learn more here: [[ paper ]][8]**
<hr>

[1]:https://www.similarweb.com/website/chat.openai.com/#geography
[2]:https://doi.org/10.2307/3587696
[3]:https://doi.org/10.4324/9781410616180
[4]:https://muse.jhu.edu/article/641206/summary
[5]:https://www.taylorfrancis.com/chapters/edit/10.4324/9780203986615-17/linguistic-profiling-john-baugh
[6]:https://www.routledge.com/Language-Society-and-Power-An-Introduction/Mooney-Evans/p/book/9780367638443
[7]:https://books.google.com/books?id=QRFIsGWZ5O4C
[8]:https://arxiv.org/pdf/2406.08818
