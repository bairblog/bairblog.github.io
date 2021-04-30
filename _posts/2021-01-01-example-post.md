---
layout:             post
title:              "Example Post Title"
date:               2021-01-01  9:00:00
author:             <a href="">John Doe</a> and <a href="">Jane Doe</a>
img:                assets/example_post/image1.png
excerpt_separator:  <!--more-->
visible:            False
show_comments:      False
---

<!-- twitter -->
<meta name="twitter:title" content="Example Post Title">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/example_post/image1.png">

<meta name="keywords" content="keyword1, keyword2">
<meta name="description" content="The BAIR Blog">
<meta name="author" content="John Doe, Jane Doe">

<!--
These are comments in HTML The above text is useful for getting tweets to work.
The actual text for the blog post appears below.
-->

This is a template for [BAIR blog][1] posts. Here is an example image.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/example_post/image1.png" width="50%">
<br>
<i><b>Figure title.</b> Figure caption. This image is centered and set to 50%
page width.</i>
</p>

<!--more-->

The content here after the excerpt separator will not appear on the front page
of the BAIR blog but will show in the post.

# Text formatting

Markdown provides text formatting such as **bold** and *italic*.

LaTeX is also supported, such as $y = \beta x + \alpha$ inline, or as a separate
line

$$y = \beta x + \alpha.$$

URLs can be inserted through square brackets, such as [this][1].

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/example_post/image2.png" width="30%">
<br>
<i><b>Figure title.</b> Figure caption. This image is centered and set to 30%
page width.</i>
</p>


[1]:https://bair.berkeley.edu/blog/
