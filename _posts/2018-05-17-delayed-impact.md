---
layout:             post
title:              "Delayed Impact of Fair Machine Learning"
date:               2018-05-17 9:00:00
author:             <a href="http://lydiatliu.github.io/">Lydia T. Liu</a>, 
                    <a href="http://people.eecs.berkeley.edu/~sarahdean/">Sarah Dean</a>, 
                    <a href="https://people.eecs.berkeley.edu/~erolf/">Esther Rolf</a>, 
                    <a href="http://people.eecs.berkeley.edu/~msimchow/index.html">Max Simchowitz</a>, 
                    <a href="http://mrtz.org/">Moritz Hardt</a>
img:                /assets/delayed_impact/outcome_curve_blog.png
excerpt_separator:  <!--more-->
visible:            True
show_comments:      False
jsarr:
                   - fairness_visualization/visualization.js
                   - fairness_visualization/d3.min.js
---
<style type="text/css">
  td {
    font-size: 10pt;
    font-family: 'Roboto', sans-serif;
    border: none !important;
    padding: 0 !important;
  }
  ul {
    line-height: 180%;
    font-family: 'Roboto', sans-serif;
  }
  .thin {
    width: 170px;
  }
  .annotation {
    color: #a00;
    font-size: 10pt;
    visibility: hidden;
    stroke: #d00;
    stroke-width: 5;
    fill:none;
    font-family: 'Roboto', sans-serif;
  }
  .demo {
    font: 10pt;
    color: #fff;
    padding: 6px;
    border: 0;
    border-radius: 4px;
    box-shadow: none;
    margin-bottom: 6px;
    width: 100%;
    background: #555;
    opacity: .5;
    font-family: 'Roboto', sans-serif;
  }
  .broken {
    color: #f00;
  }
  .readout {
    font-weight: 700;
  }
  .title {
    font-weight: 700;
    font-family: 'Roboto', sans-serif;
  }
  .big-label {
    font-size: 16pt;
    font-family: 'Roboto', sans-serif;
  }
  .medium-label {
    font-size: 12pt;
    font-family: 'Roboto', sans-serif;
  }
  .figure-title {
    font-size: 24px;
    font-weight: 400;
    font-family: 'Roboto', sans-serif;
  }
  .figure-caption {
    font-weight:100;
    font-size: 12pt;
    margin-bottom: 10px;
    font-family: 'Roboto', sans-serif;
  }
  .histogram-axis text {
    font: 9pt 'Roboto', sans-serif;
    font-weight: 100;
    color: #000;
  }
  .histogram-legend {
    margin-top: 16px;
    font-family: 'Roboto', sans-serif;
  }
  .instructions {
    font-weight: 700;
    font-family: 'Roboto', sans-serif;
  }
  .correctness-label {
    font-size: 9pt;
    font-weight: 700;
    color: #000;
    font-family: 'Roboto', sans-serif;
  }
  .explanation {
    font-size: 9pt;
    font-weight: 100;
    color: #ccc;
    font-family: 'Roboto', sans-serif;
  }
  .pie-label {
    font-size: 9pt;
    font-weight: 700;
    color: #000;
    font-family: 'Roboto', sans-serif;
  }
  .pie-label1 {
    font-size: 12pt;
    font-weight: 700;
    color: #000;
    font-family: 'Roboto', sans-serif;
  }
  .pie-number {
    font-size: 9pt;
    font-weight: 300;
    color: #000;
    font-family: 'Roboto', sans-serif;
  }
  .line {
    fill: none;
    stroke: darkgrey;
    stroke-width: 2px;
  }
  .line_maxprof {
    fill: none;
    stroke: orange;
    stroke-width: 2px;
  }
  .line_dempar {
    fill: none;
    stroke: teal;
    stroke-width: 2px;
  }
  .line_eqop {
    fill: none;
    stroke: magenta;
    stroke-width: 2px;
  }
  .tick line{
    stroke: lightgrey;
    stroke-opacity: 0.7;
    shape-rendering: crispEdges;
  }
  .legend-label {
    font-size: 8pt;
    font-weight: 300;
    color: #666;
    font-family: 'Roboto', sans-serif;
  }
  .bold-label {
    font-size: 10pt;
    font-weight: 700;
    font-family: 'Roboto', sans-serif;
  }
  .margin-text {
    font-size: 9pt;
    font-weight: 300;
    color: #666;
    font-family: 'Roboto', sans-serif;
  }
  .margin-bold {
    font-size: 9pt;
    font-weight: 700;
    font-family: 'Roboto', sans-serif;
  }
  .domain {
    display: none;
  }
  .profit-readout {
    margin-left: 10px;
    font-family: 'Roboto', sans-serif;
  }
  #profit-title {
    font-size: 18pt;
    font-family: 'Roboto', sans-serif;
  }
  #total-profit {
    font-size: 18pt;
    font-weight: 700;
    font-family: 'Roboto', sans-serif;
  }
  #top-sidebar {
    font-size: 10pt;
    color: #555;
    font-family: 'Roboto', sans-serif;
  }
  #single-histogram-table {
    font-family: 'Roboto', sans-serif;
  }
</style>


Machine learning systems trained to minimize prediction error may often exhibit
discriminatory behavior based on sensitive characteristics such as race and
gender. One reason could be due to historical bias in the data. In various
application domains including lending, hiring, criminal justice, and
advertising, machine learning has been criticized for its potential to *harm*
historically underrepresented or disadvantaged groups.

In this post, we talk about our recent work on aligning decisions made by
machine learning with long term social welfare goals. Commonly, machine learning
models produce a **score** that summarizes information about an individual in
order to make decisions about them. For example, a *credit score* summarizes an
individual’s credit history and financial activities in a way that informs the
bank about their creditworthiness. Let us continue to use the lending setting as
a running example.

<!--more-->

Any group of individuals has a particular distribution of credit scores, as
visualized below.


<div id="single-histogram-table">
</div>


Scores can be turned into decisions by defining a threshold. For example,
individuals above the threshold score are granted loans, and individuals below
the score threshold are denied loans. Such a decision rule is called a
*threshold policy*.

Scores can be interpreted as encoding estimated probabilities of defaulting on a
loan. For example, 90% of people with a credit score of 650 may be expected to
repay a loan granted to them. This allows the bank to forecast the profit they
would expect to make by providing identical loans to individuals with credit
score 650. In the same fashion, the bank can predict the profit they would
expect by loaning to all individuals with a credit above 650, or indeed above
any given threshold.


<div id="single-histogram-interactive-table">
</div>


Without other considerations, a bank will attempt to maximize its total profit.
The profit depends on the ratio of the amount the bank gains from a repaid loan
to the amount the bank loses from a defaulted loan. In the above interactive
figure, this ratio of gain to loss is 1 to -4. As losses become more costly
relative to gains, the bank will issue loans more conservatively, and raise its
loaning threshold. We call the fraction of the population above this threshold
the *selection rate*.


## The outcome curve

Lending decisions not only affect the institution, but the individuals as well.
A default event (a borrower’s failure to repay a loan) not only diminishes
profit for the bank; it also worsens the credit score of the borrower. A
successful lending outcome leads to profit for the bank and also to an increase
in credit score for the borrower. In our running example, the ratio of the
change in credit scores for a borrower is 1 (repaid) to -2 (defaulted).

For threshold policies, the outcome, defined as expected change in score for a
population, can be parametrized as a function of the selection rate; we call
this function the **outcome curve**. As the selection rate in one group varies, the
outcome experienced by the group also varies. These population-level outcomes
depend both on the probability of repayment (as encoded by the score), and cost
and benefit of the loaning decision to the individual.

<center>
  <img src="/assets/delayed_impact/outcome_curve_blog.png" alt="drawing" style="width: 500px;"/>
  <p>
  <i>
  Figure 3
  </i>
  </p>
</center>


The above figure shows the outcome curve for a typical population. When enough
individuals in a group are granted loans and successfully repay them, the
average credit score in that group is likely to increase. Unconstrained
profit-maximization results in a positive average score change in the population
in this case. As we deviate from profit maximization to give out loans to more
people, the average score change increases up to a certain point where it is
maximized. We may call this the *altruistic optimum*. We can also increase the
selection rate up to a point where the average score change is lower than under
unconstrained profit-maximization but still positive, as illustrated by the
dotted yellow regions. We say that selection rates in the dotted yellow regions
are causing *relative harm*. However, if too many individuals are unable to
repay their loans, the average credit score for the group will decrease, as is
the case in the red striped region.


<div id="single-curves-table">
</div>




## Multiple groups

How does a given threshold policy affect individuals across different groups?
Two groups with different distributions of credit scores will experience
different outcomes.

Suppose the second group has a distribution of credit scores that is different
from the first group, and also has fewer people. We can think of this group as a
historically disadvantaged minority. Let’s denote this group as the *blue group*;
we would like to ensure that the bank’s lending policy doesn’t hurt or
shortchange them disproportionately.

We imagine the bank can choose different thresholds for each group. While
group-dependent thresholds may face legal challenges, they are inevitable in
avoiding the differential outcomes that a fixed-threshold decision could invoke.


<div id="comparison-histogram-table">
</div>


It makes sense to ask what choices of thresholds lead to an expected improvement
in the score distribution within the blue group. As we mentioned before, an
unconstrained bank policy would maximize profit, choosing thresholds that meet a
break-even point above which it is profitable to give out loans. In fact, the
profit-maximizing threshold (a credit score of 580) is the same for
both groups.


## Fairness Criteria

Groups with different distributions over scores will have differently shaped
outcome curves (see the top half of Figure 6, which shows outcome curves
resulting from actual credit score data and a simple outcome model).  As an
alternative to unconstrained profit maximization, one might consider *fairness
constraints*, which equalize decisions between groups with respect to some
objective function. A variety of fairness criteria have been proposed to protect
the disadvantaged group by an appeal to intuition. With a model of outcomes, we
are now equipped to answer concretely whether fairness constraints actually
encourage more positive outcomes.

One frequently proposed fairness criterion, *demographic parity*, requires the
bank to lend to both groups at an equal rate. Subject to this requirement, the
bank would continue to maximize profit to the extent possible.  Another
criterion, *equality of opportunity*, equalizes the *true positive rates*
between the two groups, requiring the bank to lend in both groups at an equal
rate among individuals who will repay their loan.

While these fairness criteria are a natural way to think about equalizing static
decisions, they often ignore the future effects these policies have on
population outcomes. Figure 6 illustrates this point by contrasting the policies
resulting from max profit, demographic parity and equal opportunity. Try
selecting each loan strategy to see the bank profit and credit score change they
result in! Both demographic parity and equal opportunity reduce the bank’s
profit compared to max profit. But do they improve the outcome of the blue
population over max profit? While the max profit strategy underloans to the blue
population relative to the altruistic optimum, equal opportunity overloans
relative to the altruistic optimum, and demographic parity overloans to the
point of causing relative harm in the blue population.

<div id="comparison-curves-table">
</div>


If the goal of employing fairness criteria is to increase or equalize long term
well-being over all populations, we’ve just shown that there are scenarios where
fairness criteria actually act against this objective. In other words, fairness
constraints can also reduce welfare in already disadvantaged populations.
Constructing an accurate model to forecast the effects that
decisions have on population outcomes may help to mitigate possible
unanticipated harms of applying fairness constraints.

## Considering outcomes for “fair” machine learning

We advocate for a view toward long-term outcomes in the discussion of “fair''
machine learning. Without a careful model of delayed outcomes, one cannot
foresee the impact a fairness criterion would have if enforced as a constraint
on a classification system.  However, if an accurate outcome model is available,
there are more direct ways to optimize for positive outcomes than via existing
fairness criteria. Specifically, the outcome curve gives us a way to deviate
from the maximum profit strategy in a way that most directly improves outcomes.

An outcome model is a concrete way to incorporate domain knowledge into the
classification process. This aligns well with much scholarship that points to
the context-sensitive nature of fairness in machine learning. The outcome curve
provides an interpretable visual device to highlight application-specific
tradeoffs.

For more details, please check out [the full version of our paper][1], which will 
also appear at the 35th International Conference on Machine learning in Stockholm, Sweden. Our work
is only a start in exploring how outcome models can mitigate undesirable societal
impacts of machine learning algorithms. We believe there is much more work to be
done in order to ensure the long-term fairness of machine learning, as
algorithms impact the lives of more people.


*Acknowledgements*. The authors would like to thank Martin Wattenberg and
Fernanda Viégas for the [interactive visualizations][2] that inspired us and were used as reference
for those in this post.


[1]:https://arxiv.org/abs/1803.04383
[2]:https://research.google.com/bigpicture/attacking-discrimination-in-ml/
