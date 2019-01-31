---
layout:     post
title:      Entropy (2)
subtitle:   properties
date:       2019-01-06
author:     Lisha Chen
header-img: img/post-bg-chalkboard.png
catalog: true
tags:
    - uncertainty
    - entropy

---



In the last article, I listed the 4 basic properties that make entropy a good representation of uncertainty. Here, as a continuation of the last article, I want to list some other important properties of entropy.

---
The following property 5-7 are summarized from [1].

#### Property 5: Uniform distribution with more classes has higher entropy
$H_m>H_n$ if $m>n$. Where $m$ and $n$ are number of classes.

#### Property 6: Entropy is non-negative
$H(X) \geq 0$
with equality achieved when eg. $H(0,\dots,0,1)$.

#### Property 7: Change the order of the arguments does not change the entropy
$H(p_1,p_2) = H(p_2,p_1)$
Or to say it has some symmetry.

---
The following properties are collected from other sources. They may relate to specific distributions or specific real world problems.

#### Property 8: Gaussian has max entropy in continuous distribution with finite variance
The Gaussian distribution has max entropy compared to all continuous distributions covering the entire real line  $x\in(-\infty,\infty)$ but having a finite mean and variance.
Or to say that among all the distributions that has a fixed variance, Gaussian has the largest entropy.

#### Property 9: Shannon entropy is the average (expected) length of the random event
For a message or event with probability $p$, the most efficient (i.e. compact) encoding of that message will require $-\log_2(p)$ bits.
Then for a random event with $K$ outcomes each having probability $p_i$, the expected length of the event is $-\sum_{i=1}^K p_i \log_2 p_i$.

## References
[1] [Entropy is a measure of uncertainty](https://towardsdatascience.com/entropy-is-a-measure-of-uncertainty-e2c000301c2c), Sebastian Kwiatkowski