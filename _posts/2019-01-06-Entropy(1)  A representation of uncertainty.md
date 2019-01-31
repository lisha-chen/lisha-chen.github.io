---
layout:     post
title:      Entropy (1)
subtitle:   uncertainty
date:       2019-01-06
author:     Lisha Chen
header-img: img/post-bg-chalkboard.png
catalog: true
tags:
    - uncertainty
    - entropy

---



I first learned entropy in physics, which in thermodynamics, measures energy dispersal at a specific temperature. Later I learned about the term information entropy in communication. And then in machine learning, it is widely used as a representation of uncertainty.
Here we are talking about Shannon entropy, whose definition is $$-\sum_{i=1}^K p_i\log_2 p_i$$. There are other measures of uncertainty, but for some reason, people choose Shannon entropy more often for its good properties [1].

The following are mainly summarized and extended from [1].

### What are the basic properties of Shannon entropy?
#### Property 1: Uniform distribution has max entropy
This can be proved using Weighted AM–GM inequality [2]:
$$\frac{w_{1}x_{1}+w_{2}x_{2}+\cdots +w_{n}x_{n}}{w}\geq \sqrt[{w}]{x_{1}^{w_{1}}x_{2}^{w_{2}}\cdots x_{n}^{w_{n}}}$$
by letting $$\frac{w_i}{w}=p_i$$, $$x_i=\frac{1}{p_i}$$.

#### Property 2: Additivity of independent events
To formulate this property in math equations, we have
$$H(X,Y) = H(X) + H(Y)$$, if $$X\perp Y$$

Another function $$-\sum_{i=1}^K p_i^2$$, which satisfies the first property, does not satisfy this one. That's why trace of covariance as a representation of uncertainty may not be as good as entropy.

#### Property 3: Zero-prob outcome does not contribute to entropy
$$H(p_1,p_2,\dots,p_n) = H(p_1,p_2,\dots,p_n,p_{n+1}=0)$$

#### Property 4: Continuity in all arguments
Some other measurements also satisfies this property, such as trace of covariance matrix, determinant of covariance matrix.

Note: there is a **Uniqueness Theorem** [1]
>Khinchin (1957) showed that the only family of functions satisfying the four basic properties described above are of the following form:
$$H(p_1,p_2,\dots,p_K)=-\lambda\sum_{i=1}^K p_i\log_2 p_i$$
Functions that satisfy the 4 basic properties
where $$\lambda$$ is a positive constant. Khinchin referred to this as the Uniqueness Theorem. Setting $$\lambda = 1$$ and using the binary logarithm gives us the Shannon entropy.
To reiterate, entropy is used because it has desirable properties and is the natural choice among the family functions that satisfy all items on the basic wish list (properties 1–4).


Besides the above discussion for the basic 4 properties of entropy, there are some other interesting facts about entropy that I will explore later.

## References
[1] [Entropy is a measure of uncertainty](https://towardsdatascience.com/entropy-is-a-measure-of-uncertainty-e2c000301c2c), Sebastian Kwiatkowski

[2] [Inequality of arithmetic and geometric means](https://en.wikipedia.org/wiki/Inequality_of_arithmetic_and_geometric_means#Weighted_AM.E2.80.93GM_inequality), Wikipedia