---
layout:     post
title:      Bayesian inference of Categorical distribution
subtitle:   
date:       2019-02-17
author:     Lisha Chen
header-img: img/post-bg-chalkboard.png
catalog: true
tags:
    - Bayesian inference
    - Bayesian Networks
    

---

## Introduction

This post will introduce some basic concepts of Bayesian inference (BI) and provide a simple example on Categorical distribution and compare the results from (frequentist) maximum likelihood estimation (MLE) and maximum a posteriori (MAP) estimation.

For this simple example with a conjugate Dirichlet prior, it will give a closed-form solution for Bayesian inference instead of widely used MCMC sampling methods. And the conclusion can directly be applied to Bayesian inference for discrete Bayesian Networks.

## Bayesian inference

Suppose $$x$$ is a random variable that follows a certain distribution with parameter $$\theta$$, i.e. $$x \sim p(x\mid \theta)$$. We observe some data $$D = \{x_1, x_2,\dots, x_N\}$$ sampled from $$p(x\mid \theta)$$. Then the likelihood of $$\theta$$ is $$p(D\mid \theta)$$, according to Bayes rule, the posterior of $$\theta$$ is 
$$p(\theta\mid D, \alpha) = \frac{p(D\mid \theta) p(\theta\mid \alpha)}{p(D)} \propto p(D\mid \theta) p(\theta\mid \alpha) $$.

The MLE of $$\theta$$ is 
$$ \theta^* = {\arg\max}_{\theta} p(D \mid \theta )$$.
The MAP estimation of $$\theta$$ is 
$$ \theta^* = {\arg\max}_{\theta} p(\theta \mid D , \alpha)$$.
After getting estimation $$\theta^*$$ from the observation $$D$$, to perform inference, $$x^* = {\arg\max}_x p(x\mid \theta^*)$$.

The Bayesian inference does not rely on a point estimation of $$\theta$$, instead, it directly uses the posterior distribution  $$p(\theta \mid D, \alpha)$$ and integrate out all possible $$\theta$$ over the distribution to get the final inference result.

To perform Bayesian inference, 
$$ x^* = {\arg\max}_{x} p(x\mid D, \alpha) = {\arg\max}_{x} \int p(x\mid \theta) p(\theta\mid D, \alpha) d\theta$$

If $$D = \emptyset $$, then we integrate over the prior $$p(\theta \mid \alpha)$$ instead of the posterior $$p(\theta \mid D, \alpha)$$.

### Categorical distribution and its conjugate prior

The probability mass function for a Categorical distribution with $$K$$ classes is

$$p(x\mid \theta) = \prod_{i=1}^{K} \theta_i^{[x=i]}$$

where $$\theta_i = p(x=i\mid \theta)$$.

Dirichlet distribution is a conjugate prior for the Categorical likelihood. Suppose the parameter for the Dirichlet distribution is $$\boldsymbol {\alpha }$$, then the prior Dirichlet distribution is 

$$p(\theta \mid \alpha) = {\frac {1}{\mathrm {B} ({\boldsymbol {\alpha }})}}\prod _{i=1}^{K}\theta_{i}^{\alpha _{i}-1}$$

The posterior distribution $$p(\theta \mid D,\boldsymbol {\alpha }) \propto p(D\mid \theta) p(\theta\mid \alpha) $$. Assuming samples $$x_j, j=1,2,\dots,N$$ are independent given parameter $$\theta$$, then $$p(D\mid \theta) = \prod_{j=1}^N p(x_j\mid \theta)$$. Therefore,

$$p(\theta \mid D,\boldsymbol {\alpha }) \propto \prod_{j=1}^N \prod_{i=1}^{K} \theta_i^{[x_j=i]}  \prod _{i=1}^{K}\theta_{i}^{\alpha _{i}-1} = \prod _{i=1}^{K}\theta_{i}^{\alpha _{i}-1 + \sum_{j=1}^N [x_j=i]}$$

We can see that $$p(\theta \mid D,\boldsymbol{\alpha})$$ still follows Dirichlet distribution with new parameter $$\tilde{\alpha}_i = \alpha _{i} + \sum_{j=1}^N [x_j=i]$$ where $$\sum_{j=1}^N [x_j=i]$$ is the count of the number of samples that belong to class $$i$$. Denote it as $$c_i$$.

$$p(\theta \mid D,\boldsymbol {\alpha }) = {\frac {1}{\mathrm {B} ({\boldsymbol {\alpha } + \mathbf{c}})}} \prod _{i=1}^{K}\theta_{i}^{\alpha _{i} + c_i -1}$$

### Bayesian inference for Categorical distribution

$$p(x = i\mid D, \boldsymbol{\alpha}) = \int p(x = i\mid \theta) p(\theta\mid D, \boldsymbol{\alpha}) d\theta = \int \theta_i p(\theta\mid D, \boldsymbol{\alpha}) d\theta = \operatorname{E}_{p(\theta\mid D, \boldsymbol{\alpha})}[\theta]_i$$

Using the mean for a Dirichlet distribution, we have

$$p(x = i\mid D, \boldsymbol{\alpha}) = \frac{\alpha_i + c_i}{\sum_{k=1}^K \alpha_k + c_k} = \frac{\alpha_i + c_i}{N + \sum_{k=1}^K \alpha_k}$$

Compared to MLE, where $$p(x = i\mid \theta^*) = \frac{c_i}{N}$$, Bayesian inference will give more accurate results if the prior is reasonable and there is not enough data. But if there is enough data, MLE will give similar results.

Compared to MAP, where $$p(x = i\mid \theta^*) = \frac{\alpha_i + c_i - 1}{N - K + \sum_{k=1}^K \alpha_k}$$, results will be very similar if the mode and mean of $$p(\theta\mid D,\mathbf{\alpha})$$ are close to each other. And will be the same if $$\alpha_i + c_i = \alpha_k + c_k, \forall i,k =1,2,\dots, K$$.

Comparing MLE and MAP, if $$\alpha_i = 1, i =1,2,\dots, K$$, their results will be the same.

|MLE|MAP|BI|
|---|---|--|
|$$\theta^* = {\arg\max}_{\theta}p(D\mid \theta)$$|$$\theta^* = {\arg\max}_{\theta}p(\theta\mid D,\boldsymbol{\alpha})$$| $$p(\theta\mid D,\boldsymbol{\alpha})$$|
|$$p(x = i\mid \theta^*)$$ | $$p(x = i\mid \theta^*) $$ | $$p(x = i\mid D, \boldsymbol{\alpha}) $$|
|$$p_i = \frac{c_i}{N}$$ | $$p_i = \frac{\alpha_i + c_i - 1}{N - K + \sum_{k=1}^K \alpha_k} $$ | $$p_i = \frac{\alpha_i + c_i }{N + \sum_{k=1}^K \alpha_k}$$ |


### Derivation of mean and mode of Dirichlet

The above discussion uses the the mean and mode of Dirichlet distribution, here I will show the derivation.

**Dirichlet**

\begin{equation}
\begin{aligned}
p(\theta\mid \boldsymbol{\alpha}) &= \frac{1}{B(\boldsymbol{\alpha})}\prod_{i=1}^{K}\theta_i^{\alpha_i-1},\, B(\boldsymbol{\alpha})= \frac{\prod_{i=1}^{K}\Gamma(\alpha_i)}{\Gamma(\sum_{i=1}^K\alpha_i)}\\
\end{aligned}
\end{equation}


#### Mean


$$\operatorname{E}[\theta_i \mid \boldsymbol{\alpha}] =\frac{\boldsymbol{\alpha}_i}
{\sum_k \boldsymbol{\alpha}_k}$$

$$
\begin{aligned}
\operatorname{E}[\theta_i \mid \boldsymbol{\alpha}]
&= \int \theta_i p(\theta \mid \boldsymbol{\alpha}) d\theta_i \\
&= \frac{1}{B(\boldsymbol{\alpha})}\int \theta_i  \prod_{k=1}^{K}\theta_k^{\alpha_k-1} d\theta \\
&= \frac{1}{B(\boldsymbol{\alpha})}\int \prod_k \theta_k^{y_k  + \alpha_k - 1} d\theta , \,  s.t. \begin{cases} y_k=1, k = i\\y_k =0, o.w. \end{cases} \\
&= \frac{B(\mathbf{y} + \boldsymbol{\alpha})}{B(\boldsymbol{\alpha})}\\
&= \frac{\prod_k \Gamma(y_k + \boldsymbol{\alpha}_k)}
{\prod_k \Gamma(\boldsymbol{\alpha}_k)}
\frac {\Gamma(\sum_k \boldsymbol{\alpha}_k)}
{ \Gamma(\sum_k y_k + \boldsymbol{\alpha}_k)}\\
&= \frac {\Gamma(1+ \boldsymbol{\alpha}_i)}
{ \Gamma(\boldsymbol{\alpha}_i)}
\frac{1}{\sum_k \boldsymbol{\alpha}_k}\\
&= \frac{\boldsymbol{\alpha}_i}
{\sum_k \boldsymbol{\alpha}_k}\\
\end{aligned}
$$

Since $$\Gamma(x+1) = x\Gamma(x)$$.



#### Mode


$$\theta_{i}^*={\arg\max}_{\theta} p(\theta\mid \boldsymbol{\alpha}) = {\frac {\alpha _{i}-1}{\sum _{k=1}^{K}\alpha _{k}-K}}$$


**Take derivative**

$$
\begin{aligned}
\theta_{i}^* &={\arg\max}_{\theta} p(\theta\mid \boldsymbol{\alpha}) = {\arg\max}_{\theta} \frac{1}{B(\boldsymbol{\alpha})}\prod_{i=1}^{K}\theta_i^{\alpha_i-1}, \, s.t.\, \sum_{i=1}^K \theta_i = 1
\end{aligned}
$$

Set derivative of log probability to zero,

$$ \frac{\alpha_i - 1}{\theta_i} - \frac{\alpha_K - 1}{1 - \sum_{k=1}^{K-1}\theta_k}= 0, i=1,2,\dots, K$$

Solving the linear equations we will get $$\theta_{i}^*= {\frac {\alpha _{i}-1}{\sum _{k=1}^{K}\alpha _{k}-K}}$$.

**Gibbs' inequality**

We could directly apply Gibbs' inequality to get $$\theta^* = {\arg\max}_{\theta} \sum_{i=1}^{K} {(\alpha_i-1)}\log \theta_i, \, s.t.\, \sum_{i=1}^K \theta_i = 1 $$.

## Extension to discrete Bayesian Networks

Denote the $$n$$-th node in the BN as $$x_n$$, the parents as $$\pi(x_n)$$ and the parameter for the node as $$\theta_n$$. The joint probability by BN is

$$ p(x \mid \theta) = \prod_{n}p(x_n\mid \pi(x_n), \theta_n)$$


$$
\begin{aligned}
p(\theta\mid D,\boldsymbol{\alpha}) 
&\propto p(D\mid \theta)p(\theta\mid \boldsymbol{\alpha})\\ 
&= \prod_{n}p(D_n\mid \theta_n) \prod_{n}p(\theta_n\mid \boldsymbol{\alpha}_n)\\
&\propto \prod_{n}p(\theta_n\mid D_n,  \boldsymbol{\alpha}_n)\\
\end{aligned}
$$

$$p(\theta\mid D,\boldsymbol{\alpha}) =\prod_{n}p(\theta_n\mid D_n,  \boldsymbol{\alpha}_n) $$ because it is a valid probability.

$$ \begin{aligned}
p(x \mid D, \boldsymbol{\alpha}) 
&= \int p(x\mid \theta) p(\theta\mid D, \boldsymbol{\alpha}) d\theta \\
&= \int \prod_{n}p(x_n\mid \pi(x_n), \theta_n) \prod_{n} p(\theta_n\mid D_n, \boldsymbol{\alpha}_n) d\theta\\
&= \prod_{n} \int p(x_n\mid \pi(x_n), \theta_n) p(\theta_n\mid D_n, \boldsymbol{\alpha}_n) d\theta_n\\
&= \prod_{n} p(x_n\mid \pi(x_n), D_n,\boldsymbol{\alpha}_n)
\end{aligned}$$

We can directly use the above conclusion in discrete Bayesian Network where each node $$x_n$$ given its parents $$\pi(x_n)$$ follows a Categorical distribution.

During learning, we get $$p_i$$ for each node given its parents from one of MLE, MAP or BI directly applying the equation for each of the 3 methods in the Table. After obtaining all the conditional probability tables (CPTs), we can get the joint probability. Given the joint distribution, we can do any inference.


