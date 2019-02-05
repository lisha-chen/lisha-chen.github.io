---
layout:     post
title:      Understanding dropout (1)
subtitle:   Implementation & properties
date:       2019-01-06
author:     Lisha Chen
header-img: img/post-bg-chalkboard.png
catalog: true
tags:
    - Bayesian inference
    - deep learning
    - regularization

---




## What is dropout

Dropout is a technique used in deep neural network training and typically understood as a regularization technique to penalize the model complexity and reduce overfitting. It was first proposed in [1]. Although first proposed in feed forward neural networks, this can also be used in graphical models such as Boltzmann Machines [3].

### Properties

In the paper [1], it is discussed that using dropout has following properties

>In networks with a single hidden layer of $$N$$ units and a “softmax” output layer for computing the probabilities of the class labels, using the mean network is exactly equivalent to taking the geometric mean of the probability distributions over labels predicted by all $$2^N$$ possible networks. [1]

the network with a single hidden layer: $$\mathbf{x}\rightarrow \mathbf{h} \rightarrow \mathbf{f}(\mathbf{h}) \rightarrow s(\mathbf{f}(\mathbf{h}))$$
where $$\mathbf{f}(\mathbf{h})=W\mathbf{h}+b, s(\cdot)$$ represents softmax.

Denote $$\mathbf{h}_j$$ as the hidden layer of the $$j$$-th network resulting from dropout, $$j=1,\dots,2^ N$$, and $$\mathbf{h}$$ as the hidden layer of the  network with all the nodes, then $$\operatorname{E}[\mathbf{h}_j] = \frac{1}{2}\mathbf{h}$$. 

For a testing sample $$\{\mathbf{x},\mathbf{y}\}$$, the probability from the mean network is $$p(\mathbf{y}\mid \mathbf{x};\theta) = \prod_k s(\mathbf{f}_k(\frac{1}{2}\mathbf{h}))^{[\mathbf{y}=k]}$$. And the probability from the $$j$$-th network is $$p(\mathbf{y}\mid \mathbf{x};\theta_j) = \prod_k s(\mathbf{f}_k(\mathbf{h}_j))^{[\mathbf{y}=k]}$$. The expectation of the log of the dropout networks' predicted probabilities is $$\operatorname{E}[\log p(\mathbf{y}\mid \mathbf{x};\theta_j)] = \operatorname{E}\big[ \sum_k [\mathbf{y}=k] \log s(\mathbf{f}_k(\mathbf{h}_j))\big] = \sum_k [\mathbf{y}=k] \operatorname{E}\big[\log s(\mathbf{f}_k(\mathbf{h}_j))\big]$$. $$\because \log s(\mathbf{f}_k(\mathbf{h}_j)) = \mathbf{f}_k(\mathbf{h}_j) - \log \sum_k \exp(\mathbf{f}_k(\mathbf{h}_j)) = \mathbf{f}_k(\mathbf{h}_j) - C_j$$ , $$\therefore \operatorname{E}[\log p(\mathbf{y}\mid \mathbf{x};\theta_j)] = \sum_k [\mathbf{y}=k] \operatorname{E}\big[\mathbf{f}_k(\mathbf{h}_j) - C_j\big] = \sum_k [\mathbf{y}=k]\big(\mathbf{f}_k( \operatorname{E}[\mathbf{h}_j]) -  \operatorname{E}[C_j] \big)$$
Therefore $$\log p(\mathbf{y}\mid \mathbf{x};\theta) = \operatorname{E}[\log p(\mathbf{y}\mid \mathbf{x};\theta_j)] + constant$$.

>Assuming the dropout networks do not all make identical predictions, the prediction of the mean network is guaranteed to assign a higher log probability to the correct answer than the mean of the log probabilities assigned by the individual dropout networks. Similarly, for regression with linear output units, the squared error of the mean network is always better than the average of the squared errors of the dropout networks. [1]

This can be shown by Jensen's inequality for concave function $$\varphi$$, $$\varphi \left(\operatorname {E} [X]\right)\geq \operatorname {E} \left[\varphi (X)\right]$$, (Since $$\log softmax(\cdot)$$ is concave). This is to claim that using the mean network, the performance is better than the expected performance of using one of the dropout networks. However, this is only true for concave function. For more general neural networks with many layers of dropout, the function is neither concave nor convex, so it is not necessarily true.

### Connection to other techniques

In [1,3], the authors gave explainations to why dropout works and the connection to other widely used techniques including Bagging, Naive Bayes, Bayesian model averaging. 

#### Cutout
If we apply dropout at the input layer for input being an image, it is similar to applying data augmentation technique: cutout. The difference is that cutout drops out contiguous sections of inputs rather than individual pixels, which was reported to be able to reduce co-adaptation of spatial features better [2].

#### Bagging

>A popular alternative to Bayesian model averaging is “bagging” in which different models are trained on different random selections of cases from the training set and all models are given equal weight in the combination. Dropout can be seen as an extreme form of bagging in which each model is trained on a single case and each parameter of the model is very strongly regularized by sharing it with the corresponding parameter in all the other models. This is a much better regularizer than the standard method of shrinking parameters towards zero. [1]

For each minibatch, it forms 1 dropout network, whose parameter update is only based on the minibatch data. Training a neural network with dropout can be seen as training a collection of $$2^N$$ thinned networks with extensive weight sharing, where each thinned network gets trained very rarely, if at all [3].

#### Naive Bayes

>A familiar and extreme case of dropout is “naive bayes” in which each input feature is trained separately to predict the class label and then the predictive distributions of all the features are multiplied together at test time. When there is very little training data, this often works much better than logistic classification which trains each input feature to work well in the context of all the other features. [1]

An example of naive Bayes for 2 features, the BN structure $$x_1\leftarrow y \rightarrow x_2$$, and the joint probability $$p(y, x_1, x_2) = p(y)p(x_1\mid y)p(x_2\mid y)$$. Then $$p(y\mid x_1, x_2) = \frac{p(y\mid x_1)p(y\mid x_2)}{p(y)}\frac{p(x_1)p(x_2)}{p(x_1,x_2)}$$. If we assume $$p(y)$$ is uniform then the model using all the features $$p(y\mid x_1, x_2)$$ is proportional to the product of the models using individual features $$p(y\mid x_i)$$ (The product is different from the geometric mean, but the idea is similar).

#### Adding noise to the units

>The idea of adding noise to the states of units has previously been used in the context of Denoising Autoencoders (DAEs) by Vincent et al. (2008, 2010) where noise is added to the input units of an autoencoder and the network is trained to reconstruct the noise-free input. Dropout can be seen as a stochastic regularization technique. [3]

The idea of adding noise to NN is also present in other works for regularization. The most simple example is adding random noise to the input as a data augmentation approach. There is also theoretical proof that training with noise is equivalent to certain types of regularization [5].


#### Bayesian model averaging

>Dropout can be seen as a way of doing an equally-weighted averaging of exponentially many models with shared weights. On the other hand, Bayesian neural networks (Neal, 1996) are the proper way of doing model averaging over the space of neural network structures and parameters. In dropout, each model is weighted equally, whereas in a Bayesian neural network each model is weighted taking into account the prior and how well the model fits the data, which is the more correct approach. [3]

It is proposed in [1,3] that sampling multiple dropout networks and combining their predictions can be interpreted as Bayesian model averaging. And in [4], it is proved that MC dropout is equivalent to variational inference for Bayesian neural networks.

Basically the MC dropout performs the same random dropout procedure during training and testing, with or without scaling. By performing MC dropout at test time, we obtain multiple sets of neural network parameters $$\theta_s$$, in [4] it is derived that they are equivalent to being sampled from a variational distribution $$q(\theta \mid D)$$, so that the final predicted probability is computed by 

$$ p(y\mid x, D) \approx \sum_s p(y\mid x,\theta_s) $$


## Dropout implementation


### Implementation 1: Mean network

Input:
Set dropout keep probability $$p_i$$ for each layer $$i$$. Randomly initialize NN parameters.
Training:
  * Step1: For each batch of training data, randomly dropout units (nodes) with probability $$1-p_i$$ in layer $$i$$, along with its links (weights).
  Note: it is equivalent to multiplying a mask with 0s and 1s where $$P(h=1)=p_i$$
  * Step2: Optimize the objective function starting from the current weights by gradient descent and get updated weights for the remaining units.
  * Step3: For each batch of training data, repeat Step1 to Step2 until convergence.
  
Testing:
Multiply each hidden layer $$i$$ with probability $$p_i$$ while performing the forward inference.

### Implementation 2: Inverted dropout

Input:
Set dropout keep probability $$p_i$$ for each layer $$i$$. Randomly initialize NN parameters.
Training:
  * Step1: For each batch of training data, randomly remove units (nodes) with probability $$p_i$$, in each layer, along with its links (weights).
  * Step2: Optimize the objective function starting from the current weights by gradient descent and get updated weights for the remaining units.
  * Step3: Divide the node value by $$p_i$$.
	Note: this is to keep the expected node value to be the same as without dropout. (Do not have the scaling problem during test time.)
  * Step4: For each batch of training data, repeat Step1 to Step3 until convergence.

Testing:
Directly perform the forward inference.

I checked the document and source code of Tensorflow/Keras, they all follow the implementation2: inverted dropout.



## References

[1] Improving neural networks by preventing co-adaptation of feature detectors\\
[2] Improved Regularization of Convolutional Neural Networks with Cutout\\
[3] Dropout: A Simple Way to Prevent Neural Networks from Overfitting\\
[4] Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning\\
[5] Training with Noise is Equivalent to Tikhonov Regularization