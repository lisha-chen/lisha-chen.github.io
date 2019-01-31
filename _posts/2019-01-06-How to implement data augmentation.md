---
layout:     post
title:      data augmentation
subtitle:   properties
date:       2019-01-06
author:     Lisha Chen
header-img: img/post-bg-chalkboard.png
catalog: true
tags:
    - deep learning
    

---


This article trys to discuss about implementation of data augmentation. Because during my study of deep learning, I encountered some questions for data augmentation.

### Q1: Do we generate augmented images and added to the training images before actually doing the training or do we augment images on the fly during training?


### Q2: Do




>Q1: are the images being distorted actually added to the pool of original images?
A1: It depends on the definition of the pool. In tensorflow, you have ops which are basic objects in your network graph. here, data production is an op itself. Thus you do not have a finite set of training samples, instead you have a potentialy infinite set of samples generated from the training set.

>Q2: or are the distorted images used instead of the originals?
A2: As you can see from the source you included - sample is taken from the training batch, then it is randomly transformed, thus there is very small probability of using unaltered image (especially that cropping is used, which always modifies).

>Q3: how many distorted images are being produced? (i.e. what augmentation factor was defined?)
A3: There is no such thing, this is never ending process. Think about this in terms of random access to possibly infinite source of data, as this is what is efficiently happening here. Every single batch can be different from the previous one.



AutoAugment: Learning Augmentation Policies from Data”
Bayesian data augmentation
Learning to Compose Domain-Specific Transformations for Data Augmentation
Improved Regularization of Convolutional Neural Networks with Cutout
mixup: BEYOND EMPIRICAL RISK MINIMIZATION
Shake-Shake regularization (adding noise to gradient, similar to adversarial training?)