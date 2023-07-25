# Covert Communication in Autoencoder Wireless Systems

**covert-ml** is a GAN-based covert communication method that enables establishing a reliable, undetectable covert
channel within autoencoder wireless communication systems with the minimum impact.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Thesis](#thesis)
- [Contact Information](#contact-information)

## Introduction

This repository contains the project that was created throughout the course of my master's thesis, titled "Covert
Communication in Autoencoder Wireless Systems." This research project explores the concept of using generative AI to
produce covert signals within a Generative Adversarial Network (GAN) framework. The objective is to create signals that
are indistinguishable from normal signals, enabling covert communication in autoencoder wireless systems.

## Installation

To run this project locally, follow these installation steps:

1. Clone this repository to your local machine.
```bash
git clone https://github.com/ali-mohammadi/covert-ml.git
```
2. Install the required Python packages and dependencies

## Usage

Once the installation is complete, you can use the project for training the autoencoder, covert models, and
evaluating their results. The main scripts for each step are provided in the project repository. Here's an overview of
how
to use the project:

1. **Autoencoder Training/Evaluation:** Execute the `wireless_autoencoder/siso/train.py`
   or `wireless_autoencoder/mimo/train.py` scripts to train the autoencoder using the prepared
   dataset. Model hyperparameters can be modified from `parameters.py` as needed for your specific use case.

2. **Covert Models Training/Evaluation:** After successfully training the autoencoder, use
   the `wireless_covert/siso/train.py`
   or `wireless_covert/mimo/train.py` scripts to train the covert models. Similarly, hyperparameters can be modified
   from `parameters.py` as needed for your specific use case.

You can also change the channel configuration from `shared_parameters.py`

## Experimental Setup

Covert and autoencoder wireless models were trained on a high-
performance desktop computer with an RTX 3080Ti graphics card. The desktop configuration included an Intel Core i7
10700KF 3.80GHz CPU, 16GB of DDR4
3200MHz RAM, and 12GB of GPU memory.

Although training the models using a high performance GPU significantly reduce the training time, we have
provided support
for both CPU and GPU training in this project.

## Thesis

### Abstract

The broadcast nature of wireless communications presents security and privacy challenges.
Covert communication is a wireless security practice that focuses on intentionally hiding
transmitted information. Recently, wireless systems have experienced significant growth,
including the emergence of autoencoder-based models.
These models, like other DNN
architectures, are vulnerable to adversarial attacks, highlighting the need to study their
susceptibility to covert communication. While there is ample research on covert communication
in traditional wireless systems, the investigation of autoencoder wireless systems remains
scarce. Furthermore, many existing covert methods are either detectable analytically or
difficult to adapt to diverse wireless systems.
The first part of this thesis provides a comprehensive examination of autoencoder-based
communication systems in various scenarios and channel conditions. It begins with an
introduction to autoencoder communication systems, followed by a detailed discussion of
our own implementation and evaluation results.
This serves as a solid foundation for
the subsequent part of the thesis, where we propose a GAN-based covert communication
model. By treating the covert sender, covert receiver, and observer as generator, decoder,
and discriminator neural networks, respectively, we conduct joint training in an adversarial
setting to develop a covert communication scheme that can be integrated into any normal
autoencoder. Our proposal minimizes the impact on ongoing normal communication, addressing
previous works shortcomings. We also introduce a training algorithm that allows for the
desired tradeoff between covertness and reliability. Numerical results demonstrate the establishment
of a reliable and undetectable channel between covert users, regardless of the cover signal or
channel condition, with minimal disruption to the normal system operation.

### Conclusion

Our covert model successfully
demonstrated its ability to embed secret messages into covert signals without relying on
handcrafted features. Through the utilization of the generative adversarial training framework,
we significantly reduced the detection probability of the covert signals produced. Furthermore,
we proposed a training procedure that allows us to adjust the trade-off between the conflicting objectives of covert
communication, which are reliability of communication and the probability
of detection. This adjustment is achieved by introducing regularizers into the model’s loss
function, independent of channel conditions or user messages. Our findings demonstrate
that our covert model is channel-agnostic and insensitive to cover signals. To evaluate the
performance of our model, we conducted assessments across three channel models: AWGN,
Rayleigh, and Rician fading. We varied covert rates and the number of system users to
analyze the model’s robustness. Additionally, we investigated the impact of our covert
signals on the ongoing normal system and confirmed that our covert scheme causes minimal
disruption to the system.

## Published Papers
[Covert Communication in Autoencoder Wireless Systems](https://arxiv.org/abs/2307.08195) (Ali Mohammadi Teshnizi, Majid Ghaderi, Dennis Goeckel)

## Contact Information

If you have any questions, suggestions, or inquiries related to this project or thesis, feel free to contact me at:

- Ali Mohammadi: ali.mohammaditeshniz@ucalgary.ca
