---

layout: post
title:  "Fixing Mode Collapse in GANs using Guided Latent Spaces"
hero: ../uploads/model1.png
overlay: red
published: true

---

## Introduction

Generative Adversarial Models have really kicked off ever since Ian Goodfellow published his paper on them a few years ago. There have been major advancements and vast applications of GANs including Wasserstein GANs, Stack GANs, Wave GANs, etc. This study is an effort to understand one of the key underlying problems of the GAN architecture, known as "mode collapse" and to attempt at preventing and/or removing mode collapse from GANs in general.

For this final objective, the following analysis is quintessential in understanding the situation in it's entireity :

<ul>
<li>How do Generative and Discriminative Networks work? What is their "learning curve"?</li>
<li>What is the cause of mode collapse?</li>
<li>Personal attempt to fix the problem</li>
</ul>

The subsequent sections shall span my explorations in all of the aforementioned subtopics, followed by my outlook and attempt to tackle the problem at hand. 

## Discriminator networks - functioning

The following is the loss function while training a Discriminator network during GAN training: (this is the Vanilla setup, but the intuition is similar for more recent GAN setups as well)

<img src="../uploads/dsicriminator_loss.PNG"> 

In simpler terms, we are providing the Discriminator with both real and generated samples of data, and penalizing it for predicting the real samples to be fake, and the generated samples to be real. The tricky part here is that, we aren't really making sure that the Discriminator ONLY learns the features of the real data distribution, and hence be able to distinguish it from the the generated data (although this might be the intended solution). The data distribution generated from the generator network is also impacting the learning process of the discriminator in unexpected ways, which is a part of the mode collapse problem. We'll come back to this later while discussing mode collapse.

<div style="color: red">
It is extremely important to include the generated image part in the Discriminator loss function, as this induces a lot of stability and leads to consistent Discriminator outputs of real to fake probability, to be used eventually for the Generator Network. Hence, we cannot simply remove that aspect of the loss function.
</div>
{: .notice-alert}

## Generator networks - functioning

The following is the loss function while training a Generator network during GAN training:

<img src="../uploads/generator_loss.PNG"> 

Diving a bit deeper into the problem, the basic intention of the loss function is to use the Discriminator gradients as a means for the Generator Network to converge to a mapping to the real data distributions. As we train the Generator by feeding ion random noise inputs  to generate from and computing the loss function on the generated data, the Nash Equilibrium expectation is that the Discriminator outputs a probability of 0.5 for both real and generated images, as they become indistinguishable from one another. 

The Generator Network (after training) is basically a mapping from a "latent space" of the dimensions of our Generator input, to the "real space" in which the real data distribution is depicted. Every random input to the Generator then corresponds to an output generated data point to be weighed as real or fake by the Discriminator. Here comes the kicker to all this, and to the method that I will be proposing : <b>The Generator Network learns certain directions (linear/non-linear) or rather, lower dimensional regions, to map to a certain specific feature of the real data distribution in order to fool the discriminator regardless of the input fed into it.</b> The following picture is a t-SNE latent space on the MNIST dataset, wherein clusters are formed for similar digits, indicating that the mapping function in this case has been able to extract features meaningful to human beings and encode it into regions of the latent space.

<img src="../uploads/t-sne.png"> 

The fact that meaningful features have been encoded into the various directions/regions of the latent space can be showing by doing arithmetic on the latent space representations of known data points as mapped by a trained function approximator :

//arithmetic latent space

This is a similarity between Autoencoders and GANs. For Autoencoders, the Encoder is forced to extract meaningful information from an input data point, and encode the point into its latent space, for the Decoder to eventually decode back into the real space. In GANs, the Generator Network is forced to attribute regions in it's input latent space to meaningful features of the real data distribution to which it maps out random inputs points to, for convergence of generating real-seeming data points (fooling the Discriminator).

//Generator latent space depiction

A fairly important point to be noted here is that the mapping from the "latent space" to the "real space" is many to one, implying that multiple points in the former can lead to the same result in the latter.

## Mode Collapse 

Now that we have some intuition about the learning curves and Nash Equilibrium of GAN training, for both the Generator and the Discriminator Network, let's take a look at mode collapse. 

1) Suppose that we are presented with the following data distribution :

<img src="../uploads/real_data.jpg"> 

2) Our Generator Network starts producing the following data distribution and starts fooling the Discriminator:

<img src="../uploads/generated_data.jpg"> 

3) Here, we link back to the part we discussed about how the Discriminator is forced into learning that the generated data is fake. After some training, the Discriminator counters this by learning that the following data points are real :

<img src="../uploads/real_points.jpg"> 

and essentially starts guessing on the following data, as it is indistinguishable in it's realness or fakeness :

<img src="../uploads/real_guess.jpg"> 

4) The Generator then exploits the Discriminator by starting to generate another cluster that resembles a part of the real data distribution :

<img src="../uploads/generate_2.jpg"> 

...and this whole process repeats for ever changing "modes" of the data distribution. This is mode collapse. 

This game of cat-and-mouse repeats ad nauseum, with the generator never being directly incentivised to cover both modes. In such a scenario the generator will exhibit very poor diversity amongst generated samples, which limits the usefulness of the learnt GAN. In reality, the severity of mode collapse varies from complete collapse (all generated samples are virtually identical) to partial collapse (most of the samples share some common properties). Unfortunately, mode collapse can be triggered in a seemingly random fashion, making it very difficult to play around with GAN architectures.

Clearly, Mode Collapse is an <b>architectural</b> problem.

## The fix

In my attempt, I view the basic problem of mode discovery as well as mode retention to fool the Discriminator. The most effective way to understand the real data distribution is via the use of a neural network classifier appropriately trained on the data. Let's assume we have a given data distribution, on which we fully train a classifier. We are now dealing with three spaces, the space with the real data distribution represented in the standard form, the calssification space of the newly trained network, and the space from which our generator maps out of. 

The idea I propose is to develop a mechanism for the generator to be able recognise to various possible data clusters/modes at the same time, without constantly trying to just fool the discriminator. For this, we jump into a latent space, with extended dimensions as compared to a latent space dimensionality for the standard benchmark of the given dataset, intentionally. We then come up with a way to sectionalise the given space, into smaller subspaces, which then would be used to train the GAN as the regular generator method procedure. 

This is done because given the standard latent space dimensions, we can make sure that the generator has enough scope to map out the entire dataset, making our subspaces large (dimensionally) enough for the generator to work with. But, mode collapse cannot be prevented here, simply because it is an inherent fault in the training procedure. Now, each of these subspaces is restricted to generate data of one mode specifically, with a penalizing factor that we can account for by using our classifier network.

To provide further clarity, let us analyze the proposed model on the MNIST dataset for the scope of this article. Firstly, we train a CNN classifier on the dataset to near perfection. 

From then on, the following diagram depicts the training loop of the modified GAN being proposed.

<img src="../uploads/training_loop.jpg"> 

//loss function

As seen, the loss function resembles closely the Vanilla GAN implementation, but with changes in the Generator loss function, now incorporating the losses due to CNN classifier as well as statistical parameters per batch, to maintain variety in the outputs per class. If the statistical loss aspect is disregarded, the Generator could just output a single digit for each class, regardless of the random input being provided to it, just based on the class label part of the input, and be able to bypass both the classifier and real/fake discriminators with ease. The values of the variances are taken to be equal to the real data variance as measured by the classifier.

The following shows the intended mapping for the Generator from the latent space (extended) to the real space.

<img src="../uploads/mapping.jpg"> 

## Results 

Coming soon.

## Further possibilities 

The following studies are worth looking into :
<ul>
<li>The penalizing factor from the classifier</li>
<li>Getting rid of the Discriminator entirely</li>
<li>Intetionally weighting the gradient contributions to the loss functions</li>
<li>Dimensional study, deciding the dimensions of the encoder vs. latent space</li>
</ul>

## References 

