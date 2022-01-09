## About the project

In this project I got a chance to explore the use of different type of attribution algorithms - both gradient and perturbation - for images, and understand their
differences using the Captum model interpretability tool for PyTorch.


When training a model, we define a loss function which measures our current unhappiness with the model’s performance; we then use backpropagation
to compute the gradient of the loss with respect to the model parameters, and perform gradient descent on the model parameters to minimize the loss.
In this homework, we will do something slightly different. We will start from a convolutional neural network model which has been pretrained to
perform image classification on the ImageNet dataset. We will use this model to define a loss function which quantifies our current unhappiness with our
image, then use backpropagation to compute the gradient of this loss with respect to the pixels of the image. We will then keep the model fixed, and
perform gradient descent on the image to synthesize a new image which minimizes the loss.

We will explore four different techniques:
1. Saliency Maps: Saliency maps are a quick way to tell which part of the image influenced the classification decision made by the network.
2. GradCAM: GradCAM is a way to show the focus area on an image for a given label.
3. Fooling Images: We can perturb an input image so that it appears the same to humans, but will be misclassified by the pretrained network.
4. Class Visualization: We can synthesize an image to maximize the classification score of a particular class; this can give us some sense of
what the network is looking for when it classifies images of that class.

### Saliency Map
Using this pretrained model, we will compute class saliency maps as described in the paper:

A saliency map tells us the degree to which each pixel in the image affects the classification score for that image. To compute it, we compute the
gradient of the unnormalized score corresponding to the correct class (which is a scalar) with respect to the pixels of the image. If the image has shape
(3, H, W) then this gradient will also have shape (3, H, W); for each pixel in the image, this gradient tells us the amount by which the classification score
will change if the pixel changes by a small amount. To compute the saliency map, we take the absolute value of this gradient, then take the maximum
value over the 3 input channels; the final saliency map thus has shape (H, W) and all entries are nonnegative.


### GradCam
GradCAM (which stands for Gradient Class Activation Mapping) is a technique that tells us where a convolutional network is looking when it is making
a decision on a given input image. There are three main stages to it:
1. Guided Backprop (Changing ReLU Backprop Layer, Link)
2. GradCAM (Manipulating gradients at the last convolutional layer,
Link)
3. Guided GradCAM (Pointwise multiplication of above stages)



### Fooling Image
We can also use the similar concept of image gradients to study the stability
of the network. Consider a state-of-the-art deep neural network that generalizes
well on an object recognition task. We expect such network to be
robust to small perturbations of its input, because small perturbation cannot
change the object category of an image. However, [2] find that applying
an imperceptible non-random perturbation to a test image, it is possible to
arbitrarily change the network’s prediction.


### Class Visualization
By starting with a random noise image and performing gradient ascent on
a target class, we can generate an image that the network will recognize as
the target class. This idea was first presented in [1]; [3] extended this idea
by suggesting several regularization techniques that can improve the quality
of the generated image.
Concretely, let I be an image and let y be a target class. Let sy(I) be the
score that a convolutional network assigns to the image I for class y; note
that these are raw unnormalized scores, not class probabilities. We wish to
generate an image I that achieves a high score for the class y by solving the
problem
![eqn1]("https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/viz1.PNG")

where R is a (possibly implicit) regularizer (note the sign of R(I) in
the argmax: we want to minimize this regularization term). We can solve
this optimization problem using gradient ascent, computing gradients with
respect to the generated image. We will use (explicit) L2 regularization of
the form
![eqn1]("https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/viz2.PNG")
and implicit regularization as suggested by [3] by periodically blurring
the generated image. We can solve this problem using gradient ascent on the
generated image.


## References
1. Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. ”Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps”, ICLR Workshop 2014.
2. Mukund Sundararajan, Ankur Taly, Qiqi Yan, ”Axiomatic Attribution for Deep Networks”, ICML, 2017
3. Matthew D Zeiler, Rob Fergus, ”Visualizing and Understanding Convolutional Networks”, Visualizing and Understanding Convolutional Networks, 2013.
4.Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra, Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, 2016

[1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. ”Deep
Inside Convolutional Networks: Visualising Image Classification Models and
Saliency Maps”, ICLR Workshop 2014(https://arxiv.org/abs/1312.6034)
[2] Szegedy et al, ”Intriguing properties of neural networks”, ICLR 2014
Given an image and a target class, we can perform gradient ascent over
the image to maximize the target class, stopping when the network classifies
the image as the target class. We term the so perturbed examples “adversarial
examples”.
[3] Yosinski et al, ”Understanding Neural Networks Through Deep Visualization”,
ICML 2015 Deep Learning Workshop
1. Szegedy et al, ”Intriguing properties of neural networks”, ICLR 2014
2. Yosinski et al, ”Understanding Neural Networks Through Deep Visualization”,
ICML 2015 Deep Learning Workshop
[1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. ”Deep
Inside Convolutional Networks: Visualising Image Classification Models and
Saliency Maps”, ICLR Workshop 2014.
