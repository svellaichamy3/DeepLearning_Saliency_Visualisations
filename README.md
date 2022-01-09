## About the project
Being able to explain a decision taken by the classifier is essential for deploying it in real life. An explainable model inevitably adds more trust and transparency. 
When the system performs better than humans, it could teach a thing or two to humans too!

## Introduction
In this project, I explored the use of different type of attribution algorithms - both gradient and perturbation - for images, and understand their
differences using the Captum model interpretability tool for PyTorch.

For the project, I use four different techniques:


1. **Saliency Maps**: Saliency maps are a quick way to tell which part of the image influenced the classification decision made by the network.

2. **GradCAM**: GradCAM is a way to show the focus area on an image for a given label.

3. **Fooling Images**: We can perturb an input image so that it appears the same to humans, but will be misclassified by the pretrained network.


4. **Class Visualization**: We can synthesize an image to maximize the classification score of a particular class; this can give us some sense of what the network is looking for when it classifies images of that class.

### Saliency Map
**Concept:** When training a model, we define a loss function which measures our current unhappiness with the model’s performance; we then use backpropagation
to compute the gradient of the loss with respect to the model parameters, and perform gradient descent on the model parameters to minimize the loss.
In this project, we do something slightly different. We will start from a convolutional neural network model which has been pretrained to
perform image classification on the ImageNet dataset. We will use this model to define a loss function which quantifies our current unhappiness with our
image, then use backpropagation to compute the gradient of this loss with respect to the pixels of the image. We will then keep the model fixed, and
perform gradient descent on the _image_ to synthesize a new image which minimizes the loss. Using this pretrained model, we will compute class saliency maps as described in the paper [1].


**Method**: A saliency map tells us the degree to which each pixel in the image affects the classification score for that image. To compute it, we compute the
gradient of the unnormalized score corresponding to the correct class (which is a scalar) with respect to the pixels of the image. If the image has shape
(3, H, W) then this gradient will also have shape (3, H, W); for each pixel in the image, this gradient tells us the amount by which the classification score
will change if the pixel changes by a small amount. To compute the saliency map, we take the absolute value of this gradient, then take the maximum
value over the 3 input channels; the final saliency map thus has shape (H, W) and all entries are nonnegative. 


<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/sal_map_wm.png" width="800" height="400" />
</p>

**Captum** is a library in Pytorch built for interpretability research. Using Captum we visualise the saliency map as seen below:

<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/sal_map_captum_wm.png" width="800" height="400" />
</p>

**Guided Backprop** is a pointwise multiplication of the map with the image and highlights the exact pixels of the image that is crucial to the decision making.

<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/guided_bp_wm.png" width="700" height="150" />
</p>


### GradCam
GradCAM (which stands for Gradient Class Activation Mapping) is a technique that tells us where a convolutional network is looking when it is making
a decision on a given input image. Here, we backpropagate until the last convolutional layer to get the weights for each of the activations of the last layer. The weighted sum is resized to the image size to get the GradCam. There are three main stages to it:
1. Guided Backprop (Changing ReLU Backprop Layer)
2. GradCAM (Manipulating gradients at the last convolutional layer)
3. Guided GradCAM (Pointwise multiplication of above stages)

**GradCam Output**
<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/gradcam_wm.png" width="800" height="400" />
</p>


We use **Captum** to visualise the GradCAM:

<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/sal_map_captum_wm.png" width="800" height="400" />
</p>

Guided Gradcam
Taking a pointwise multiplication of the map with the image to get:
<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/guided_gradcam_wm.png" width="700" height="150" />
</p>

Saliency map and Gradcam tells us which region of the image is useful for the ML system (in this case a classifier) to make a decision. This gives us a possible explanation behind the decision. They explain at a pixel level making us understand the pixels/group of pixels responsible for the decision.

### Fooling Image
We can also use the similar concept of image gradients to study the stability of the network. Consider a state-of-the-art deep neural network that generalizes
well on an object recognition task. We expect such network to be robust to small perturbations of its input, because small perturbation cannot
change the object category of an image. However, [2] find that applying an imperceptible non-random perturbation to a test image, it is possible to
arbitrarily change the network’s prediction. Here, we look to visualise the phenomena. Given an image and a target class, we can perform gradient ascent over
the image to maximize the target class, stopping when the network classifies the image as the target class. We term the so perturbed examples **“adversarial
examples”**.

<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/fooling_wm.PNG" width="700" height="150" />
</p>

We do not have to change the picture drastically for the image to cause misclassification. This presents a huge vulnerability of DNNs especially in safety critical applications. I was expecting concepts to emerge out of the fooled image e.g.wings popping out of hay which could potentially explain the rationale behind the decision but I see that the visual system of humans behave differently from how the neural network ‘sees’ the image. 

### Class Visualization
By starting with a random noise image and performing gradient ascent on a target class, we can generate an image that the network will recognize as
the target class. This idea was first presented in [1]; [3] extended this idea by suggesting several regularization techniques that can improve the quality
of the generated image. Concretely, let I be an image and let y be a target class. Let s<sub>y</sub>(I) be the score that a convolutional network assigns to the image I for class y

<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/viz1.PNG" width="400" height="70" />
</p>

where R is a (possibly implicit) regularizer (note the sign of R(I) in
the argmax: we want to minimize this regularization term). We can solve
this optimization problem using gradient ascent, computing gradients with
respect to the generated image. We will use (explicit) L2 regularization of
the form

<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/viz2.PNG" width="200" height="70" />
</p>

and implicit regularization as suggested by [3] by periodically blurring the generated image. We can solve this problem using gradient ascent on the
generated image.
<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/class_viz1_wm.png" width="700" height="150" />
</p>
<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/class_viz2_wm.png" width="700" height="150" />
</p>
<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/class_viz3_wm.png" width="700" height="150" />
</p>
<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/class_viz4_wm.png" width="700" height="150" />
</p>
<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/class_viz5_wm.png" width="700" height="150" />
</p>
<p align="center">
<img src="https://github.com/svellaichamy3/DeepLearning_Saliency_Visualisations/blob/main/images/class_viz6_wm.png" width="700" height="150" />
</p>


## References

[1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. ”Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps”, ICLR Workshop 2014(https://arxiv.org/abs/1312.6034)


[2] Szegedy et al, ”Intriguing properties of neural networks”, ICLR 2014


[3] Yosinski et al, ”Understanding Neural Networks Through Deep Visualization”, ICML 2015 Deep Learning Workshop


[4] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra, Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, 2016


[5] Mukund Sundararajan, Ankur Taly, Qiqi Yan, ”Axiomatic Attribution for Deep Networks”, ICML, 2017


[6] Matthew D Zeiler, Rob Fergus, ”Visualizing and Understanding Convolutional Networks”, Visualizing and Understanding Convolutional Networks, 2013.
