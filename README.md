# Grey-Scale-Image-Colorization

## Arhitecture
1. Conditional Generative Adversarial Networks (cGANs)

The base model will be a conditional GAN (cGAN). In this model, a generator network learns to map grayscale images to colorized images. The generator takes a grayscale image as input and produces a 2-channel image (for *a and *b color channels in Lab color space).

The discriminator network then takes these generated color channels, combines them with the original grayscale image to form a 3-channel image, and learns to classify these images as "real" or "fake". Real images for the discriminator will be true color images, while fake images will be the ones generated by the generator network.

The grayscale images serve as a "condition" for both the generator and the discriminator. Both networks are expected to take this condition into account when generating and evaluating images, respectively.

In addition, the architecture includes dropout layers to introduce noise into the generator network.

2. Additional Architectural Features

Several architectural features will be used to enhance the capabilities of the model:

Residual Connections: These are used to allow the gradient to flow directly through several layers, alleviating the vanishing gradient problem and enabling the model to learn more complex functions.

Dilated Convolutions: This allows the model to capture a wider context without increasing computation or parameter count.

Attention Mechanisms: These allow the model to focus on specific parts of the image when colorizing a particular pixel, improving the ability to capture dependencies between different parts of the image.

3. Loss Function


The GAN Loss function for this architecture will incorporate elements to address the inherent color imbalance in image colorization tasks. This is done by replacing the standard cross entropy loss with a weighted loss.

This imbalance arises because the distribution of colors in images often favors desaturated or greyish colors, making colorful colors less common and thus less favored in the standard loss calculation. By using a weighted loss based on the prior probabilities of the color classes, we can counterbalance this preference.

The weights for the loss function are determined from the inverse log-probability of the color class distribution, which is calculated from a pre-existing 'prior_probs.npy' file. The idea here is to assign more importance to underrepresented (i.e., more colorful) classes.

By doing so, we reduce the chance that these underrepresented classes will be neglected by the model during training. This helps the model to produce more vibrant and diverse colors in the final colorized images, enhancing their aesthetic quality.

This approach has been utilized and discussed in detail in the paper "Colorful Image Colorization" by Richard Zhang et al.

In summary, the GAN loss for this architecture will be a combination of the adversarial loss and the weighted loss. The adversarial loss encourages the generator to produce colorized images that the discriminator cannot distinguish from real images, while the weighted loss rebalances the distribution of predicted colors towards more vibrant and diverse colors. This loss combination should make the model more effective at colorizing grayscale images in a visually pleasing manner.