# DCGAN_mnsit
This code implements a DCGAN to generate realistic images of handwritten digits from the MNIST dataset, training the generator and discriminator alternatively for a set number of epochs.

This code implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images of handwritten digits from the MNIST dataset. Here's a breakdown of the code:

1. **Imports**:
   - Necessary libraries are imported, including layers from Keras, TensorFlow, tqdm for progress bars, numpy, and functions to load and plot the MNIST dataset.

2. **Data Processing**:
   - The MNIST dataset is loaded and preprocessed.
   - Images are normalized and reshaped to be in the range [-1, 1] and with shape `(28, 28, 1)`.

3. **Discriminator Creation**:
   - A discriminator model is created using convolutional layers with LeakyReLU activations and dropout layers.
   - It outputs a single value representing the probability of the input being real.
   - The discriminator is compiled with binary cross-entropy loss and the Adam optimizer.

4. **Generator Creation**:
   - A generator model is created using transpose convolutional layers with BatchNormalization and ReLU activations.
   - It takes random noise as input and generates images with the same dimensions as MNIST digits.
   - The generator is compiled with binary cross-entropy loss and the Adam optimizer.

5. **Combining Generator and Discriminator to Form GAN**:
   - The generator and discriminator are combined sequentially to form the GAN model.
   - The discriminator's trainable parameter is set to `False` to prevent it from being trained during the generator training.
   - The GAN is compiled with binary cross-entropy loss and the Adam optimizer.

6. **Training**:
   - The GAN is trained for a specified number of epochs.
   - In each epoch, for each batch in the training dataset, the generator and discriminator are alternately trained.
   - The discriminator is first trained on real and fake images with corresponding labels.
   - Then, the generator is trained to generate images that fool the discriminator.

7. **Saving the Generator Model**:
   - After training, the generator model is saved to a file named "Generator.h5".

This code demonstrates the implementation of a DCGAN for generating realistic-looking handwritten digits.
