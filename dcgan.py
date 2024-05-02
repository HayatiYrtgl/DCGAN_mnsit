from keras.layers import *
import tensorflow as tf
from tqdm import tqdm as progress_bar
from keras.datasets import mnist
import numpy as np
from keras.utils import plot_model
from keras.models import Sequential


# load dataset and process it
def create_dataset(batch_size):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    batch_size = batch_size

    my_data = x_train

    my_data = my_data / 255

    # normalize

    my_data = my_data.reshape(-1, 28, 28, 1) * 2 - 1

    print(my_data.shape, my_data.min(), my_data.max())

    # dataset preprocessing
    dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    dataset = dataset.prefetch(1)

    return dataset, batch_size


train_dataset, batch_size = create_dataset(32)


# discriminator creation
def create_discriminator():
    """this method creates discriminator"""
    discriminator = Sequential(name="DISCRIMINATOR")

    discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(),
                             input_shape=(28, 28, 1)))
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same", activation=LeakyReLU()))

    discriminator.add(Dropout(0.3))

    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation="sigmoid"))

    discriminator.compile(loss="binary_crossentropy", optimizer="adam")

    print(discriminator.summary())

    return discriminator


# generator function
def create_generator(coding_size=200):
    """this method create generator"""
    generator = Sequential(name="GENERATOR")

    generator.add(Dense(7*7*28, input_shape=(coding_size,)))
    generator.add(Reshape((7, 7, 28)))

    generator.add(BatchNormalization())

    generator.add(Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"))

    generator.add(Conv2DTranspose(128, 3, padding="same", strides=2, activation="relu"))

    generator.add(BatchNormalization())

    generator.add(Conv2DTranspose(1, kernel_size=3,strides=1, padding="same", activation="tanh"))

    print(generator.summary())

    return generator, coding_size


generator, coding_size = create_generator()
discriminator = create_discriminator()

# create gan
Gan = Sequential([generator, discriminator], name="COMPLETE_GAN")
generator, discriminator = Gan.layers

# set discriminator trainable false
discriminator.trainable = False

# plot model
plot_model(model=Gan, show_dtype=True, show_trainable=True, show_shapes=True, show_layer_names=True, to_file="DCGAN.png")

# compile gan
Gan.compile(loss="binary_crossentropy", optimizer="adam")

# epochs
epochs = 20

for epoch in range(epochs):
    print(f"\nEpoch is {epoch+1}\n")
    # progress
    for X_batch, progress in zip(train_dataset, progress_bar(range(len(train_dataset)), desc="Progress", colour="blue")):

        # noise
        noise = tf.random.normal(shape=(batch_size, coding_size))

        # create images
        gen_images = generator(noise)

        # fake and real compare
        fake_or_real = tf.concat([gen_images, tf.dtypes.cast(X_batch, tf.float32)], axis=0)

        # create labels
        y1 = tf.constant([[0.0]]*batch_size + [[1.0]]*batch_size)

        # discriminator activate it
        discriminator.trainable = True

        # train on batch
        discriminator.train_on_batch(fake_or_real, y1)

        # generator
        # noise
        noise = tf.random.normal(shape=(batch_size, coding_size))

        # label
        y2 = tf.constant([[1.0]]*batch_size)

        discriminator.trainable = False

        # train gan
        Gan.train_on_batch(noise, y2)


generator.save("Generator.h5")






