from __future__ import absolute_import, division, print_function, unicode_literals


import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from forecaster.apps.video_forecast import data

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(data.HEIGHT, data.WIDTH, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim)],
            name='inference_net')

        self.generative_net = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=16 * 16 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(16, 16, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME")],
            name='generative_net')

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, inputs, output_variation=False, **kwargs):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        outputs = tf.nn.sigmoid(x_logit)
        if output_variation:
            return outputs, (x_logit, mean, logvar, z)
        return outputs


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def elmo_loss(self, x, predictions):
    x_logit, mean, logvar, z = predictions
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = self.log_normal_pdf(z, 0., 0.)
    logqz_x = self.log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_and_save_images(model, test_input):
    predictions = model.sample(test_input)
    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.show()


def train():
    epochs = 20
    latent_dim = 64
    num_examples_to_generate = 16

    train_dataset, test_dataset = data.make_cvae_datasets()
    model = CVAE(latent_dim)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    # generate_and_save_images(model, 0, random_vector_for_generation)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            compute_apply_gradients(model, train_x, optimizer)
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, '
                  'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))
    generate_and_save_images(model, random_vector_for_generation)


def test():
    import matplotlib.pyplot as plt

    train_dataset, test_dataset = data.make_cvae_datasets()
    cvae_model = CVAE(latent_dim=64)
    cvae_model.load_weights('ckpts/99.ckpt')
    for x in test_dataset.take(1):
        y = tf.nn.sigmoid(cvae_model(x)[0])

        plt.imshow(x[0, :, :, 0].numpy() * 255)
        plt.imshow(y[0, :, :, 0].numpy() * 255)
    plt.show()


def load_cvae_model(path):
    cvae_model = CVAE(latent_dim=64)
    cvae_model.load_weights(path)
    return cvae_model


if __name__ == '__main__':
    train()
