import os
import tensorflow as tf

from tensorflow.python.keras.losses import LossFunctionWrapper
from forecaster.apps.video_forecast import cvae
from forecaster.apps.video_forecast import data
from forecaster.apps import base
from forecaster.data import datasets
from forecaster.models import layers
from forecaster.models import losses
from forecaster.models import metrics
from forecaster.models import networks
from forecaster.models import optimizers


def binary_crossentropy(y_true, y_pred):  # pylint: disable=missing-docstring
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.where(y_true > 0.5, tf.ones_like(y_true, y_true.dtype), tf.zeros_like(y_true, y_true.dtype))

    return tf.keras.backend.mean(
        tf.keras.backend.binary_crossentropy(y_true, y_pred), axis=-1)


class BinaryCrossEntropy(LossFunctionWrapper):
    def __init__(self, name=None):
        super(BinaryCrossEntropy, self).__init__(
            binary_crossentropy,
            name=name)


class VideoForecast(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 conv_kernel_size,
                 image_encoder,
                 image_generator,
                 input_shape=None,
                 name=None):
        super(VideoForecast, self).__init__(name=name or 'video_forecast')
        self._inputs = tf.keras.layers.InputLayer(input_shape=input_shape)
        self._image_encoder = image_encoder
        self._d_model = d_model
        self._extractor = networks.SequenceEncoder(
            num_layers=num_layers, d_model=d_model,
            num_attention_heads=num_attention_heads, conv_kernel_size=conv_kernel_size,
            numeric_normalizer_fn=None, name='sequence_encoder')
        self._image_generator = image_generator

    def call(self, inputs, **kwargs):
        # inputs: (N, T, H, W, C)
        shape = tf.shape(inputs, name='shape')
        input_images = tf.reshape(inputs, (-1, shape[2], shape[3], shape[4]), name='input_video2image')
        encoded_images = self._image_encoder(input_images)[..., :self._d_model]  # (NT, D)
        encoded_video = tf.reshape(encoded_images, (shape[0], shape[1], self._d_model), name='encoded_image2video')
        extracted = self._extractor(encoded_video, mask=None)  # （N, T, D)
        extracted_images = tf.reshape(extracted, (-1, self._d_model), name='extracted_video2image')  # (NT, D)
        output_images = self._image_generator(extracted_images)  # (NT, H, W, C)
        outputs = tf.reshape(output_images, shape, name='generated_image2video')  # (N, T, H, W, C)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def main():
    train_dataset, eval_dataset, test_dataset = data.make_datasets()

    cvae_model = cvae.load_cvae_model(  # 'forecaster/apps/video_forecast/'
                                      'cvae_ckpt/99.ckpt')
    image_encoder = cvae_model.inference_net
    image_generator = cvae_model.generative_net
    model = VideoForecast(3, 64, 8, 5, image_encoder, image_generator, input_shape=(20, 64, 64))
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['mse'])
    # a = tf.random.normal((3, 20, 64, 64, 1))
    # b = model(a)
    # tf.print(tf.shape(b))

    callbacks = [
        tf.keras.callbacks.TensorBoard('/Users/Tianziyang/projects/MovingMnist/tensorboard'),
        tf.keras.callbacks.ModelCheckpoint('ckpt', save_best_only=True)
    ]
    model.fit(train_dataset, epochs=100, validation_data=eval_dataset,
              steps_per_epoch=data.TRAIN_STEPS_PER_EPOCH,
              validation_steps=data.VALIDATION_STEPS,
              callbacks=callbacks)


def test():
    train_dataset, eval_dataset, test_dataset = data.make_datasets()

    cvae_model = cvae.load_cvae_model(  # 'forecaster/apps/video_forecast/'
        'cvae_ckpt/99.ckpt')
    image_encoder = cvae_model.inference_net
    image_generator = cvae_model.generative_net
    model = VideoForecast(3, 64, 8, 5, image_encoder, image_generator, input_shape=(20, 64, 64))
    # model.load_weights(...)

    for x, y in test_dataset.take(1):
        y_pred = model(x)
        xy = tf.concat([x, y_pred], axis=1) * 255

    import matplotlib.pyplot as plt

    for j in range(20):
        plt.ioff()
        for i in range(4):
            ax = plt.subplot(2, 2, i + 1)
            video = xy.numpy()[i][:, :, :, 0]
            # print(video.shape)
            ax.imshow(video[j])
            plt.pause(0.01)

    plt.show()


if __name__ == '__main__':
    test()
