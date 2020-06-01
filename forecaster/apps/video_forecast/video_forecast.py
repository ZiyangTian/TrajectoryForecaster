import os
import tensorflow as tf

from forecaster.apps.video_forecast import cvae
from forecaster.apps.video_forecast import data
from forecaster.apps import base
from forecaster.data import sequence
from forecaster.models import layers
from forecaster.models import losses
from forecaster.models import metrics
from forecaster.models import networks
from forecaster.models import optimizers


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
        encoded_video = tf.reshape(encoded_images, (shape[0], shape[1], -1), name='encoded_image2video')  # (N, T, D)
        extracted = self._extractor(encoded_video, mask=None)  # （N, T, D)
        extracted_images = tf.reshape(extracted, (-1, self._d_model), name='extracted_video2image')  # (NT, D)
        output_images = self._image_generator(extracted_images)  # (NT, H, W, C)
        outputs = tf.reshape(output_images, shape, name='generated_image2video')  # (N, T, H, W, C)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def main():
    train_dataset, eval_dataset, test_dataset = data.make_datasets()

    cvae_model = cvae.load_cvae_model(# 'forecaster/apps/video_forecast/'
                                      'cvae_ckpt/99.ckpt')
    image_encoder = cvae_model.inference_net
    image_generator = cvae_model.generative_net
    model = VideoForecast(3, 64, 8, 5, image_encoder, image_generator, input_shape=(20, 64, 64))
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['mse'])
    a = tf.random.normal((3, 20, 64, 64, 1))
    b = model(a)
    tf.print(tf.shape(b))

    exit()
    callbacks = [
        tf.keras.callbacks.TensorBoard('tensorboard'),
        tf.keras.callbacks.ModelCheckpoint('ckpt', save_best_only=True)
    ]
    model.fit(train_dataset, epochs=100, validation_data=eval_dataset,
              steps_per_epoch=data.TRAIN_STEPS_PER_EPOCH,
              validation_steps=data.VALIDATION_STEPS,
              callbacks=callbacks)


if __name__ == '__main__':
    main()
