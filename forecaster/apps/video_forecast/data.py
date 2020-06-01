import os
import numpy as np
import tensorflow as tf


DATA_FILE = 'data/mnist_test_seq.npy'
TRAIN_EXAMPLES = 8000
VALIDATION_EXAMPLES = 1000
TEST_EXAMPLES = 1000
TIME_STEPS = 20
INPUT_STEPS = 10
TARGET_STEPS = TIME_STEPS - INPUT_STEPS
HEIGHT = 64
WIDTH = 64

DATA_DIR = 'data'
TRAIN_DIR = 'data/train'
EVAL_DIR = 'data/eval'
TEST_DIR = 'data/test'

BATCH_SIZE = 16
CVAE_BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 8000
TRAIN_STEPS_PER_EPOCH = TRAIN_EXAMPLES // BATCH_SIZE
VALIDATION_STEPS = VALIDATION_EXAMPLES // BATCH_SIZE


def export_tfrecords():
    data = np.load(DATA_FILE).astype(np.int)
    data = np.transpose(data, (1, 0, 2, 3))
    np.random.shuffle(data)

    for i in range(10000 // 100):
        mini_data = data[100 * i: 100 * (i + 1)]
        with tf.io.TFRecordWriter('data/MovingMnist_%02d.tfrecords' % i) as writer:
            for d in mini_data:
                video_flat = np.reshape(d, (-1,))
                feature = {'video': tf.train.Feature(int64_list=tf.train.Int64List(value=video_flat))}
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized = example_proto.SerializeToString()
                writer.write(serialized)
        print('%i files completed.' % i)


def parse_serialized(serialized):
    feature_description = {'video': tf.io.FixedLenFeature([TIME_STEPS * HEIGHT * WIDTH], tf.int64)}
    video_flat = tf.io.parse_single_example(serialized, feature_description)
    video = tf.reshape(video_flat['video'], (TIME_STEPS, HEIGHT, WIDTH))
    return video


def scale_video(video):
    video = tf.cast(video, tf.float32) / 255.
    video = video[..., tf.newaxis]
    return video


def to_image(video):
    return tf.reshape(video, (-1, HEIGHT, WIDTH, 1))


def split_xy(video):
    return video[:INPUT_STEPS], video[INPUT_STEPS:]


@tf.function
def parse_tfrecord_and_convert(serialized):
    video = parse_serialized(serialized)
    scaled_video = scale_video(video)
    video_x, video_y = split_xy(scaled_video)
    return video_x, video_y


@tf.function
def parse_tfrecord_and_convert_for_cvae(serialized):
    video = parse_serialized(serialized)
    scaled_video = scale_video(video)
    images = to_image(scaled_video)
    return images


def make_dataset(tfrecords_dir):
    data_files = tf.io.gfile.glob(os.path.join(tfrecords_dir, '*.tfrecords'))
    dataset = tf.data.TFRecordDataset(data_files)
    dataset = dataset.map(parse_tfrecord_and_convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def make_datasets():
    train_dataset = make_dataset(TRAIN_DIR).repeat().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
    eval_dataset = make_dataset(EVAL_DIR).repeat().batch(BATCH_SIZE)
    test_dataset = make_dataset(TEST_DIR).repeat().batch(BATCH_SIZE)
    return train_dataset, eval_dataset, test_dataset


def make_cvae_datasets():
    train_files = tf.io.gfile.glob(os.path.join(TRAIN_DIR, '*.tfrecords'))
    train_files += tf.io.gfile.glob(os.path.join(EVAL_DIR, '*.tfrecords'))
    train_dataset = tf.data.TFRecordDataset(train_files).map(
        parse_tfrecord_and_convert_for_cvae, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    train_dataset = train_dataset.repeat().shuffle(SHUFFLE_BUFFER_SIZE).batch(CVAE_BATCH_SIZE)

    test_files = tf.io.gfile.glob(os.path.join(TEST_DIR, '*.tfrecords'))
    test_dataset = tf.data.TFRecordDataset(test_files).map(
        parse_tfrecord_and_convert_for_cvae, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    test_dataset = test_dataset.repeat().batch(CVAE_BATCH_SIZE)
    return train_dataset, test_dataset


if __name__ == '__main__':
    export_tfrecords()
