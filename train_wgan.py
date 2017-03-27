# -*- coding: utf-8 -*-
# file: train_wgan.py
# author: JinTian
# time: 16/03/2017 11:59 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""
train Wasserstein GAN, this is a litter complicated exactly
everything contains in this file, but this is all.
Let's do some magic!
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from models.wgan import w_gan
import cv2


tf.app.flags.DEFINE_string('tf_records_dir', './data/girls_tfrecords', 'tf_records dir')
tf.app.flags.DEFINE_string('checkpoints_dir', './checkpoints', 'your checkpoints dir')
tf.app.flags.DEFINE_string('checkpoints_prefix', 'girls', 'your checkpoints save prefix')
tf.app.flags.DEFINE_integer('save_steps', 400, 'auto save per step.')


tf.app.flags.DEFINE_integer('target_height', 64, 'target image height input into network.')
tf.app.flags.DEFINE_integer('target_width', 64, 'target image width input into network.')
tf.app.flags.DEFINE_integer('batch_size', 10, 'batch size for train.')
tf.app.flags.DEFINE_integer('noise_dim', 64*64, 'generator input noise dimension.')

tf.app.flags.DEFINE_integer('max_steps', 50000, 'max train steps.')
tf.app.flags.DEFINE_boolean('restore', False, 'to restore from checkpoints or not.')

tf.app.flags.DEFINE_string('generated_images_save_dir', './generated_images', 'path you save your generated images.')

FLAGS = tf.app.flags.FLAGS


def tf_records_walker(tf_records_dir):
    all_files = [os.path.abspath(os.path.join(tf_records_dir, i_)) for i_ in os.listdir(tf_records_dir)]
    if all_files:
        print("[INFO] %s files were found under current folder. " % len(all_files))
        print("[INFO] Please be noted that only files end with '*.tfrecord' will be load!")
        tf_records_files = [i for i in all_files if os.path.basename(i).endswith('tfrecord')]
        if tf_records_files:
            for i_ in tf_records_files:
                print('[INFO] loaded train tf_records file at: {}'.format(i_))
            return tf_records_files
        else:
            raise FileNotFoundError("Can not find any records file.")
    else:
        raise Exception("Cannot find any file under this path.")


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
        })

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.resize_image_with_crop_or_pad(
        image=image,
        target_height=FLAGS.target_height,
        target_width=FLAGS.target_width,
    )
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image


def inputs(is_train, batch_size):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            tf_records_walker(tf_records_dir=FLAGS.tf_records_dir),
            num_epochs=None,
            shuffle=True)
        image = read_and_decode(filename_queue)

        images = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=2,
            capacity=10 + 3 * batch_size,
            min_after_dequeue=10)
        return images


def generate_stochastic_noise(batch_size=FLAGS.batch_size, noise_dim=FLAGS.noise_dim):
    batch_z = np.random.normal(0, 1, [batch_size, noise_dim]).astype(np.float32)
    return batch_z


def save_generated_image(generated_imgs):
    for i in range(FLAGS.batch_size):
        img = generated_imgs[i]
        img = np.array(img*255, np.int32)
        cv2.imwrite(os.path.join(FLAGS.generated_images_save_dir, 'generated_image_{}.jpg'.format(i)), img)
        print('saved generated images into: %s' % os.path.join(FLAGS.generated_images_save_dir,
                                                               'generated_image_{}'.format(i)))


def run_training():
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    if not os.path.exists(FLAGS.generated_images_save_dir):
        os.makedirs(FLAGS.generated_images_save_dir)
    images = inputs(is_train=True, batch_size=FLAGS.batch_size)
    batch_noise = generate_stochastic_noise()

    noise = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.noise_dim))
    real_data = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.target_height, FLAGS.target_width, 3))

    # plug the placeholder into graph then feed data into placeholder
    endpoints = w_gan(real_data=real_data, noise=noise, batch_size=FLAGS.batch_size)

    saver = tf.train.Saver(max_to_keep=2)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init_op)

        # model saved generator
        step_g = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("[INFO] restore from the checkpoint {0}".format(ckpt))
                step_g += int(ckpt.split('-')[-1])
        print('[INFO] training start...')
        try:
            while not coord.should_stop():
                if step_g < 25 or step_g % 500 == 0:
                    d_iterations = 100
                else:
                    d_iterations = 5

                for d_iteration in range(d_iterations):
                    batch_noise = generate_stochastic_noise()
                    _, step_d, loss_d = sess.run([endpoints['train_op_d'], endpoints['global_step_d'],
                                                  endpoints['loss_d']],
                                                 feed_dict={real_data: images.eval(), noise: batch_noise})
                    print('[INFO] Epoch discriminator %d, discriminator loss  %.5f' % (step_d, loss_d))

                batch_noise = generate_stochastic_noise()
                _, step_g, loss_g, generated_images = sess.run([endpoints['train_op_g'], endpoints['global_step_g'],
                                                               endpoints['loss_g'], endpoints['generated_data']],
                                                               feed_dict={real_data: images.eval(), noise: batch_noise})
                print('[INFO] Epoch generator %d, generator loss  %.5f' % (step_g, loss_g))
                print(np.array(generated_images[0]*225, np.int32))

                if step_g > FLAGS.max_steps:
                    break
                if step_g % FLAGS.save_steps == 1:
                    save_generated_image(generated_images)
                    print('[INFO] Save the ckpt of {0}'.format(step_g))
                    saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.checkpoints_prefix),
                               global_step=step_g)
        except tf.errors.OutOfRangeError:
            print('[INFO] train finished.')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.checkpoints_prefix), global_step=step_g)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.checkpoints_prefix), global_step=step_g)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(step_g))
        finally:
            coord.request_stop()
            coord.join(threads)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()