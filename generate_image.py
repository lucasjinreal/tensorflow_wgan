# -*- coding: utf-8 -*-
# file: generate_image.py
# author: JinTian
# time: 17/03/2017 10:16 AM
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
import tensorflow as tf
import os
import numpy as np
from models.wgan import w_gan
import cv2

tf.app.flags.DEFINE_integer('noise_dim', 64 * 64, 'generator input noise dimension.')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size for train.')
tf.app.flags.DEFINE_string('checkpoints_dir', './checkpoints', 'your checkpoints dir')


FLAGS = tf.app.flags.FLAGS


def generate_image():
    noise = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.noise_dim))
    batch_noise = np.random.normal(0, 1, [FLAGS.batch_size, FLAGS.noise_dim]).astype(np.float32)
    endpoints = w_gan(real_data=None, noise=noise, batch_size=FLAGS.batch_size)

    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        if ckpt:
            saver.restore(sess, ckpt)

        generated_images = sess.run([endpoints['generated_data']], feed_dict={noise: batch_noise})
        image = generated_images[0]
        cv2.imshow('FUCK!!!', image)
        cv2.waitKey(0)


def main(_):
    generate_image()


if __name__ == '__main__':
    tf.app.run()
