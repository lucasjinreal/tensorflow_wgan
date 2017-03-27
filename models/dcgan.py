# -*- coding: utf-8 -*-
# file: dcgan.py
# author: JinTian
# time: 16/03/2017 12:00 PM
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
import tensorflow.contrib.slim as slim
import tensorflow as tf


def leaky_relu(x, leak=0.3):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def discriminator_convolution(inputs, batch_size, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        net = slim.conv2d(inputs, 64, [3, 3], padding='SAME', stride=2, activation_fn=leaky_relu)
        net = slim.batch_norm(net)

        net = slim.conv2d(net, 128, [3, 3], padding='SAME', stride=2, activation_fn=leaky_relu)
        net = slim.batch_norm(net)

        net = slim.conv2d(net, 256, [3, 3], padding='SAME', stride=2, activation_fn=leaky_relu)
        net = slim.batch_norm(net)

        net = slim.conv2d(net, 512, [3, 3], padding='SAME', stride=2, activation_fn=leaky_relu)
        net = slim.batch_norm(net)

        # in discriminator, he's duty is identify 2 classes, true or false, so output is 0 or 1
        net = slim.fully_connected(tf.reshape(net, [batch_size, -1]), num_outputs=1, activation_fn=None)
    return net


def generator_convolution(inputs):

    net = slim.fully_connected(inputs, num_outputs=512*7*7, activation_fn=tf.nn.relu)
    net = slim.batch_norm(net)

    net = tf.reshape(net, (-1, 7, 7, 512))

    net = slim.conv2d_transpose(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu)
    net = slim.batch_norm(net)

    net = slim.conv2d_transpose(net, 128, [3, 3], stride=2, activation_fn=tf.nn.relu)
    net = slim.batch_norm(net)

    net = slim.conv2d_transpose(net, 64, [3, 3], stride=2, activation_fn=tf.nn.relu)
    net = slim.batch_norm(net)

    net = slim.conv2d_transpose(net, 3, [3, 3], stride=1, activation_fn=tf.nn.relu)
    net = slim.batch_norm(net)
    print('generator out net shape: ', net.get_shape())
    # (10, 56, 56, 3), for rgb image should be 3

    return net


def generator_mlp(inputs):
    net = slim.fully_connected(inputs, num_outputs=4*4*512, activation_fn=tf.nn.relu)
    net = slim.batch_norm(net)

    net = slim.fully_connected(net, 64, activation_fn=tf.nn.relu)
    net = slim.batch_norm(net)

    net = slim.fully_connected(net, 64*64*3, activation_fn=tf.nn.relu)
    net = slim.batch_norm(net)

    # return generated image with shape (64, 64, 3)
    net = tf.reshape(net, (64, 64, 3))
    return net

