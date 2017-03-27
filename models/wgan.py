# -*- coding: utf-8 -*-
# file: wgan.py
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
"""
heart of Wasserstein GAN, I am trying to make this simple and clean
it's really not complicated, everything seems like a water stream
the data flows in the graph, and you can feel it.
"""
import tensorflow as tf
from models.dcgan import discriminator_convolution, generator_convolution


def w_gan(real_data, noise, batch_size, lr_g=5e-5, lr_d=5e-5, is_train=True):

    endpoints = {}

    # first generator use noise generate an fake image
    generated_data = generator_convolution(noise)

    if is_train:
        # then discriminator will get true logits and fake logits
        true_logits = discriminator_convolution(real_data, batch_size=batch_size)
        fake_logits = discriminator_convolution(generated_data, batch_size=batch_size, reuse=True)

        # discriminator loss
        loss_d = tf.reduce_mean(true_logits - fake_logits)
        loss_g = tf.reduce_mean(-fake_logits)

        global_step_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        train_op_g = tf.train.RMSPropOptimizer(learning_rate=lr_g).minimize(loss_g, global_step=global_step_g)

        global_step_d = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        train_op_d = tf.train.RMSPropOptimizer(learning_rate=lr_d).minimize(loss_d, global_step=global_step_d)

        endpoints['train_op_g'] = train_op_g
        endpoints['train_op_d'] = train_op_d
        endpoints['global_step_g'] = global_step_g
        endpoints['global_step_d'] = global_step_d
        endpoints['loss_g'] = loss_g
        endpoints['loss_d'] = loss_d
        endpoints['generated_data'] = generated_data
    else:
        endpoints['generated_data'] = generated_data

    return endpoints




