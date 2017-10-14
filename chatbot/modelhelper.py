# Copyright 2017 Bo Shao. All Rights Reserved.
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
# ==============================================================================
import tensorflow as tf


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.contrib.keras.initializers.glorot_normal(seed=seed)
    elif init_op == "glorot_uniform":
        return tf.contrib.keras.initializers.glorot_uniform(seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


# def get_device_str(device_id, num_gpus):
#     """Return a device string for multi-GPU setup."""
#     if num_gpus == 0:
#         return "/cpu:0"
#     device_str_output = "/gpu:%d" % (device_id % num_gpus)
#     return device_str_output


def create_embbeding(vocab_size, embed_size, dtype=tf.float32, scope=None):
    """Create embedding matrix for both encoder and decoder."""
    with tf.variable_scope(scope or "embeddings", dtype=dtype):
        embedding = tf.get_variable("embedding", [vocab_size, embed_size], dtype)

    return embedding


def _single_cell(num_units, keep_prob, device_str=None):
    """Create an instance of a single RNN cell."""
    single_cell = tf.contrib.rnn.GRUCell(num_units)

    if keep_prob < 1.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=keep_prob)

    # Device Wrapper
    if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)

    return single_cell


def create_rnn_cell(num_units, num_layers, keep_prob):
    """Create multi-layer RNN cell."""
    cell_list = []
    for i in range(num_layers):
        single_cell = _single_cell(num_units=num_units, keep_prob=keep_prob)
        cell_list.append(single_cell)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)


def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
        tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary
