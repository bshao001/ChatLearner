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
import math
import numpy as np
import os
import tensorflow as tf

from time import time
from chatbot.tfcopy.seq2seq import embedding_attention_seq2seq


class BasicModel:
    def __init__(self, tokenized_data, num_layers, num_units, input_keep_prob=0.8,
                 output_keep_prob=0.8, embedding_size=32, batch_size=32):
        """
        A basic Neural Conversational Model to predict the next sentence given an input 
        sentence. It is a simplified implementation of the seq2seq model as described: 
        https://arxiv.org/abs/1506.05869.
        Args:
            tokenized_data: An object of TokenizedData that holds the data prepared for 
                training. Corpus data should have been loaded (as implemented) before 
                pass here as a parameter.
            num_layers: The number of layers of RNN model used in both encoder and decoder.
            num_units: The number of units in each of the RNN layer.
            input_keep_prob: The input_keep_prob float number.
            output_keep_prob: The output_keep_prob float number.
            embedding_size: Integer, the length of the embedding vector for each word.
            batch_size: The number of samples to be used in one step of the optimization 
                process.
        """
        self.tokenized_data = tokenized_data
        self.buckets = tokenized_data.buckets
        self.max_enc_len = self.buckets[-1][0]  # Last bucket has the biggest size
        self.max_dec_len = self.buckets[-1][1]

        self.num_samples = tokenized_data.num_samples
        self.vocabulary_size = tokenized_data.vocabulary_size

        self.num_layers = num_layers
        self.num_units = num_units
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob

        self.embedding_size = embedding_size
        self.batch_size = batch_size

    def train(self, num_epochs, train_dir, result_file):
        """
        Launch the training process and save the training data.
        Args:
            num_epochs: The number of epochs for the training.
            train_dir: The full path to the folder in which the result_file locates.
            result_file: The file name to save the train result.
        """
        def_graph = tf.Graph()
        with def_graph.as_default():
            output_projection, softmax_loss_func = self._define_out_projection()

            encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{0}'.format(i))
                              for i in range(self.max_enc_len)]
            decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{0}'.format(i))
                              for i in range(self.max_dec_len)]

            in_keep_prob = tf.placeholder(tf.float32, name='input_keep_prob')
            out_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')
            feed_prev = tf.placeholder(tf.bool, shape=[], name='feed_previous')

            print("Building inference graph ...")
            t0 = time()
            outputs = self._build_inference_graph(encoder_inputs, decoder_inputs, in_keep_prob,
                                                  out_keep_prob, output_projection, feed_prev)
            t1 = time()
            print("Inference graph created. Time spent: {:6.2f} seconds".format(t1 - t0))

            for i in range(self.max_enc_len):
                tf.add_to_collection("encoder_input{0}".format(i), encoder_inputs[i])
            for i in range(self.max_dec_len):
                tf.add_to_collection("decoder_input{0}".format(i), decoder_inputs[i])
            for j, (_, dec_len) in enumerate(self.buckets):
                for i in range(dec_len):
                    tf.add_to_collection("decoder_output{}_{}".format(j, i), outputs[j][i])

            tf.add_to_collection("input_keep_prob", in_keep_prob)
            tf.add_to_collection("output_keep_prob", out_keep_prob)
            tf.add_to_collection("feed_previous", feed_prev)

            # We save only the inference graph for prediction purpose. In case you need to save
            # the training graph, move this line underneath the line creating the training graph.
            saver = tf.train.Saver()

            targets = [tf.placeholder(tf.int32, [None], name='targets{0}'.format(i))
                       for i in range(self.max_dec_len)]
            weights = [tf.placeholder(tf.float32, [None], name='weights{0}'.format(i))
                       for i in range(self.max_dec_len)]
            learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

            print("Building training graph ...")
            t2 = time()
            train_ops, losses = self._build_training_graph(outputs, targets, weights,
                                                           softmax_loss_func, learning_rate)
            t3 = time()
            print("Training graph created. Time spent: {:6.2f} seconds".format(t3 - t2))

            # Place the saver creation here instead if you want to save the training graph as well.
            # saver = tf.train.Saver()

        with tf.Session(graph=def_graph) as sess:
            print("Initializing variables ...")
            sess.run(tf.global_variables_initializer())

            print("Variables initialized")
            save_file = os.path.join(train_dir, result_file)

            loss_list = []
            last_perp = 200.0
            write_meta_graph = True
            for epoch in range(num_epochs):
                batches = self.tokenized_data.get_training_batches(self.batch_size)

                lr_feed = BasicModel._get_learning_rate(last_perp)
                for b in batches:
                    bucket_enc_len, bucket_dec_len = self.buckets[b.bucket_id]

                    f_dict = {}

                    for i in range(bucket_enc_len):
                        f_dict[encoder_inputs[i].name] = b.encoder_seqs[i]

                    for i in range(bucket_dec_len):
                        f_dict[decoder_inputs[i].name] = b.decoder_seqs[i]
                        f_dict[targets[i].name] = b.targets[i]
                        f_dict[weights[i].name] = b.weights[i]

                    f_dict[in_keep_prob] = self.input_keep_prob
                    f_dict[out_keep_prob] = self.output_keep_prob
                    f_dict[feed_prev] = False
                    f_dict[learning_rate] = lr_feed

                    _, loss_val = sess.run([train_ops[b.bucket_id], losses[b.bucket_id]],
                                           feed_dict=f_dict)
                    loss_list.append(loss_val)

                # Output training status
                if epoch % 8 == 0 or epoch == num_epochs - 1:
                    mean_loss = sum(loss_list) / len(loss_list)
                    perplexity = np.exp(float(mean_loss)) if mean_loss < 300 else math.inf
                    print("At epoch {}: learning_rate = {:.6f}, mean loss = {:.4f}, perplexity = {:.4f}".
                          format(epoch, lr_feed, mean_loss, perplexity))
                    if epoch == num_epochs - 1:
                        saver.save(sess, save_file)  # Write meta graph at the last save
                    elif perplexity < 1.08:
                        if perplexity < 1.02:  # Write meta graph before break
                            write_meta_graph = True
                        saver.save(sess, save_file, global_step=epoch, write_meta_graph=write_meta_graph)
                        write_meta_graph = False

                    if perplexity < 1.02:
                        break

                    loss_list = []
                    last_perp = perplexity

    def _define_out_projection(self):
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_func = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.num_samples < self.vocabulary_size:
            w_t = tf.get_variable("proj_w", [self.vocabulary_size, self.num_units],
                                  dtype=tf.float32)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.vocabulary_size], dtype=tf.float32)
            output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(tf.nn.sampled_softmax_loss(
                    weights=local_w_t,
                    biases=local_b,
                    labels=labels,
                    inputs=local_inputs,
                    num_sampled=self.num_samples,
                    num_classes=self.vocabulary_size), tf.float32)

            softmax_loss_func = sampled_loss

        return output_projection, softmax_loss_func

    def _build_inference_graph(self, encoder_inputs, decoder_inputs, input_keep_prob,
                               output_keep_prob, output_projection, feed_previous):
        """
        Create the inference graph for training or prediction.
        Args:
            encoder_inputs: The placeholder for encoder_inputs.
            decoder_inputs: The placeholder for decoder_inputs.
            input_keep_prob: The placeholder for input_keep_prob.
            output_keep_prob: The placeholder for output_keep_prob.
            output_projection: The weights and biases for output_projection.
            feed_previous: The placeholder for feed_previous.
        Returns:
            outputs: A list of decoder_outputs from embedding_rnn_seq2seq function.
        """
        def create_rnn_layer(num_units):
            layer = tf.contrib.rnn.GRUCell(num_units)
            return tf.contrib.rnn.DropoutWrapper(layer,
                                                 input_keep_prob=input_keep_prob,
                                                 output_keep_prob=output_keep_prob,
                                                 state_keep_prob=1.0)

        enc_net = tf.contrib.rnn.MultiRNNCell([create_rnn_layer(self.num_units)
                                               for _ in range(self.num_layers)])
        dec_net = tf.contrib.rnn.MultiRNNCell([create_rnn_layer(self.num_units)
                                               for _ in range(self.num_layers)])

        def embed_seq2seq(enc_inputs, dec_inputs):
            return embedding_attention_seq2seq(
                enc_inputs, dec_inputs, enc_net, dec_net, self.vocabulary_size,
                self.vocabulary_size, self.embedding_size, num_heads=1,
                output_projection=output_projection,
                feed_previous=feed_previous, dtype=tf.float32)

        outputs = []
        for j, bucket in enumerate(self.buckets):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                bucket_outputs, _ = embed_seq2seq(encoder_inputs[:bucket[0]],
                                                  decoder_inputs[:bucket[1]])
                outputs.append(bucket_outputs)

        return outputs

    def _build_training_graph(self, outputs, targets, weights, softmax_loss_function,
                              learning_rate):
        """
        Create the training graph for the training.
        Args:
            outputs: A list of the decoder_outputs from the model.
            targets: The placeholder for targets.
            weights: The placeholder for weights.
            softmax_loss_function: The softmax_loss_function.
            learning_rate: The placeholder for learning_rate.
        Returns:
            train_op: The Op for training.
            loss: The Op for calculating loss.
        """
        losses = []
        train_ops = []
        for j, bucket in enumerate(self.buckets):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                loss = tf.contrib.legacy_seq2seq.sequence_loss(
                    logits=outputs[j], targets=targets[:bucket[1]], weights=weights[:bucket[1]],
                    average_across_batch=False, softmax_loss_function=softmax_loss_function)
                losses.append(loss)

                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
                train_ops.append(train_op)

        return train_ops, losses

    @staticmethod
    def _get_learning_rate(perplexity):
        if perplexity <= 1.1:
            return 9.2e-5
        elif perplexity <= 1.2:
            return 9.6e-5
        elif perplexity <= 1.4:
            return 1e-4
        elif perplexity <= 2.0:
            return 1.2e-4
        elif perplexity <= 3.0:
            return 1.6e-4
        elif perplexity <= 4.0:
            return 2e-4
        elif perplexity <= 6.0:
            return 2.4e-4
        elif perplexity <= 12.0:
            return 3.2e-4
        elif perplexity <= 30.0:
            return 4e-4
        else:
            return 8e-4

if __name__ == "__main__":
    from settings import PROJECT_ROOT
    from chatbot.tokenizeddata import TokenizedData

    dict_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'dicts.pickle')
    knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
    corpus_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')

    print("Loading training data ...")
    td = TokenizedData(dict_file=dict_file, knbase_dir=knbs_dir, corpus_dir=corpus_dir)
    print('Loaded raw data: {} words, {} samples'.format(td.vocabulary_size, td.sample_size))

    model = BasicModel(tokenized_data=td, num_layers=2, num_units=800, input_keep_prob=0.9,
                       output_keep_prob=0.9, embedding_size=64, batch_size=32)

    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    model.train(num_epochs=120, train_dir=res_dir, result_file='basic')
