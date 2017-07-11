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
import numpy as np
import os
import tensorflow as tf

from chatbot.functiondata import FunctionData

class BotPredictor:
    def __init__(self, session, tokenized_data, result_dir, result_file):
        """
        Args:
            session: The TensorFlow session used to run the prediction.
            tokenized_data: The parameter data needed for prediction.
            result_dir: The full path to the folder in which the result file locates.
            result_file: The file that saves the training results.
        """
        self.tokenized_data = tokenized_data
        self.buckets = tokenized_data.buckets
        self.max_enc_len = self.buckets[-1][0]  # Last bucket has the biggest size
        self.max_dec_len = self.buckets[-1][1]

        self.num_samples = tokenized_data.num_samples
        self.vocabulary_size = tokenized_data.vocabulary_size

        print("Restoring meta graph/model architecture, please wait ...")
        saver = tf.train.import_meta_graph(os.path.join(result_dir, result_file + ".meta"))
        print("Restoring weights and other data for the model ...")
        saver.restore(session, os.path.join(result_dir, result_file))

        # Retrieve the Ops we 'remembered'.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_outputs = [[] for _ in self.buckets]

        for i in range(self.max_enc_len):
            self.encoder_inputs.append(tf.get_collection("encoder_input{0}".format(i))[0])
        for i in range(self.max_dec_len):
            self.decoder_inputs.append(tf.get_collection("decoder_input{0}".format(i))[0])
        for j, (_, dec_len) in enumerate(self.buckets):
            for i in range(dec_len):
                self.decoder_outputs[j].append(
                    tf.get_collection("decoder_output{}_{}".format(j, i))[0])

        if 0 < self.num_samples < self.vocabulary_size:
            graph = tf.get_default_graph()
            proj_w = graph.get_tensor_by_name('proj_w:0')
            w = tf.transpose(proj_w)
            b = graph.get_tensor_by_name('proj_b:0')

            for j in range(len(self.buckets)):
                self.decoder_outputs[j] = \
                    [tf.matmul(output, w) + b for output in self.decoder_outputs[j]]

        self.input_keep_prob = tf.get_collection("input_keep_prob")[0]
        self.output_keep_prob = tf.get_collection("output_keep_prob")[0]
        self.feed_previous = tf.get_collection("feed_previous")[0]

        self.session = session

    def predict(self, sentence):
        """
        Args:
            sentence: The input sentence string from the end user.
        Returns:
            out_sentence: A human readable sentence as the final output.
        """
        pat_matched, new_sentence, num_list = \
            FunctionData.check_arithmetic_pattern_and_replace(sentence)

        for pre_time in range(2):
            batch = self.tokenized_data.get_predict_batch(new_sentence)
            bucket_enc_len, bucket_dec_len = self.buckets[batch.bucket_id]

            f_dict = {}
            for i in range(bucket_enc_len):
                f_dict[self.encoder_inputs[i].name] = batch.encoder_seqs[i]

            for i in range(bucket_dec_len):
                if i == 0:
                    f_dict[self.decoder_inputs[0].name] = [self.tokenized_data.bos_token]
                else:
                    f_dict[self.decoder_inputs[i].name] = [self.tokenized_data.pad_token]

            f_dict[self.input_keep_prob] = 1.0
            f_dict[self.output_keep_prob] = 1.0
            f_dict[self.feed_previous] = True

            dec_outputs = self.session.run(self.decoder_outputs[batch.bucket_id], feed_dict=f_dict)
            # print("Shape of dec_outputs: {}".format(np.asarray(dec_outputs).shape))

            if pat_matched and pre_time == 0:
                out_sentence, if_func_val = self._get_sentence(dec_outputs, para_list=num_list)
                if if_func_val:
                    return out_sentence
                else:
                    new_sentence = sentence
            else:
                out_sentence, _ = self._get_sentence(dec_outputs)
                return out_sentence

    def _get_sentence(self, dec_outputs, para_list=None):
        """
        Args:
            dec_outputs: A tensor with the size of dec_seq_len * vocabulary_size, which is 
                the output from the predict function.
            para_list: The python list containing the parameter real values.
        Returns:
            sentence: A human readable sentence.                 
        """
        word_ids = []
        for out in dec_outputs:
            word_ids.append(np.argmax(out))

        return self.tokenized_data.word_ids_to_str(word_ids, return_if_func_val=True,
                                                   para_list=para_list)
