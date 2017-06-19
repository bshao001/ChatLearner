import numpy as np
import os
import tensorflow as tf


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

        print("Restoring meta graph/model architecture, please wait ...")
        saver = tf.train.import_meta_graph(os.path.join(result_dir, result_file + ".meta"))
        print("Restoring weights and other data for the model ...")
        saver.restore(session, os.path.join(result_dir, result_file))

        # Retrieve the Ops we 'remembered'.
        # print("Restoring saved variables from the collections ...")
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

        self.input_keep_prob = tf.get_collection("input_keep_prob")[0]
        self.output_keep_prob = tf.get_collection("output_keep_prob")[0]
        self.feed_previous = tf.get_collection("feed_previous")[0]

        self.session = session

    def predict(self, sentence):
        """
        Args:
            sentence: The input sentence string from the end user.
        Returns:
            dec_outputs: A tensor with the size of dec_seq_len * vocabulary_size.
        """
        batch = self.tokenized_data.get_predict_batch(sentence)
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
        return dec_outputs

    def get_sentence(self, dec_outputs):
        """
        Args:
            dec_outputs: A tensor with the size of dec_seq_len * vocabulary_size, which is 
                the output from the predict function.
        Returns:
            sentence: A human readable sentence.                 
        """
        word_ids = []
        for out in dec_outputs:
            word_ids.append(np.argmax(out))

        return self.tokenized_data.word_ids_to_str(word_ids)