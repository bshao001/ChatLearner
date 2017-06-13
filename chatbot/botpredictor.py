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

        saver = tf.train.import_meta_graph(os.path.join(result_dir, result_file + ".meta"))
        saver.restore(session, os.path.join(result_dir, result_file))

        # Retrieve the Ops we 'remembered'.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_outputs = []

        for i in range(tokenized_data.enc_seq_len):
            self.encoder_inputs.append(tf.get_collection("encoder_input{0}".format(i))[0])
        for i in range(tokenized_data.dec_seq_len):
            self.decoder_inputs.append(tf.get_collection("decoder_input{0}".format(i))[0])
            self.decoder_outputs.append(tf.get_collection("decoder_output{0}".format(i))[0])

        self.feed_previous = tf.get_collection("feed_previous")[0]

        self.session = session

    def predict(self, sentence):
        batch = self.tokenized_data.get_predict_batch(sentence)

        f_dict = {}
        for i in range(self.tokenized_data.enc_seq_len):
            f_dict[self.encoder_inputs[i].name] = batch.encoder_seqs[i]

        for i in range(self.tokenized_data.dec_seq_len):
            if i == 0:
                f_dict[self.decoder_inputs[0].name] = [self.tokenized_data.bos_token]
            else:
                f_dict[self.decoder_inputs[i].name] = [self.tokenized_data.pad_token]

        f_dict[self.feed_previous] = True

        dec_outputs = self.session.run(self.decoder_outputs, feed_dict=f_dict)

        return dec_outputs
