import math
import numpy as np
import os
import tensorflow as tf


class BasicModel:
    def __init__(self, tokenized_data, num_layers, num_units, embedding_size=32, batch_size=8):
        """
        A basic Neural Conversational Model to predict the next sentence given an input sentence. It is
        a simplified implementation of the seq2seq model as described: https://arxiv.org/abs/1506.05869
        Args:
            tokenized_data: An object of TokenizedData that holds the data prepared for training. Corpus
                data should have been loaded before pass here as a parameter.
            num_layers: The number of layers of RNN model used in both encoder and decoder.
            num_units: The number of units in each of the RNN layer.
            embedding_size: Integer, the length of the embedding vector for each word.
            batch_size: The number of samples to be used in one step of the optimization process.
        """
        self.tokenized_data = tokenized_data
        self.enc_seq_len = tokenized_data.enc_seq_len
        self.dec_seq_len = tokenized_data.dec_seq_len
        self.vocabulary_size = tokenized_data.vocabulary_size

        self.num_layers = num_layers
        self.num_units = num_units

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
            encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{0}'.format(i))
                              for i in range(self.enc_seq_len)]
            decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{0}'.format(i))
                              for i in range(self.dec_seq_len)]
            feed_previous = tf.placeholder(tf.bool, shape=[], name='feed_previous')

            decoder_outputs, states = self._build_inference_graph(encoder_inputs, decoder_inputs,
                                                                  feed_previous)

            for i in range(self.enc_seq_len):
                tf.add_to_collection("encoder_input{0}".format(i), encoder_inputs[i])
            for i in range(self.dec_seq_len):
                tf.add_to_collection("decoder_input{0}".format(i), decoder_inputs[i])
                tf.add_to_collection("decoder_output{0}".format(i), decoder_outputs[i])

            tf.add_to_collection("feed_previous", feed_previous)

            targets = [tf.placeholder(tf.int32, [None], name='targets{0}'.format(i))
                       for i in range(self.dec_seq_len)]
            weights = [tf.placeholder(tf.float32, [None], name='weights{0}'.format(i))
                       for i in range(self.dec_seq_len)]
            learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

            train_op, loss = self._build_training_graph(decoder_outputs, targets, weights,
                                                        learning_rate)

            saver = tf.train.Saver()

        with tf.Session(graph=def_graph) as sess:
            sess.run(tf.global_variables_initializer())

            save_file = os.path.join(train_dir, result_file)

            loss_list = []
            last_perp = 200.0
            for epoch in range(num_epochs):
                batches = self.tokenized_data.get_training_batches(self.batch_size)

                lr_feed = self._get_learning_rate(last_perp)
                for b in batches:
                    f_dict = {}

                    for i in range(self.enc_seq_len):
                        f_dict[encoder_inputs[i].name] = b.encoder_seqs[i]

                    for i in range(self.dec_seq_len):
                        f_dict[decoder_inputs[i].name] = b.decoder_seqs[i]
                        f_dict[targets[i].name] = b.targets[i]
                        f_dict[weights[i].name] = b.weights[i]

                    f_dict[feed_previous] = False
                    f_dict[learning_rate] = lr_feed

                    _, loss_val = sess.run([train_op, loss], feed_dict=f_dict)

                # Output training status
                loss_list.append(loss_val)
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    mean_loss = sum(loss_list) / len(loss_list)
                    perplexity = np.exp(float(mean_loss)) if mean_loss < 300 else math.inf
                    print("At epoch {}: learning_rate = {}, mean loss = {:.2f}, perplexity = {:.2f}".
                          format(epoch, lr_feed, mean_loss, perplexity))

                    loss_list = []
                    last_perp = perplexity

            saver.save(sess, save_file)

    def _build_inference_graph(self, encoder_inputs, decoder_inputs, feed_previous):
        """
        Create the inference graph for training or prediction.
        Args:
            encoder_inputs: The placeholder for encoder_inputs.
            decoder_inputs: The placeholder for decoder_inputs.
            feed_previous: The placeholder for feed_previous.
        Returns:
            decoder_outputs, states: Refer to embedding_rnn_seq2seq function for details.
        """
        def create_rnn_layer(num_units):
            return tf.contrib.rnn.LSTMCell(num_units, use_peepholes=True)

        rnn_net = tf.contrib.rnn.MultiRNNCell([create_rnn_layer(self.num_units)
                                               for _ in range(self.num_layers)])

        decoder_outputs, states = \
            tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                encoder_inputs, decoder_inputs, rnn_net, self.vocabulary_size, self.vocabulary_size,
                self.embedding_size, output_projection=None, feed_previous=feed_previous,
                dtype=tf.float32)

        return decoder_outputs, states

    def _build_training_graph(self, decoder_outputs, targets, weights, learning_rate):
        """
        Create the training graph for the training.
        Args:
            decoder_outputs: The decoder output from the model.
            targets: The placeholder for targets.
            weights: The placeholder for weights.
            learning_rate: The placeholder for learning_rate.
        Returns:
            train_op: The Op for training.
            loss: The Op for calculating loss.
        """
        loss = tf.contrib.legacy_seq2seq.sequence_loss(
            logits=decoder_outputs, targets=targets, weights=weights,
            average_across_batch=True, softmax_loss_function=None)

        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return train_op, loss

    def _get_learning_rate(self, perplexity):
        if perplexity <= 1.6:
            return 9.2e-5
        elif perplexity <= 2.0:
            return 9.6e-5
        elif perplexity <= 2.4:
            return 1e-4
        elif perplexity <= 4.0:
            return 1.2e-4
        elif perplexity <= 8.0:
            return 1.6e-4
        elif perplexity <= 16.0:
            return 2e-4
        elif perplexity <= 24.0:
            return 2.4e-4
        elif perplexity <= 32.0:
            return 3.2e-4
        elif perplexity <= 40.0:
            return 4e-4
        else:
            return 8e-4

if __name__ == "__main__":
    from settings import PROJECT_ROOT
    from chatbot.tokenizeddata import TokenizedData

    data_file = os.path.join(PROJECT_ROOT, 'Data', 'Corpus', 'basic_conv.txt')
    td = TokenizedData(seq_length=10, data_file=data_file)

    model = BasicModel(tokenized_data=td, num_layers=2, num_units=128, embedding_size=32,
                       batch_size=8)

    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    model.train(num_epochs=500, train_dir=res_dir, result_file='basic')
