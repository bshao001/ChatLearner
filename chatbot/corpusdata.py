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
import os
import tensorflow as tf

from collections import namedtuple
from tensorflow.python.ops import lookup_ops

COMMENT_LINE_STT = "#=="
CONVERSATION_SEP = "==="

AUG0_FOLDER = "Augment0"
AUG1_FOLDER = "Augment1"
AUG2_FOLDER = "Augment2"

MAX_LEN = 1000  # Assume no line in the training data is having more than this number of characters
VOCAB_FILE = "vocab.txt"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class CorpusData:
    def __init__(self, corpus_dir, augment_factor=3, src_max_len=50, tgt_max_len=50,
                 source_reverse=True, buffer_size=8192):
        """
        Args:
            corpus_dir: Name of the folder storing corpus files for training.
            augment_factor: Times the training data appears. If 1 or less, no augmentation.
            src_max_len: The max length of the encoder input.
            tgt_max_len: The max length of the decoder input.
            source_reverse: Whether to reverse the order of the input sequence.
            buffer_size: The buffer size used for mapping process during data processing.
        """
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        self.text_set = None  # For debugging purpose
        self.id_set = None

        self.case_table = prepare_case_table()

        vocab_file = os.path.join(corpus_dir, VOCAB_FILE)
        self.vocab_table = lookup_ops.index_table_from_file(vocab_file, default_value=UNK_ID)

        self._load_corpus(corpus_dir, augment_factor)
        self._convert_to_tokens(source_reverse, buffer_size)

    def get_training_iterator(self, batch_size=8, num_buckets=5, num_threads=2):
        buffer_size = batch_size * 1000

        self.id_set = self.id_set.shuffle(buffer_size=buffer_size)

        # Create a target input prefixed with BOS and a target output suffixed with EOS.
        # After this mapping, each element in the id_set contains 3 columns/items.
        self.id_set = self.id_set.map(lambda src, tgt:
                                      (src, tf.concat(([BOS_ID], tgt), 0), tf.concat((tgt, [EOS_ID]), 0)),
                                      num_threads=num_threads,
                                      output_buffer_size=buffer_size)

        # Add in sequence lengths.
        self.id_set = self.id_set.map(lambda src, tgt_in, tgt_out:
                                      (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
                                      num_threads=num_threads,
                                      output_buffer_size=buffer_size)

        def batching_func(x):
            return x.padded_batch(
                batch_size,
                # The first three entries are the source and target line rows, these have unknown-length
                # vectors. The last two entries are the source and target row sizes, which are scalars.
                padded_shapes=(tf.TensorShape([None]),  # src
                               tf.TensorShape([None]),  # tgt_input
                               tf.TensorShape([None]),  # tgt_output
                               tf.TensorShape([]),      # src_len
                               tf.TensorShape([])),     # tgt_len
                # Pad the source and target sequences with eos tokens. Though we don't generally need to
                # do this since later on we will be masking out calculations past the true sequence.
                padding_values=(EOS_ID,  # src
                                EOS_ID,  # tgt_input
                                EOS_ID,  # tgt_output
                                0,       # src_len -- unused
                                0))      # tgt_len -- unused

        if num_buckets > 1:
            bucket_width = (self.src_max_len + num_buckets - 1) // num_buckets

            # Parameters match the columns in each element of the dataset.
            def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
                # Calculate bucket_width by maximum source sequence length. Pairs with length [0, bucket_width)
                # go to bucket 0, length [bucket_width, 2 * bucket_width) go to bucket 1, etc. Pairs with
                # length over ((num_bucket-1) * bucket_width) words all go into the last bucket.
                # Bucket sentence pairs by the length of their source sentence and target sentence.
                bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
                return tf.to_int64(tf.minimum(num_buckets, bucket_id))

            # No key to filter the dataset. Therefore it is unused.
            def reduce_func(unused_key, windowed_data):
                return batching_func(windowed_data)

            batched_dataset = self.id_set.group_by_window(key_func=key_func,
                                                          reduce_func=reduce_func,
                                                          window_size=batch_size)
        else:
            batched_dataset = batching_func(self.id_set)

        batched_iter = batched_dataset.make_initializable_iterator()
        return batched_iter

        # (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (batched_iter.get_next())
        #
        # return BatchedInput(initializer=batched_iter.initializer,
        #                     source=src_ids,
        #                     target_input=tgt_input_ids,
        #                     target_output=tgt_output_ids,
        #                     source_sequence_length=src_seq_len,
        #                     target_sequence_length=tgt_seq_len)

    def _load_corpus(self, corpus_dir, augment_factor):
        for fd in range(2, -1, -1):
            file_list = []
            if fd == 0:
                file_dir = os.path.join(corpus_dir, AUG0_FOLDER)
            elif fd == 1:
                file_dir = os.path.join(corpus_dir, AUG1_FOLDER)
            else:
                file_dir = os.path.join(corpus_dir, AUG2_FOLDER)

            for data_file in sorted(os.listdir(file_dir)):
                full_path_name = os.path.join(file_dir, data_file)
                if os.path.isfile(full_path_name) and data_file.lower().endswith('.txt'):
                    file_list.append(full_path_name)

            assert len(file_list) > 0
            dataset = tf.contrib.data.TextLineDataset(file_list)

            src_dataset = dataset.filter(lambda line:
                                         tf.logical_and(tf.size(line) > 0,
                                                        tf.equal(tf.substr(line, 0, 2), tf.constant('Q:'))))
            src_dataset = src_dataset.map(lambda line:
                                          tf.substr(line, 2, MAX_LEN),
                                          output_buffer_size=4096)
            tgt_dataset = dataset.filter(lambda line:
                                         tf.logical_and(tf.size(line) > 0,
                                                        tf.equal(tf.substr(line, 0, 2), tf.constant('A:'))))
            tgt_dataset = tgt_dataset.map(lambda line:
                                          tf.substr(line, 2, MAX_LEN),
                                          output_buffer_size=4096)

            src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))
            if augment_factor > 1 and fd >= 1:
                src_tgt_dataset = src_tgt_dataset.repeat(augment_factor * fd)

            if self.text_set is None:
                self.text_set = src_tgt_dataset
            else:
                self.text_set = self.text_set.concatenate(src_tgt_dataset)

    def _convert_to_tokens(self, source_reverse, buffer_size):
        # The following 3 steps act as a python String lower() function
        # Split to characters
        self.text_set = self.text_set.map(lambda src, tgt:
                                          (tf.string_split([src], delimiter='').values,
                                           tf.string_split([tgt], delimiter='').values),
                                          output_buffer_size=buffer_size)
        # Convert all upper case characters to lower case characters
        self.text_set = self.text_set.map(lambda src, tgt:
                                        (self.case_table.lookup(src), self.case_table.lookup(tgt)),
                                        output_buffer_size=buffer_size)
        # Join characters back to strings
        self.text_set = self.text_set.map(lambda src, tgt:
                                        (tf.reduce_join([src]), tf.reduce_join([tgt])),
                                        output_buffer_size=buffer_size)

        # Split to word tokens
        self.text_set = self.text_set.map(lambda src, tgt:
                                          (tf.string_split([src]).values, tf.string_split([tgt]).values),
                                          output_buffer_size=buffer_size)
        # Remove sentences longer than the model allows
        self.text_set = self.text_set.map(lambda src, tgt:
                                          (src[:self.src_max_len], tgt[:self.tgt_max_len]),
                                          output_buffer_size=buffer_size)

        # Reverse the source sentence if applicable
        if source_reverse:
            self.text_set = self.text_set.map(lambda src, tgt:
                                              (tf.reverse(src, axis=[0]), tgt),
                                              output_buffer_size=buffer_size)

        # Convert the word strings to ids.  Word strings that are not in the vocab get
        # the lookup table's default_value integer.
        self.id_set = self.text_set.map(lambda src, tgt:
                                        (tf.cast(self.vocab_table.lookup(src), tf.int32),
                                         tf.cast(self.vocab_table.lookup(tgt), tf.int32)),
                                        output_buffer_size=buffer_size)


def prepare_case_table():
    keys = tf.constant([chr(i) for i in range(32, 127)])

    l1 = [chr(i) for i in range(32, 65)]
    l2 = [chr(i) for i in range(97, 123)]
    l3 = [chr(i) for i in range(91, 127)]
    values = tf.constant(l1 + l2 + l3)

    return tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values), ' ')


class BatchedInput(namedtuple("BatchedInput",
                              ["initializer",
                               "source",
                               "target_input",
                               "target_output",
                               "source_sequence_length",
                               "target_sequence_length"])):
    pass

if __name__ == "__main__":
    from settings import PROJECT_ROOT

    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
    cdata = CorpusData(corp_dir)

    iterator = cdata.get_training_iterator()
    next_batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        print("Initialized ... ...")

        for i in range(2):
            try:
                element = sess.run(next_batch)
                print(i, element)
            except tf.errors.OutOfRangeError:
                print("end of data @ {}".format(i))
                break