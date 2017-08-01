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
import nltk
import numpy as np
import pickle
import random
import string

from chatbot.knowledgebase import KnowledgeBase
from chatbot.rawtext import RawText
from chatbot.functiondata import call_function


class TokenizedData:
    def __init__(self, dict_file, knbase_dir=None, corpus_dir=None, augment=True):
        """
        For training, both knbase_dir and corpus_dir need to be specified. For prediction, none of
        them should be given. In case only one is specified, it is ignored.
        Args:
            dict_file: The name of the pickle file saves the object used for prediction.
            knbase_dir: Name of the folder storing data files for the knowledge base.
            corpus_dir: Name of the folder storing corpus files for training.. When this is given, 
                it is for training, and the dict_file will be generated (again). Otherwise, dict_file 
                will be read only for the basic information.
            augment: Whether to apply data augmentation approach. Default to be True.
        """
        assert dict_file is not None

        # Use a number of buckets and pad the data samples to the smallest one that can accommodate.
        # For decoders, 2 slots are reserved for bos_token and eos_token. Therefore the really slots
        # available for words/punctuations are 2 less, i.e., 12, 20, 40 based on the following numbers
        # self.buckets = [(10, 14), (18, 22), (36, 42)]
        self.buckets = [(12, 18), (36, 42)]

        # Number of samples for sampled softmax. Define it here so that both the trainer and predictor
        # can easily access it.
        self.num_samples = 512

        # Python dicts that hold basic information
        if knbase_dir is None or corpus_dir is None: # If only one is given, it is ignored
            with open(dict_file, 'rb') as fr:
                dicts1, dicts2 = pickle.load(fr)
                d11, d12, d13, d14, d15 = dicts1
                d21, d22, d23 = dicts2

                self.upper_words = d11
                self.multi_words = d12
                self.multi_max_cnt = d13  # Just a number
                self.stories = d14
                self.jokes = d15  # A python list

                self.word_id_dict = d21
                self.id_word_dict = d22
                self.id_cnt_dict = d23
        else:
            self.upper_words = {}
            self.multi_words = {}
            self.multi_max_cnt = 0
            self.stories = {}
            self.jokes = []

            self.word_id_dict = {}
            self.id_word_dict = {}
            self.id_cnt_dict = {}

        # Add special tokens
        self._add_special_tokens()

        # Each item in the inner list contains a pair of question and its answer [[input, target]]
        self.training_samples = [[] for _ in self.buckets]
        self.sample_size = []
        for _ in self.buckets:
            self.sample_size.append(0)

        if knbase_dir is not None and corpus_dir is not None:
            self._load_knbase_and_corpus(knbase_dir, corpus_dir, augment=augment)

            dicts1 = (self.upper_words, self.multi_words, self.multi_max_cnt, self.stories, self.jokes)
            dicts2 = (self.word_id_dict, self.id_word_dict, self.id_cnt_dict)
            dicts = (dicts1, dicts2)
            with open(dict_file, 'wb') as fw:
                pickle.dump(dicts, fw, protocol=pickle.HIGHEST_PROTOCOL)

        self.vocabulary_size = len(self.word_id_dict)

    def get_word_id(self, word, keep_case=False, add=True):
        """
        Get the id of the word (and add it to the dictionary if not existing). If the word does not
        exist and add is False, the function will return the unk_token value.
        Args:
            word: Word to add.
            keep_case: Whether to keep the original case. If False, will use its lower case 
                counterpart.
            add: If True and the word does not exist already, the world will be added.
        Return:
            word_id: The ID of the word created.
        """
        if not keep_case:
            word = word.lower()  # Ignore case

        # At inference, we simply look up for the word
        if not add:
            word_id = self.word_id_dict.get(word, self.unk_token)
        # Get the id if the word already exist
        elif word in self.word_id_dict:
            word_id = self.word_id_dict[word]
            self.id_cnt_dict[word_id] += 1
        # If not, we create a new entry
        else:
            word_id = len(self.word_id_dict)
            self.word_id_dict[word] = word_id
            self.id_word_dict[word_id] = word
            self.id_cnt_dict[word_id] = 1

        return word_id

    def get_training_batches(self, batch_size):
        """
        Prepare all the batches for the current epoch
        Args:
            batch_size: The batch_size for min-batch training.
        Return:
            batches: A list of the batches for the coming epoch of training.
        """
        def yield_batch_samples():
            """
            Generator a batch of training samples
            """
            rand_list = np.random.permutation([x for x in range(len(self.buckets))])

            for bucket_id in rand_list:
                # print("bucket_id = {}".format(bucket_id))

                samp_size = self.sample_size[bucket_id]
                if samp_size == 0: continue
                training_set = np.random.permutation(self.training_samples[bucket_id])

                for i in range(0, samp_size, batch_size):
                    yield bucket_id, training_set[i:min(i+batch_size, samp_size)]

        batches = []
        # This for loop will loop through all the sample over the whole epoch
        for bucket_id, samples in yield_batch_samples():
            batch = self._create_batch(samples, bucket_id)
            batches.append(batch)

        return batches

    def get_predict_batch(self, sentence, use_bucket='Last'):
        """
        Encode a sequence and return a batch as an input for prediction.
        Args:
            sentence: A sentence in plain text to encode.
            use_bucket: Options are, This, Next, and Last. Default to Last. Indicates which bucket
                to use by given sentence. 
        Return:
            batch: A batch object based on the giving sentence, or None if something goes wrong
        """
        word_id_list = []

        if sentence == '':
            unk_cnt = random.randint(1, 4)
            for i in range(unk_cnt):  # Should respond based on the corresponding training sample
                word_id_list.append(self.unk_token)
        else:
            tokens = nltk.word_tokenize(sentence)

            for m in range(self.multi_max_cnt, 1, -1):
                # Slide the sentence with stride 1, window size m, and look for match
                for n in range(0, len(tokens) - m + 1, 1):
                    tmp = ' '.join(tokens[n:n+m]).strip().lower()
                    if tmp in self.multi_words:
                        # Capitalized format is stored
                        tmp_id = self.get_word_id(self.multi_words[tmp], keep_case=True,
                                                  add=False)
                        tmp_token = "_tk_" + str(tmp_id) + "_kt_"
                        # Replace with the temp token
                        del tokens[n:n + m]
                        tokens.insert(n, tmp_token)

            for token in tokens:
                if token.startswith('_tk_') and token.endswith('_kt_'):
                    word_id_list.append(int(token.replace('_tk_', '').replace('_kt_', '')))
                else:
                    word_id_list.append(self.get_word_id(token, add=False))

            if len(word_id_list) > self.buckets[-1][0]:
                word_id_list = []
                unk_cnt = random.randint(1, 6)
                for i in range(unk_cnt):  # Should respond based on the corresponding training sample
                    word_id_list.append(self.unk_token)

        bucket_id = -1
        if use_bucket == 'Last':
            bucket_id = len(self.buckets) - 1
        else:
            for bkt_id, (src_size, _) in enumerate(self.buckets):
                if len(word_id_list) <= src_size:
                    bucket_id = bkt_id
                    if use_bucket == 'Next' and bkt_id < len(self.buckets) - 1:
                        bucket_id += 1
                    break

        batch = self._create_batch([[word_id_list, []]], bucket_id)  # No target output
        return batch

    def word_ids_to_str(self, word_id_list, debug=False, return_if_func_val=False,
                        para_list=None, html_format=False):
        """
        Convert a list of integers (word_ids) into a human readable string/text.
        Used for prediction only, when debug is False.
        Args:
            word_id_list (list<int>): A list of word_ids.
            debug: Output the text including special tokens.
            return_if_func_val: Whether to include a boolean value, in the returning set, which 
                indicates if the output sentence containing a _func_val_** string.
            para_list: The python list containing the parameter real values.
            html_format: Whether out_sentence is in HTML format.
        Return:
            str: The sentence represented by the given word_id_list.
        """
        if not word_id_list:
            return ''

        sentence = []
        if_func_val = False
        if debug:
            for word_id in word_id_list:
                word = ' ' + self.id_word_dict[word_id]
                sentence.append(word)
        else:
            last_id = 0
            for word_id in word_id_list:
                if word_id == self.eos_token:  # End of sentence
                    break
                elif word_id > 3:  # Not reserved special tokens
                    word = self.id_word_dict[word_id]
                    if word.startswith('_func_val_'):
                        if_func_val = True
                        word = call_function(word[10:], tokenized_data=self, para_list=para_list,
                                             html_format=html_format)
                    else:
                        if word in self.upper_words:
                            word = self.upper_words[word]

                        if (last_id == 0 or last_id in self.cap_punc_list) \
                                and not word[0].isupper():
                            word = word.capitalize()

                    if not word.startswith('\'') and word != 'n\'t' \
                            and (word not in string.punctuation or word_id in self.con_punc_list) \
                            and last_id not in self.con_punc_list:
                        word = ' ' + word
                    sentence.append(word)

                    last_id = word_id

        if return_if_func_val:
            return ''.join(sentence).strip(), if_func_val
        else:
            return ''.join(sentence).strip()

    def _create_batch(self, samples, bucket_id):
        """
        Create a single batch from the list of given samples.
        Args:
            samples: A list of samples, each sample being in the form [input, target]
            bucket_id: The bucket ID of the given buckets defined in the object initialization.
        Return:
            batch: A batch object
        """
        enc_seq_len, dec_seq_len = self.buckets[bucket_id]

        batch = Batch()
        batch.bucket_id = bucket_id

        pad = self.pad_token
        bos = self.bos_token
        eos = self.eos_token

        smp_cnt = len(samples)
        for i in range(smp_cnt):
            sample = samples[i]

            # Reverse input, and then left pad
            tmp_enc = list(reversed(sample[0]))
            batch.encoder_seqs.append([pad] * (enc_seq_len - len(tmp_enc)) + tmp_enc)

            # Add the <bos> and <eos> tokens to the output sample
            tmp_dec = [bos] + sample[1] + [eos]
            batch.decoder_seqs.append(tmp_dec + [pad] * (dec_seq_len - len(tmp_dec)))

            # Same as decoder, but excluding the <bos>
            tmp_tar = sample[1] + [eos]
            tmp_tar_len = len(tmp_tar)
            tar_pad_len = dec_seq_len - tmp_tar_len
            batch.targets.append(tmp_tar + [pad] * tar_pad_len)

            # Weight the real targets with 1.0, while 0.0 for pads
            batch.weights.append([1.0] * tmp_tar_len + [0.0] * tar_pad_len)

        # Reorganize the data in the batch so that correspoding data items in different samples
        # of this batch are stacked together
        batch.encoder_seqs = [[*x] for x in zip(*batch.encoder_seqs)]
        batch.decoder_seqs = [[*x] for x in zip(*batch.decoder_seqs)]
        batch.targets = [[*x] for x in zip(*batch.targets)]
        batch.weights = [[*x] for x in zip(*batch.weights)]

        return batch

    def _add_special_tokens(self):
        # Special tokens.
        self.pad_token = self.get_word_id('_pad_')  # 0. Padding
        self.bos_token = self.get_word_id('_bos_')  # 1. Beginning of sequence
        self.eos_token = self.get_word_id('_eos_')  # 2. End of sequence
        self.unk_token = self.get_word_id('_unk_')  # 3. Word dropped from vocabulary

        # The word following this punctuation should be capitalized
        self.cap_punc_list = []
        for p in ['.', '!', '?']:
            self.cap_punc_list.append(self.get_word_id(p))

        # The word following this punctuation should not precede with a space.
        self.con_punc_list = []
        for p in ['(', '[', '{', '``', '$']:
            self.con_punc_list.append(self.get_word_id(p))

    def _extract_words(self, text_line):
        """
        Extract the words/terms from a sample line and represent them with corresponding word/term IDs
        Args:
            text_line: A line of the text to extract.
        Return:
            sentences: The list of sentences, each of which are words/terms (represented in corresponding 
                IDs) in the sentence.
        """
        sentences = []  # List[str]

        # Extract sentences
        token_sentences = nltk.sent_tokenize(text_line)

        # Add sentence by sentence until it reaches the maximum length
        for i in range(len(token_sentences)):
            tokens = nltk.word_tokenize(token_sentences[i])

            for m in range(self.multi_max_cnt, 1, -1):
                # Slide the sentence with stride 1, window size m, and look for match
                for n in range(0, len(tokens) - m + 1, 1):
                    tmp = ' '.join(tokens[n:n+m]).strip().lower()
                    if tmp in self.multi_words:
                        # Capitalized format is stored
                        tmp_id = self.get_word_id(self.multi_words[tmp], keep_case=True)
                        tmp_token = "_tk_" + str(tmp_id) + "_kt_"
                        # Replace with the temp token
                        del tokens[n:n + m]
                        tokens.insert(n, tmp_token)

            word_ids = []
            for token in tokens:
                if token.startswith('_tk_') and token.endswith('_kt_'):
                    word_ids.append(int(token.replace('_tk_', '').replace('_kt_', '')))
                else:
                    word_ids.append(self.get_word_id(token))

            sentences.extend(word_ids)

        return sentences

    def _load_knbase_and_corpus(self, knbase_dir, corpus_dir, augment):
        """
        Args:
            knbase_dir: Name of the folder storing data files for the knowledge base.
            corpus_dir: Name of the folder storing corpus files for training.
            augment: Whether to apply data augmentation approach.
        """
        # Load knowledge base
        knbs = KnowledgeBase()
        knbs.load_knbase(knbase_dir)

        self.upper_words = knbs.upper_words
        self.multi_words = knbs.multi_words
        self.multi_max_cnt = knbs.multi_max_cnt
        self.stories = knbs.stories
        self.jokes = knbs.jokes

        # Load raw text data from the corpus
        raw_text = RawText()
        raw_text.load_corpus(corpus_dir)

        for cid in range(3):
            if cid == 0:
                conversations = raw_text.conversations_aug0
            elif cid == 1:
                conversations = raw_text.conversations_aug1
            else:
                conversations = raw_text.conversations_aug2

            for conversation in conversations:
                step = 2
                # Iterate over all the samples of the conversation to get chat pairs
                for i in range(0, len(conversation) - 1, step):
                    input_line = conversation[i]
                    target_line = conversation[i + 1]

                    src_word_ids = self._extract_words(input_line['text'])
                    tgt_word_ids = self._extract_words(target_line['text'])

                    for bucket_id, (src_size, tgt_size) in enumerate(self.buckets):
                        if len(src_word_ids) < src_size and len(tgt_word_ids) <= tgt_size - 2:
                            self.training_samples[bucket_id].append([src_word_ids, tgt_word_ids])
                            if cid >= 1 and augment:
                                aug_len = src_size - len(src_word_ids)
                                aug_src_ids = [self.pad_token] * aug_len + src_word_ids[:]
                                self.training_samples[bucket_id].append([aug_src_ids, tgt_word_ids])
                            break
                    else:
                        print("Input ({}) or target ({}) line is too long to fit into any bucket."
                              .format(input_line, target_line))

                    if cid == 2 and augment:
                        aug_src_ids = [self.pad_token] * 2 + src_word_ids[:]
                        for bucket_id, (src_size, tgt_size) in enumerate(self.buckets):
                            if len(aug_src_ids) <= src_size and len(tgt_word_ids) <= tgt_size - 2:
                                self.training_samples[bucket_id].append([aug_src_ids, tgt_word_ids])
                                break
                        else:
                            print("Augmented Input ({}) is too long to fit into any bucket."
                                  .format(input_line))

        for bucket_id, _ in enumerate(self.buckets):
            self.sample_size[bucket_id] = len(self.training_samples[bucket_id])


class Batch:
    """
    An object holds data for a training batch
    """
    def __init__(self):
        self.bucket_id = -1
        self.encoder_seqs = []
        self.decoder_seqs = []
        self.targets = []
        self.weights = []

if __name__ == "__main__":
    import os
    from settings import PROJECT_ROOT

    dict_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'dicts.pickle')
    knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')

    td = TokenizedData(dict_file=dict_file, knbase_dir=knbs_dir, corpus_dir=corp_dir,
                       augment=False)
    print('Loaded raw data: {} words, {} samples'.format(td.vocabulary_size, td.sample_size))

    for key, value in td.id_word_dict.items():
        print("key = {}, value = {}".format(key, value))

    for bucket_id, _ in enumerate(td.buckets):
        print("Bucket {}".format(bucket_id))
        for sample in td.training_samples[bucket_id]:
            print("[{}], [{}]".format(td.word_ids_to_str(sample[0], debug=True),
                                      td.word_ids_to_str(sample[1])))