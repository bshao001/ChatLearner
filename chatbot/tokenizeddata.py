import nltk
import numpy as np
import pickle
import random
import string

from chatbot.rawtext import RawText


class TokenizedData:
    def __init__(self, dict_file, corpus_dir=None):
        """
        Args:
            dict_file: The name of the pickle file saves the object of (word_id_dict, id_word_dict, 
                id_cnt_dict).
            corpus_dir: Name of the folder storing corpus files for training.. When this is given, 
                it is for training, and the dict_file will be generated (again). Otherwise, dict_file 
                will be read only for the basic information.
        """
        assert dict_file is not None

        # Use a number of buckets and pad the data samples to the smallest one that can accommodate.
        # For decoders, 2 slots are reserved for bos_token and eos_token. Therefore the really slots
        # available for words/punctuations are 2 less, i.e., 12, 20, 40 based on the following numbers
        self.buckets = [(10, 14), (18, 22), (36, 42)]

        # Number of samples for sampled softmax. Define it here so that both the trainer and predictor
        # can easily access it.
        self.num_samples = 500

        # Python dicts that hold basic information
        if corpus_dir is None:
            with open(dict_file, 'rb') as fr:
                d1, d2, d3 = pickle.load(fr)
                self.word_id_dict = d1
                self.id_word_dict = d2
                self.id_cnt_dict = d3
        else:
            self.word_id_dict = {}
            self.id_word_dict = {}
            self.id_cnt_dict = {}

        # Special tokens
        self.pad_token = self.get_word_id('_pad_')  # Padding
        self.bos_token = self.get_word_id('_bos_')  # Beginning of sequence
        self.eos_token = self.get_word_id('_eos_')  # End of sequence
        self.unk_token = self.get_word_id('_unk_')  # Word dropped from vocabulary

        self.per_punct = self.get_word_id('.')
        self.exc_punct = self.get_word_id('!')
        self.que_punct = self.get_word_id('?')

        # Each item in the inner list contains a pair of question and its answer [[input, target]]
        self.training_samples = [[] for _ in self.buckets]
        self.sample_size = []
        for _ in self.buckets:
            self.sample_size.append(0)

        if corpus_dir is not None:
            self._load_corpus(corpus_dir)

            dicts = (self.word_id_dict, self.id_word_dict, self.id_cnt_dict)
            with open(dict_file, 'wb') as fw:
                pickle.dump(dicts, fw, protocol=pickle.HIGHEST_PROTOCOL)

        self.vocabulary_size = len(self.word_id_dict)

    def extract_words(self, text_line):
        """
        Extract the words from a sample line and represent them with corresponding word IDs
        Args:
            text_line: A line of the text to extract.
        Return:
            sentences: The list of sentences, each of which are words (represented in corresponding IDs)
                in the sentence.
        """
        sentences = []  # List[str]

        # Extract sentences
        token_sentences = nltk.sent_tokenize(text_line)

        # Add sentence by sentence until it reaches the maximum length
        for i in range(len(token_sentences)):
            tokens = nltk.word_tokenize(token_sentences[i])

            word_ids = []
            for token in tokens:
                word_ids.append(self.get_word_id(token))

            sentences.extend(word_ids)

        return sentences

    def get_word_id(self, word, add=True):
        """
        Get the id of the word (and add it to the dictionary if not existing). If the word does not
        exist and add is False, the function will return the unk_token value.
        Args:
            word: Word to add.
            add: If True and the word does not exist already, the world will be added.
        Return:
            word_id: The ID of the word created.
        """
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

    def get_predict_batch(self, sentence):
        """
        Encode a sequence and return a batch as an input for prediction.
        Args:
            sentence: A sentence in plain text to encode.
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

            for token in tokens:
                word_id_list.append(self.get_word_id(token, add=False))

            if len(word_id_list) > self.buckets[-1][0]:
                word_id_list = []
                unk_cnt = random.randint(1, 6)
                for i in range(unk_cnt):  # Should respond based on the corresponding training sample
                    word_id_list.append(self.unk_token)

        bucket_id = -1
        for bkt_id, (src_size, _) in enumerate(self.buckets):
            if len(word_id_list) <= src_size:
                bucket_id = bkt_id
                break

        batch = self._create_batch([[word_id_list, []]], bucket_id)  # No target output
        return batch

    def word_ids_to_str(self, word_id_list):
        """
        Convert a list of integers (word_ids) into a human readable string/text.
        Used for prediction only.
        Args:
            word_id_list (list<int>): A list of word_ids.
        Return:
            str: The sentence represented by the given word_id_list.
        """
        if not word_id_list:
            return ''

        last_id = 0
        sentence = []
        for word_id in word_id_list:
            if word_id == self.eos_token:  # End of sentence
                break
            elif word_id > 3:  # Not special tokens
                word = self.id_word_dict[word_id]
                if not word.startswith('\'') and word not in string.punctuation:
                    if last_id == 0 or last_id == self.per_punct \
                            or last_id == self.exc_punct or last_id == self.que_punct:
                        word = word.capitalize()
                    word = ' ' + word
                sentence.append(word)

                last_id = word_id

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

    def _load_corpus(self, corpus_dir):
        """
        Args:
            corpus_dir: Name of the folder storing corpus files for training.
        """
        # Load raw text data from the corpus, identified by the given data_file
        raw_text = RawText()
        raw_text.load_corpus(corpus_dir)

        conversations = raw_text.get_conversations()
        for conversation in conversations:
            step = 2
            # Iterate over all the samples of the conversation to get chat pairs
            for i in range(0, len(conversation) - 1, step):
                input_line = conversation[i]
                target_line = conversation[i + 1]

                src_word_ids = self.extract_words(input_line['text'])
                tgt_word_ids = self.extract_words(target_line['text'])

                bucket_found = False
                for bucket_id, (src_size, tgt_size) in enumerate(self.buckets):
                    if len(src_word_ids) <= src_size and len(tgt_word_ids) <= tgt_size - 2:
                        self.training_samples[bucket_id].append([src_word_ids, tgt_word_ids])
                        bucket_found = True
                        break
                if not bucket_found:
                    print("Input ({}) or target ({}) line is too long to fit into any bucket"
                          .format(input_line, target_line))

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
    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')

    td = TokenizedData(dict_file=dict_file, corpus_dir=corp_dir)
    print('Loaded raw data: {} words, {} samples'.format(td.vocabulary_size, td.sample_size))

    for key, value in td.id_word_dict.items():
        print("key = {}, value = {}".format(key, value))

    for bucket_id, _ in enumerate(td.buckets):
        print("Bucket {}".format(bucket_id))
        for sample in td.training_samples[bucket_id]:
            print("[{}], [{}]".format(td.word_ids_to_str(sample[0]), td.word_ids_to_str(sample[1])))

    # all_batches = td.get_training_batches(4)
    # for b in all_batches:
    #     print("ENC = {}".format(b.encoder_seqs))
    #     print("DEC = {}".format(b.decoder_seqs))
