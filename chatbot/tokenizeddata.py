import nltk
import numpy as np
import pickle
import string

from chatbot.rawtext import RawText


class TokenizedData:
    def __init__(self, seq_length, dict_file=None, train_file=None, save_dict=False):
        """
        One of the parameters dict_file and train_file has to be specified so that the dicts can have values.
        Args:
            seq_length: The maximum length the sequence allowed. The lengths in the encoder and decoder will both
                be derived.
            dict_file: The name of the pickle file saves the object of (word_id_dict, id_word_dict, id_cnt_dict).
            train_file: The name of the text file storing the conversations.
            save_dict: Whether to save the dicts to the file specified by the dict_file parameter.
        """
        assert dict_file is not None or train_file is not None

        self.seq_length = seq_length
        self.enc_seq_len = seq_length
        self.dec_seq_len = seq_length + 2

        # Python dicts that hold basic information
        if not save_dict and dict_file is not None:
            with open(dict_file, 'rb') as fr:
                d1, d2, d3 = pickle.load(fr)
                self.word_id_dict = d1
                self.id_word_dict = d2
                self.id_cnt_dict = d3
        else:
            self.word_id_dict = {}
            self.id_word_dict = {}
            self.id_cnt_dict = {}

        # A list in which each item contains a pair of question and its answer [[input, target]]
        self.training_samples = []

        self.vocabulary_size = 0
        self.sample_size = 0

        # Special tokens
        self.pad_token = self.get_word_id('_pad_')  # Padding
        self.bos_token = self.get_word_id('_bos_')  # Beginning of sequence
        self.eos_token = self.get_word_id('_eos_')  # End of sequence
        self.unk_token = self.get_word_id('_unk_')  # Word dropped from vocabulary

        self.per_punct = self.get_word_id('.')
        self.exc_punct = self.get_word_id('!')
        self.que_punct = self.get_word_id('?')

        if train_file is not None:
            self._load_corpus(train_file)

        if save_dict:
            dicts = (self.word_id_dict, self.id_word_dict, self.id_cnt_dict)
            with open(dict_file, 'wb') as fw:
                pickle.dump(dicts, fw, protocol=pickle.HIGHEST_PROTOCOL)

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
        training_set = np.random.permutation(self.training_samples)

        def yield_batch_samples():
            """
            Generator a batch of training samples
            """
            for i in range(0, self.sample_size, batch_size):
                yield training_set[i:min(i+batch_size, self.sample_size)]

        batches = []
        # This for loop will loop through all the sample over the whole epoch
        for samples in yield_batch_samples():
            batch = self._create_batch(samples)
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
        if sentence == '':
            return None

        tokens = nltk.word_tokenize(sentence)

        word_id_list = []
        for token in tokens:
            word_id_list.append(self.get_word_id(token, add=False))

        if len(word_id_list) > self.enc_seq_len:
            return None

        batch = self._create_batch([[word_id_list, []]])  # No target output
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

    def _create_batch(self, samples):
        """
        Create a single batch from the list of given samples.
        Args:
            samples: A list of samples, each sample being in the form [input, target]
        Return:
            batch: A batch object
        """
        batch = Batch()

        pad = self.pad_token
        bos = self.bos_token
        eos = self.eos_token

        smp_cnt = len(samples)
        for i in range(smp_cnt):
            sample = samples[i]
            assert len(sample[0]) <= self.enc_seq_len and len(sample[1]) + 2 <= self.dec_seq_len

            # Reverse input, and then left pad
            tmp_enc = list(reversed(sample[0]))
            batch.encoder_seqs.append([pad] * (self.enc_seq_len - len(tmp_enc)) + tmp_enc)

            # Add the <bos> and <eos> tokens to the output sample
            tmp_dec = [bos] + sample[1] + [eos]
            batch.decoder_seqs.append(tmp_dec + [pad] * (self.dec_seq_len - len(tmp_dec)))

            # Same as decoder, but excluding the <bos>
            tmp_tar = sample[1] + [eos]
            tmp_tar_len = len(tmp_tar)
            tar_pad_len = self.dec_seq_len - tmp_tar_len
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

    def _load_corpus(self, data_file):
        """
        Args:
            data_file: The name of the text file storing the conversations.
        """
        # Load raw text data from the corpus, identified by the given data_file
        raw_text = RawText()
        raw_text.load_corpus(data_file)

        conversations = raw_text.get_conversations()
        for conversation in conversations:
            step = 2
            # Iterate over all the samples of the conversation
            for i in range(0, len(conversation) - 1, step):
                input_line = conversation[i]
                target_line = conversation[i + 1]

                input_words = self.extract_words(input_line['text'])
                target_words = self.extract_words(target_line['text'])

                if input_words and target_words:  # Filter wrong samples (if one of the list is empty)
                    self.training_samples.append([input_words, target_words])

        self.vocabulary_size = len(self.word_id_dict)
        self.sample_size = len(self.training_samples)


class Batch:
    """
    An object holds data for a training batch
    """
    def __init__(self):
        self.encoder_seqs = []
        self.decoder_seqs = []
        self.targets = []
        self.weights = []

if __name__ == "__main__":
    import os
    from settings import PROJECT_ROOT

    dict_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'dicts.pickle')
    train_file = os.path.join(PROJECT_ROOT, 'Data', 'Corpus', 'basic_conv.txt')

    td = TokenizedData(10, dict_file=dict_file, save_dict=False)

    print('Loaded raw data: {} words, {} samples'.format(td.vocabulary_size, td.sample_size))

    for key, value in td.id_word_dict.items():
        print("key = {}, value = {}".format(key, value))

    for sample in td.training_samples:
        print(sample)
