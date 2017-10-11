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
import nltk
import re

CONVERSATION_SEP = "==="


class RedditDataCleaner:
    """
    Load the reddit comment corpus dnd clean the data briefly.
    
    Download the reddit torrent and then the .bz2 file. Run redditparser.py to generate files 
    as the input of this script.
    """
    def __init__(self, corpus_dir):
        """
        Args:
            corpus_dir: File directory where to load the corpus.
        """
        self.conversations = []

        for data_file in sorted(os.listdir(corpus_dir)):
            full_path_name = os.path.join(corpus_dir, data_file)
            if os.path.isfile(full_path_name) and data_file.lower().endswith('.txt'):
                with open(full_path_name, 'r', encoding='iso-8859-1') as f:
                    samples = []
                    for line in f:
                        l = line.strip()
                        if not l:
                            continue
                        if l == CONVERSATION_SEP:
                            if len(samples):
                                self.conversations.append(samples)
                            samples = []
                        else:
                            l = l[2:].strip()  # Remove Q: or A:
                            samples.append({"text": l})

                    if len(samples):  # Add the last one
                        self.conversations.append(samples)

    def write_cleaned_conversations(self, out_file):
        """
        Args:
            out_file: File to save the cleaned data. 
        """
        pat_curse = re.compile(
            r'\b(ass|asshole|bastard|bitch|child-fucker|damn|fuck|fucking|motherfucker|motherfucking|'
            r'nigger|shit|shitass)\b',
            re.IGNORECASE)

        # Based on the assumption that we have enough data, we want to get the best quality part only.
        # Get rid of pairs containing ", #, $, %, &, (, ), *, +, /, 0-9, <, = , >, @, [, \, ], ^, _, ` and
        # ord >= 123.
        # Note that keeping number characters will introduce further complexities when generating
        # vocabulary set. However, if you use only this data file for training, you may need to consider
        # keeping numbers.
        special_chars = [34, 35, 36, 37, 38, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                         60, 61, 62, 64, 91, 92, 93, 94, 95, 96]
        with open(out_file, 'a') as f_out:
            for conversation in self.conversations:
                written = False
                # Iterate over all the samples of the conversation to get chat pairs
                for i in range(0, len(conversation) - 1, 2):
                    input_line = conversation[i]['text'].strip()
                    target_line = conversation[i + 1]['text'].strip()

                    if all(ord(char) < 123 and ord(char) not in special_chars for char in input_line) and \
                            all(ord(char) < 123 and ord(char) not in special_chars for char in target_line):
                        input_line = self.get_formatted_line(input_line)
                        target_line = self.get_formatted_line(target_line)

                        # Only discard those conversation pairs in which the answer line contains
                        # any common cursing words
                        if re.search(pat_curse, target_line):
                            continue

                        # Discard sentences starting with a dot
                        if input_line.startswith(".") or target_line.startswith("."):
                            continue

                        # Discard sentences starting with a dash
                        if input_line.startswith("-") or target_line.startswith("-"):
                            continue

                        # This is to speed up the parsing below
                        if len(input_line) > 180 or len(target_line) > 180:
                            continue

                        in_tokens = nltk.word_tokenize(input_line)
                        tg_tokens = nltk.word_tokenize(target_line)
                        if 8 <= len(in_tokens) <= 32 and 8 <= len(tg_tokens) <= 32:
                            f_out.write("{}\n".format(input_line))
                            f_out.write("{}\n".format(target_line))
                            written = True

                if written:
                    f_out.write("===\n")

    @staticmethod
    def get_formatted_line(line):
        pat_dot = re.compile(r'\.\s+\.')
        pat_dash = re.compile(r'-\s+-')
        # pat_html = re.compile(r'<.*?>')

        # Use formal ellipsis and dashes
        while re.search(pat_dot, line):
            line = re.sub(pat_dot, '..', line)

        while re.search(pat_dash, line):
            line = re.sub(pat_dash, '--', line)

        line = re.sub('\.{3,}', '... ', line)
        line = re.sub('-{2,}', ' -- ', line)

        # Use formal apostrophe
        line = line.replace(' \' ', '\'')

        # Remove extra spaces
        line = re.sub('\s+', ' ', line).strip()
        line = line.replace(' .', '.').replace(' ?', '?').replace(' !', '!')

        # Remove HTML tags
        # line = re.sub(pat_html, '', line)

        # Remove extra punctuations and m's
        line = re.sub('\?{2,}', '?', line)
        line = re.sub('!{2,}', '!', line)
        line = re.sub('m{3,}', 'mm', line)

        return line

if __name__ == "__main__":
    from settings import PROJECT_ROOT

    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus', 'RedditData', 'standard')
    cd = RedditDataCleaner(corp_dir)
    print("{} conversations loaded.".format(len(cd.conversations)))

    out_file = os.path.join(corp_dir, 'reddit_cleaned.txt')
    cd.write_cleaned_conversations(out_file)