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

AUG0_FOLDER = "Augment0"
AUG1_FOLDER = "Augment1"
AUG2_FOLDER = "Augment2"

VOCAB_FILE = "vocab.txt"


def generate_vocab_file(corpus_dir):
    """
    Generate the vocab.txt file for the training and prediction/inference. 
    Manually remove the empty bottom line in the generated file.
    """
    vocab_list = []

    # Special tokens.
    for t in ['_pad_', '_bos_', '_eos_', '_unk_']:
        vocab_list.append(t)

    # The word following this punctuation should be capitalized in the prediction output.
    for t in ['.', '!', '?']:
        vocab_list.append(t)

    # The word following this punctuation should not precede with a space in the prediction output.
    for t in ['(', '[', '{', '``', '$']:
        vocab_list.append(t)

    for fd in range(2, -1, -1):
        if fd == 0:
            file_dir = os.path.join(corpus_dir, AUG0_FOLDER)
        elif fd == 1:
            file_dir = os.path.join(corpus_dir, AUG1_FOLDER)
        else:
            file_dir = os.path.join(corpus_dir, AUG2_FOLDER)

        for data_file in sorted(os.listdir(file_dir)):
            full_path_name = os.path.join(file_dir, data_file)
            if os.path.isfile(full_path_name) and data_file.lower().endswith('.txt'):
                with open(full_path_name, 'r') as f:
                    for line in f:
                        l = line.strip()
                        if not l:
                            continue
                        if l.startswith("Q:") or l.startswith("A:"):
                            tokens = l[2:].strip().split(' ')
                            for token in tokens:
                                if len(token) and token != ' ':
                                    t = token.lower()
                                    if t not in vocab_list:
                                        vocab_list.append(t)

    vocab_file = os.path.join(corp_dir, VOCAB_FILE)
    with open(vocab_file, 'a') as f_voc:
        for v in vocab_list:
            f_voc.write("{}\n".format(v))

if __name__ == "__main__":
    from settings import PROJECT_ROOT

    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
    generate_vocab_file(corp_dir)