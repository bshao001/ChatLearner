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
import os

COMMENT_LINE_STT = "#=="
CONVERSATION_SEP = "==="


def corpus_pre_process(file_dir):
    """
    Pre-process the training data so that it is ready to be handled by TensorFlow TextLineDataSet
    """
    for data_file in sorted(os.listdir(file_dir)):
        full_path_name = os.path.join(file_dir, data_file)
        if os.path.isfile(full_path_name) and data_file.lower().endswith('.txt'):
            new_name = data_file.lower().replace('.txt', '_new.txt')
            full_new_name = os.path.join(file_dir, new_name)

            conversations = []
            with open(full_path_name, 'r') as f:
                samples = []
                for line in f:
                    l = line.strip()
                    if not l or l.startswith(COMMENT_LINE_STT):
                        continue
                    if l == CONVERSATION_SEP:
                        if len(samples):
                            conversations.append(samples)
                        samples = []
                    else:
                        samples.append({"text": l})

                if len(samples):  # Add the last one
                    conversations.append(samples)

            with open(full_new_name, 'a') as f_out:
                i = 0
                for conversation in conversations:
                    i += 1
                    step = 2
                    # Iterate over all the samples of the conversation to get chat pairs
                    for i in range(0, len(conversation) - 1, step):
                        source_tokens = nltk.word_tokenize(conversation[i]['text'])
                        target_tokens = nltk.word_tokenize(conversation[i + 1]['text'])

                        source_line = "Q: " + ' '.join(source_tokens[:]).strip()
                        target_line = "A: " + ' '.join(target_tokens[:]).strip()

                        f_out.write("{}\n".format(source_line))
                        f_out.write("{}\n".format(target_line))

                    f_out.write("===\n")

if __name__ == "__main__":
    from settings import PROJECT_ROOT

    file_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus', 'temp')
    corpus_pre_process(file_dir)