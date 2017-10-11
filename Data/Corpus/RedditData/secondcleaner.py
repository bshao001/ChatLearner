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
"""
This optional cleaning step is to further remove those pairs that including the words in the 
excluded.txt file (created by vocabgenerator.py), which are infrequent words. This can take 
very long time (10 hours or more). If you choose to run this, you will have to generate the 
vocab file again based on the newly generated data file.
"""

COMMENT_LINE_STT = "#=="
CONVERSATION_SEP = "==="

REDDIT_INPUT = "reddit_cleaned_new.txt"
REDDIT_OUTPUT = "reddit_cleaned_new2.txt"
EXCLUDED_FILE = "excluded.txt"


def clean():
    exc_list = []

    with open(EXCLUDED_FILE, 'r') as f_exc:
        for line in f_exc:
            l = line.strip()
            if not l:
                continue
            exc_list.append(l)

    conversations = []
    with open(REDDIT_INPUT, 'r') as f_inp:
        samples = []
        for line in f_inp:
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

    with open(REDDIT_OUTPUT, 'a') as f_out:
        cnt = 0
        for conversation in conversations:
            written = False
            # Iterate over all the samples of the conversation to get chat pairs
            for i in range(0, len(conversation) - 1, 2):
                src_line = conversation[i]['text'].strip()
                tgt_line = conversation[i + 1]['text'].strip()

                assert src_line.startswith("Q:") and tgt_line.startswith("A:")

                skip = False
                tokens = (src_line[2:] + ' ' + tgt_line[2:]).split(' ')
                for token in tokens:
                    if len(token) and token != ' ':
                        t = token.lower()
                        if t in exc_list:
                            skip = True
                            cnt += 1
                            if cnt % 1000 == 0:
                                print("{:,} pairs skipped.".format(cnt))
                            break

                if not skip:
                    f_out.write("{}\n".format(src_line))
                    f_out.write("{}\n".format(tgt_line))
                    written = True

            if written:
                f_out.write("===\n")

if __name__ == "__main__":
    clean()
