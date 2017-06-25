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
Load data from a dataset of simply-formatted data

from A to B
from B to A
from A to B
from B to A
===
from C to D
from D to C
from C to D
from D to C
from C to D
from D to C
...

`===` lines just separate linear conversations between 2 people.
"""
import os

CONVERSATION_SEP = "==="


class RawText:
    def __init__(self):
        self.conversations = []

    def load_corpus(self, corpus_dir):
        """
        Args:
             corpus_dir: Name of the folder storing corpus files for training.
        """
        for data_file in sorted(os.listdir(corpus_dir)):
            full_path_name = os.path.join(corpus_dir, data_file)
            if os.path.isfile(full_path_name) and data_file.lower().endswith('.txt'):
                with open(full_path_name, 'r') as f:
                    samples = []
                    for line in f:
                        l = line.strip()
                        if l == CONVERSATION_SEP:
                            self.conversations.append(samples)
                            samples = []
                        else:
                            samples.append({"text": l})

                    if len(samples):  # Add the last one
                        self.conversations.append(samples)

    def get_conversations(self):
        return self.conversations
