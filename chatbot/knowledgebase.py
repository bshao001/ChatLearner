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
import re
import os

UPPER_FILE = "upper_words.txt"
MULTI_FILE = "multi_words.txt"


class KnowledgeBase:
    def __init__(self):
        self.upper_words = {}
        self.multi_words = {}
        self.multi_max_cnt = 0

    def load_knbase(self, knbase_dir):
        """
        Args:
             knbase_dir: Name of the KnowledgeBase folder. The file names inside are fixed.
        """
        upper_file_name = os.path.join(knbase_dir, UPPER_FILE)
        multi_file_name = os.path.join(knbase_dir, MULTI_FILE)

        with open(upper_file_name, 'r') as upper_f:
            for line in upper_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                cap_words = ln.split(',')
                for cpw in cap_words:
                    tmp = cpw.strip()
                    self.upper_words[tmp.lower()] = tmp

        with open(multi_file_name, 'r') as multi_f:
            for line in multi_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                mul_words = ln.split(',')
                for mlw in mul_words:
                    tmp = re.sub('\s+', ' ', mlw).strip() # Replace multiple spaces with 1
                    self.multi_words[tmp.lower()] = tmp

                    word_cnt = len(mlw.split())
                    if word_cnt > self.multi_max_cnt:
                        self.multi_max_cnt = word_cnt