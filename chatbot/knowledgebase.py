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
"""This class is only used at inference time."""
import os

UPPER_FILE = "upper_words.txt"
STORIES_FILE = "stories.txt"
JOKES_FILE = "jokes.txt"


class KnowledgeBase:
    def __init__(self):
        self.upper_words = {}
        self.stories = {}
        self.jokes = []

    def load_knbase(self, knbase_dir):
        """
        Args:
             knbase_dir: Name of the KnowledgeBase folder. The file names inside are fixed.
        """
        upper_file_name = os.path.join(knbase_dir, UPPER_FILE)
        stories_file_name = os.path.join(knbase_dir, STORIES_FILE)
        jokes_file_name = os.path.join(knbase_dir, JOKES_FILE)

        with open(upper_file_name, 'r') as upper_f:
            for line in upper_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                cap_words = ln.split(',')
                for cpw in cap_words:
                    tmp = cpw.strip()
                    self.upper_words[tmp.lower()] = tmp

        with open(stories_file_name, 'r') as stories_f:
            s_name, s_content = '', ''
            for line in stories_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                if ln.startswith('_NAME:'):
                    if s_name != '' and s_content != '':
                        self.stories[s_name] = s_content
                        s_name, s_content = '', ''
                    s_name = ln[6:].strip().lower()
                elif ln.startswith('_CONTENT:'):
                    s_content = ln[9:].strip()
                else:
                    s_content += ' ' + ln.strip()

            if s_name != '' and s_content != '':  # The last one
                self.stories[s_name] = s_content

        with open(jokes_file_name, 'r') as jokes_f:
            for line in jokes_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                self.jokes.append(ln)