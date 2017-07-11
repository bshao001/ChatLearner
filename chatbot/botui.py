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
import sys
import tensorflow as tf

from settings import PROJECT_ROOT
from chatbot.tokenizeddata import TokenizedData
from chatbot.botpredictor import BotPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def bot_ui():
    print("Loading saved dictionaries for words and IDs ... ")
    dict_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'dicts.pickle')
    td = TokenizedData(dict_file=dict_file)

    print("Creating TF session ...")
    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    with tf.Session() as sess:
        predictor = BotPredictor(sess, td, res_dir, 'basic')
        # Predict one and discard the output, as the very first one is slower.
        predictor.predict("Hello")

        print("Welcome to Chat with ChatLearner!")
        print("Type exit and press enter to end the conversation.")
        # Waiting from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            if sentence.strip() == 'exit':
                print("Thank you for using ChatLearner. Goodbye.")
                break

            print(predictor.predict(sentence))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

if __name__ == "__main__":
    bot_ui()
