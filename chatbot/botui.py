import numpy as np
import os
import sys
import tensorflow as tf

from settings import PROJECT_ROOT
from chatbot.tokenizeddata import TokenizedData
from chatbot.botpredictor import BotPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def bot_ui():
    dict_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'dicts.pickle')
    td = TokenizedData(seq_length=10, dict_file=dict_file)

    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    with tf.Session() as sess:
        predictor = BotPredictor(sess, td, res_dir, 'basic')

        print("Welcome to Chat with ChatLearner!")
        print("Type exit and press enter to end the conversation.")
        # Waiting from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            if sentence.strip() == 'exit':
                print("Thank you for using ChatLearner. Bye.")
                break

            dec_outputs = predictor.predict(sentence)

            word_ids = []
            for out in dec_outputs:
                word_ids.append(np.argmax(out))

            print(td.word_ids_to_str(word_ids))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

if __name__ == "__main__":
    bot_ui()
