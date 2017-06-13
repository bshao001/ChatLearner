import numpy as np
import os
import sys
import tensorflow as tf

from settings import PROJECT_ROOT
from chatbot.tokenizeddata import TokenizedData
from chatbot.botpredictor import BotPredictor


def bot_ui():
    data_file = os.path.join(PROJECT_ROOT, 'Data', 'Corpus', 'basic_conv.txt')
    td = TokenizedData(seq_length=10, data_file=data_file)

    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
    with tf.Session() as sess:
        predictor = BotPredictor(sess, td, res_dir, 'basic')

        # Waiting from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
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
