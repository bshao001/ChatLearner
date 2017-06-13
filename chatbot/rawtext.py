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
CONVERSATION_SEP = "==="


class RawText:
    def __init__(self):
        self.conversations = []

    def load_corpus(self, data_file):
        """
        Args:
             data_file: Name of the file containing the text format of conversations.
        """
        with open(data_file, 'r') as f:
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
