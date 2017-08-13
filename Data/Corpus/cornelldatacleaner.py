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
import ast
import nltk
import re

MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]


class CornellDataCleaner:
    """
    Load the cornell movie dialog corpus and clean the data briefly.

    Available from here:
    http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
    
    Please download the zip file, extract the movie_lines.txt and movie_conversations.txt,
    and copy them to the corpus_dir below before you run this script.
    
    Other cleaning work could have been automated, but were later found during the manual 
    cleaning process.
     1. Remove all '*' characters.
     2. Remove markers like [3], [4], [5], and [beat]. And remove all these brackets.
    """
    def __init__(self, corpus_dir):
        """
        Args:
            corpus_dir: File directory where to load the corpus.
        """
        line_file = os.path.join(corpus_dir, "movie_lines.txt")
        conv_file = os.path.join(corpus_dir, "movie_conversations.txt")

        self.lines = self.load_lines(line_file, MOVIE_LINES_FIELDS)
        self.conversations = self.load_conversations(conv_file, MOVIE_CONVERSATIONS_FIELDS)

    def load_conversations(self, filename, fields):
        """
        Args:
            filename: File to load.
            fields: Fields to extract.
        Return:
            dict<dict<str>>: the extracted fields for each line
        """
        conv_list = []

        with open(filename, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")

                # Extract fields
                conv_obj = {}
                for i, field in enumerate(fields):
                    conv_obj[field] = values[i]

                # Convert string to list (conv_obj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                line_ids = ast.literal_eval(conv_obj["utteranceIDs"])

                # Reassemble lines
                conv_obj["lines"] = []
                for line_id in line_ids:
                    conv_obj["lines"].append(self.lines[line_id])

                conv_list.append(conv_obj)

        return conv_list

    def write_cleaned_conversations(self, out_file):
        """
        Args:
            out_file: File to save the cleaned data. 
        """
        pat_curse = re.compile(
            r'\b(ass|asshole|bastard|bitch|child-fucker|damn|fuck|fucking|motherfucker|motherfucking|'
            r'nigger|shit|shitass)\b',
            re.IGNORECASE)

        with open(out_file, 'a') as f_out:
            for c in self.conversations:
                written = False
                for i in range(0, len(c['lines']) - 1, 2):
                    input_line = c['lines'][i]['text'].strip()
                    target_line = c['lines'][i + 1]['text'].strip()

                    if all(ord(char) < 128 for char in input_line) and \
                            all(ord(char) < 128 for char in target_line):
                        input_line = self.get_formatted_line(input_line)
                        target_line = self.get_formatted_line(target_line)

                        # Only discard those conversation pairs in which the answer line contains
                        # any common cursing words
                        if re.search(pat_curse, target_line):
                            continue

                        # Discard sentences starting with an ellipsis
                        if input_line.startswith("...") or target_line.startswith("..."):
                            continue

                        # Discard sentences starting with a dash
                        if input_line.startswith("-") or target_line.startswith("-"):
                            continue

                        # This is to speed up the parsing below
                        if len(input_line) > 160 or len(target_line) > 160:
                            continue

                        in_tokens = nltk.word_tokenize(input_line)
                        tg_tokens = nltk.word_tokenize(target_line)
                        if 8 <= len(in_tokens) < 36 and 6 <= len(tg_tokens) <= 40:
                            f_out.write("{}\n".format(input_line))
                            f_out.write("{}\n".format(target_line))
                            written = True

                if written:
                    f_out.write("===\n")

    @staticmethod
    def get_formatted_line(line):
        pat_dot = re.compile(r'\.\s+\.')
        pat_dash = re.compile(r'-\s+-')
        pat_html = re.compile(r'<.*?>')

        # Use formal ellipsis and dashes
        while re.search(pat_dot, line):
            line = re.sub(pat_dot, '..', line)

        while re.search(pat_dash, line):
            line = re.sub(pat_dash, '--', line)

        line = re.sub('\.{3,}', '... ', line)
        line = re.sub('-{2,}', ' -- ', line)

        # Use formal apostrophe
        line = line.replace(' \' ', '\'')

        # Remove extra spaces
        line = re.sub('\s+', ' ', line).strip()
        line = line.replace(' .', '.').replace(' ?', '?').replace(' !', '!')

        # Remove HTML tags
        line = re.sub(pat_html, '', line)

        # Remove extra punctuations and m's
        line = re.sub('\?{2,}', '?', line)
        line = re.sub('!{2,}', '!', line)
        line = re.sub('m{3,}', 'mm', line)

        return line

    @staticmethod
    def load_lines(filename, fields):
        """
        Args:
            filename: File to load.
            fields: Fields to extract.
        Return:
            dict<dict<str>>: the extracted fields for each line
        """
        lines = {}

        with open(filename, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")

                # Extract fields
                line_obj = {}
                for i, field in enumerate(fields):
                    line_obj[field] = values[i]

                lines[line_obj['lineID']] = line_obj

        return lines

if __name__ == "__main__":
    from settings import PROJECT_ROOT

    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
    cd = CornellDataCleaner(corp_dir)
    print("{} conversations loaded.".format(len(cd.conversations)))

    out_file = os.path.join(corp_dir, 'cornell_cleaned.txt')
    cd.write_cleaned_conversations(out_file)