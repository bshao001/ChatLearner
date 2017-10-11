"""
This file was modified from the reddit parser on https://github.com/pender/chatbot-rnn.
When unzipped, the generated files contains conversations in the same format as those 
in other data files in this Papaya Data Set corpus. They will be further cleaned and 
processed using other scripts in this folder.
"""
from bz2 import BZ2File
import os
import json
import re
import sys

CONFIG_FILE = "redditparser_config.json"
FILE_SUFFIX = ".bz2"


class RedditParser(object):
    def __init__(self):
        with open(CONFIG_FILE, 'r') as cf:
            config = json.load(cf)

        self.input_file = config['input_file']
        self.output_dir = config['output_dir']
        self.output_file = config['output_file']
        self.report_file = config['report_file']
        self.comment_cache_size = config['comment_cache_size']
        self.output_file_size = config['output_file_size']
        self.print_every = config['print_every']

        self.subreddit_blacklist = set(config['subreddit_blacklist'])
        self.subreddit_whitelist = set(config['subreddit_whitelist'])
        self.substring_blacklist = set(config['substring_blacklist'])

    def parse(self):
        if not os.path.exists(self.input_file):
            print("File not found: {}".format(self.input_file))
            return
        if os.path.isfile(self.output_dir):
            print("A File with the same name already exists at output directory location: {}".
                  format(self.output_dir))
            return

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        subreddit_dict = {}
        comment_dict = {}
        cache_count = 0

        raw_data = self.get_raw_data_enumerator()
        output_handler = OutputHandler(os.path.join(self.output_dir, self.output_file),
                                       self.output_file_size)
        for cnt, line in enumerate(raw_data):
            line = line.decode("utf-8")
            if len(line) > 1 and (line[-1] == '}' or line[-2] == '}'):
                comment = json.loads(line)
                if self.post_qualifies(comment):
                    sub = comment['subreddit']
                    if sub in subreddit_dict:
                        subreddit_dict[sub] += 1
                    else:
                        subreddit_dict[sub] = 1
                    comment_dict[comment['name']] = RedditComment(comment)
                    cache_count += 1
                    if cache_count % self.print_every == 0:
                        print("\rCached {} comments. ".format(cache_count), end='')
                        sys.stdout.flush()
                    if cache_count > self.comment_cache_size:
                        print()
                        self.process_comment_cached(comment_dict)
                        self.write_comment_cached(comment_dict, output_handler)
                        self.write_report(subreddit_dict)
                        comment_dict.clear()
                        cache_count = 0

        print("\nRead all {} lines from {}.".format(cnt, self.input_file))
        self.process_comment_cached(comment_dict)
        self.write_comment_cached(comment_dict, output_handler)
        self.write_report(subreddit_dict)

    def get_raw_data_enumerator(self):
        print("Reading from {}".format(self.input_file))
        with BZ2File(self.input_file, "r") as raw_data:
            for line in raw_data:
                yield line

    def post_qualifies(self, json_object):
        body = json_object['body'].encode('ascii', 'ignore').strip()
        body = body.decode("utf-8")

        post_length = len(body)
        if post_length < 8 or post_length > 240:
            return False

        subreddit = json_object['subreddit']

        # Filter posts based on the configured whitelist and blacklist
        if len(self.subreddit_whitelist) > 0 and subreddit not in self.subreddit_whitelist:
            return False
        if len(self.subreddit_blacklist) > 0 and subreddit in self.subreddit_blacklist:
            return False
        if len(self.substring_blacklist) > 0:
            for substring in self.substring_blacklist:
                if body.find(substring) >= 0:
                    return False

        # Preprocess the comment text
        body = re.sub('[ \t\n]+', ' ', body) # Replace runs of whitespace with a single space.
        body = re.sub('\^', '', body) # Strip out carets.
        body = re.sub('\\\\', '', body) # Strip out backslashes.
        body = re.sub('&lt;', '<', body) # Replace '&lt;' with '<'
        body = re.sub('&gt;', '>', body) # Replace '&gt;' with '>'
        body = re.sub('&amp;', '&', body) # Replace '&amp;' with '&'

        post_length = len(body)
        if post_length < 8 or post_length > 240:
            return False

        json_object['body'] = body # Save our changes

        return True

    def process_comment_cached(self, comment_dict):
        i = 0
        for my_id, my_comment in comment_dict.items():
            i += 1
            if i % self.print_every == 0:
                print("\rProcessed {} comments".format(i), end='')
                sys.stdout.flush()

            if my_comment.parent_id is not None:  # If we're not a top-level post...
                if my_comment.parent_id in comment_dict:  # ...and the parent is in our data set...
                    parent = comment_dict[my_comment.parent_id]
                    if parent.child_id is None:  # If my parent doesn't already have a child, adopt me!
                        parent.child_id = my_id
                    else:  # My parent already has a child.
                        parent_previous_child = comment_dict[parent.child_id]
                        if parent.parent_id in comment_dict:  # If my grandparent is in our data set...
                            grandparent = comment_dict[parent.parent_id]
                            if my_comment.author == grandparent.author:
                                # If I share an author with grandparent, adopt me!
                                parent.child_id = my_id
                            elif (parent_previous_child.author != grandparent.author
                                and my_comment.score > parent_previous_child.score):
                                # If the existing child doesn't share an author with grandparent,
                                # higher score prevails.
                                parent.child_id = my_id
                        elif my_comment.score > parent_previous_child.score:
                            # If there's no grandparent, the higher-score child prevails.
                            parent.child_id = my_id
                else:
                    # Parent IDs that aren't in the data set get de-referenced.
                    my_comment.parent_id = None
        print()

    def write_comment_cached(self, comment_dict, output_handler):
        i = 0
        prev_print_count = 0
        for k, v in comment_dict.items():
            if v.parent_id is None and v.child_id is not None:
                comment = v
                depth = 0
                output_string = ""
                while comment is not None:
                    depth += 1
                    if depth % 2 == 1:
                        output_string += 'Q: '
                    else:
                        output_string += 'A: '
                    output_string += comment.body + '\n'
                    if comment.child_id in comment_dict:
                        comment = comment_dict[comment.child_id]
                    else:
                        comment = None
                        if depth % 2 == 0:
                            output_handler.write(output_string + '===\n')
                            i += depth
                            if i > prev_print_count + self.print_every:
                                prev_print_count = i
                                print("\rWrote {} comments".format(i), end='')
                                sys.stdout.flush()
        print()

    def write_report(self, subreddit_dict):
        out_report_file = os.path.join(self.output_dir, self.report_file)
        print("Updating subreddit report file")
        subreddit_list = sorted(subreddit_dict.items(), key=lambda x: -x[1])
        with open(out_report_file, "w") as f:
            for item in subreddit_list:
                f.write("{}: {}\n".format(*item))


class RedditComment(object):
    def __init__(self, json_object):
        self.body = json_object['body']
        self.score = json_object['ups'] - json_object['downs']
        self.author = json_object['author']
        self.parent_id = json_object['parent_id']
        self.child_id = None


class OutputHandler(object):
    def __init__(self, path, output_file_size):
        if path.endswith(FILE_SUFFIX):
            path = path[:-len(FILE_SUFFIX)]
        self.base_path = path
        self.output_file_size = output_file_size
        self.file_reference = None

    def write(self, data):
        if self.file_reference is None:
            self._get_current_path()
        self.file_reference.write(data.encode('ascii', 'ignore'))
        self.current_file_size += len(data)
        if self.current_file_size >= self.output_file_size:
            self.file_reference.close()
            self.file_reference = None

    def _get_current_path(self):
        i = 1
        while True:
            path = "{} {}{}".format(self.base_path, i, FILE_SUFFIX)
            if not os.path.exists(path): break
            i += 1
        self.current_path = path
        self.current_file_size = 0
        self.file_reference = BZ2File(self.current_path, "w")


if __name__ == '__main__':
    RedditParser().parse()
