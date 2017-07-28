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
import calendar as cal
import datetime as dt
import random
import re
import time


class FunctionData:
    easy_list = [
        "", "",
        "Here you are: ",
        "Here is the result: ",
        "That's easy: ",
        "That was an easy one: ",
        "It was a piece of cake: ",
        "That's simple, and I know how to solve it: ",
        "That wasn't hard. Here is the result: ",
        "Oh, I know how to deal with this: "
    ]
    hard_list = [
        "", "",
        "Here you are: ",
        "Here is the result: ",
        "That's a little hard: ",
        "That was an tough one, and I had to use a calculator: ",
        "That's a little difficult, but I know how to solve it: ",
        "It was hard and took me a little while to figure it out. Here is the result: ",
        "It took me a little while, and finally I got the result: ",
        "I had to use my cell phone for this calculation. Here is the outcome: "
    ]

    def __init__(self, tokenized_data, html_format):
        """
        Args:
            tokenized_data: The parameter data needed for prediction.
            html_format: Whether out_sentence is in HTML format.
        """
        self.tokenized_data = tokenized_data
        self.html_format = html_format

    def get_story_any(self):
        stories = self.tokenized_data.stories
        _, content = random.choice(list(stories.items()))
        if not self.html_format:
            content = re.sub(r'_np_', '', content)
        return content

    def get_story_name(self, story_name):
        stories = self.tokenized_data.stories
        content = stories[story_name]
        if not self.html_format:
            content = re.sub(r'_np_', '', content)
        return content

    def get_joke_any(self):
        jokes = self.tokenized_data.jokes
        content = random.choice(jokes)
        if not self.html_format:
            content = re.sub(r'_np_', '', content)
        return content

    @staticmethod
    def get_date_time():
        return time.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def get_time():
        return time.strftime("%I:%M %p")

    @staticmethod
    def get_today():
        return "{:%B %d, %Y}".format(dt.date.today())

    @staticmethod
    def get_weekday(day_delta):
        now = dt.datetime.now()
        if day_delta == 'd_2':
            day_time = now - dt.timedelta(days=2)
        elif day_delta == 'd_1':
            day_time = now - dt.timedelta(days=1)
        elif day_delta == 'd1':
            day_time = now + dt.timedelta(days=1)
        elif day_delta == 'd2':
            day_time = now + dt.timedelta(days=2)
        else:
            day_time = now

        weekday = cal.day_name[day_time.weekday()]
        return "{}, {:%B %d, %Y}".format(weekday, day_time)

    @staticmethod
    def get_number_plus(num1, num2):
        res = num1 + num2
        desc = random.choice(FunctionData.easy_list)
        return "{}{} + {} = {}".format(desc, num1, num2, res)

    @staticmethod
    def get_number_minus(num1, num2):
        res = num1 - num2
        desc = random.choice(FunctionData.easy_list)
        return "{}{} - {} = {}".format(desc, num1, num2, res)

    @staticmethod
    def get_number_multiply(num1, num2):
        res = num1 * num2
        if num1 > 100 and num2 > 100 and num1 % 2 == 1 and num2 % 2 == 1:
            desc = random.choice(FunctionData.hard_list)
        else:
            desc = random.choice(FunctionData.easy_list)
        return "{}{} * {} = {}".format(desc, num1, num2, res)

    @staticmethod
    def get_number_divide(num1, num2):
        if num2 == 0:
            return "Sorry, but that does not make sense as the divisor cannot be zero."
        else:
            res = num1 / num2
            if isinstance(res, int):
                if 50 < num1 != num2 > 50:
                    desc = random.choice(FunctionData.hard_list)
                else:
                    desc = random.choice(FunctionData.easy_list)
                return "{}{} / {} = {}".format(desc, num1, num2, res)
            else:
                if num1 > 20 and num2 > 20:
                    desc = random.choice(FunctionData.hard_list)
                else:
                    desc = random.choice(FunctionData.easy_list)
                return "{}{} / {} = {:.2f}".format(desc, num1, num2, res)

    @staticmethod
    def check_arithmetic_pattern_and_replace(sentence):
        pat_matched, ind_list, num_list = FunctionData.contains_arithmetic_pattern(sentence)
        if pat_matched:
            s1, e1 = ind_list[0]
            s2, e2 = ind_list[1]
            # Leave spaces around the special tokens so that NLTK knows they are separate tokens
            new_sentence = sentence[:s1] + ' _num1_ ' + sentence[e1:s2] + ' _num2_ ' + sentence[e2:]
            return True, new_sentence, num_list
        else:
            return False, sentence, num_list

    @staticmethod
    def contains_arithmetic_pattern(sentence):
        numbers = [
            "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
            "hundred", "thousand", "million", "billion", "trillion"]

        pat_op = re.compile(
            r'\s(plus|add|added|sum|\+|minus|subtract|subtracted|-|times|multiply|multiplied|product|\*|divide|(divided\s+by)|/)\s',
            re.IGNORECASE)
        pat_as = re.compile(r'((\bis\b)|=|(\bequals\b)|(\bget\b))', re.IGNORECASE)

        mat_op = re.search(pat_op, sentence)
        mat_as = re.search(pat_as, sentence)
        if mat_op and mat_as:  # contains an arithmetic operator and an assign operator
            # Replace all occurrences of word "and" with 3 whitespaces before feeding to
            # the pattern matcher.
            pat_and = re.compile(r'\band\b', re.IGNORECASE)
            tmp_sentence = pat_and.sub('   ', sentence)

            number_rx = r'(?:{})'.format('|'.join(numbers))
            pat_num = re.compile(r'\b{0}(?:(?:\s+(?:and\s+)?|-){0})*\b|\d+'.format(number_rx),
                                 re.IGNORECASE)
            ind_list = [(m.start(0), m.end(0)) for m in re.finditer(pat_num, tmp_sentence)]
            num_list = []
            if len(ind_list) == 2:  # contains exactly two numbers
                for start, end in ind_list:
                    text = sentence[start:end]
                    text_int = FunctionData.text2int(text)
                    if text_int == -1:
                        return False, [], []
                    num_list.append(text_int)

                return True, ind_list, num_list

        return False, [], []

    @staticmethod
    def text2int(text):
        if text.isdigit():
            return int(text)

        num_words = {}
        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        num_words["and"] = (1, 0)
        for idx, word in enumerate(units):
            num_words[word] = (1, idx)
        for idx, word in enumerate(tens):
            num_words[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            num_words[word] = (10 ** (idx * 3 or 2), 0)

        current = result = 0
        for word in text.replace("-", " ").lower().split():
            if word not in num_words:
                return -1

            scale, increment = num_words[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0

        return result + current


def call_function(func_info, tokenized_data=None, para_list=None, html_format=False):
    func_data = FunctionData(tokenized_data, html_format=html_format)

    func_dict = {
        'get_story_any': func_data.get_story_any,
        'get_story_name': func_data.get_story_name,
        'get_joke_any': func_data.get_joke_any,

        'get_date_time': FunctionData.get_date_time,
        'get_time': FunctionData.get_time,
        'get_today': FunctionData.get_today,
        'get_weekday': FunctionData.get_weekday,

        'get_number_plus': FunctionData.get_number_plus,
        'get_number_minus': FunctionData.get_number_minus,
        'get_number_multiply': FunctionData.get_number_multiply,
        'get_number_divide': FunctionData.get_number_divide
    }

    para1_index = func_info.find('_para1_')
    para2_index = func_info.find('_para2_')
    if para1_index == -1:  # No parameter at all
        func_name = func_info
        if func_name in func_dict:
            return func_dict[func_name]()
    else:
        func_name = func_info[:para1_index]
        if para2_index == -1:  # Only one parameter
            func_para = func_info[para1_index+7:]
            return func_dict[func_name](func_para)
        else:
            func_para1 = func_info[para1_index+7:para2_index]
            func_para2 = func_info[para2_index+7:]
            if para_list is not None:
                num1_val = para_list[0]
                num2_val = para_list[1]

                if func_para1 == '_num1_' and func_para2 == '_num2_':
                    return func_dict[func_name](num1_val, num2_val)
                elif func_para1 == '_num2_' and func_para2 == '_num1_':
                    return func_dict[func_name](num2_val, num1_val)

    return "You beat me to it, and I cannot tell which is which for this question."

if __name__ == "__main__":
    import os
    from settings import PROJECT_ROOT
    from chatbot.tokenizeddata import TokenizedData

    dict_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'dicts.pickle')
    td = TokenizedData(dict_file=dict_file)

    print(call_function('get_story_any', td, html_format=True))
    print(call_function('get_story_any', td, html_format=False))
    print(call_function('get_joke_any', td, html_format=True))
    print(call_function('get_joke_any', td, html_format=False))
    print(call_function('get_weekday_para1_d_2'))
    print(call_function('get_weekday_para1_d_1'))
    print(call_function('get_weekday_para1_d0'))
    print(call_function('get_weekday_para1_d1'))
    print(call_function('get_weekday_para1_d2'))
