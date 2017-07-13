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
        "",
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
        "",
        "Here you are: ",
        "Here is the result: ",
        "That's a little hard: ",
        "That was an tough one, and I had to use a calculator: ",
        "That's a little difficult, but I know how to solve it: ",
        "It was hard and took me a little while to figure it out. Here is the result: ",
        "It took me a little while, and finally I got the result: ",
        "I had to use my cell phone for this calculation. Here is the outcome: "
    ]

    def __init__(self, tokenized_data):
        """
        Args:
            tokenized_data: The parameter data needed for prediction.
        """
        self.tokenized_data = tokenized_data

    def get_story_any(self):
        stories = self.tokenized_data.stories
        _, content = random.choice(list(stories.items()))
        return content

    def get_story_name(self, story_name):
        stories = self.tokenized_data.stories
        return stories[story_name]

    def get_joke_any(self):
        jokes = self.tokenized_data.jokes
        return random.choice(jokes)

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
    def get_today_weekday():
        today = dt.date.today()
        weekday = cal.day_name[today.weekday()]
        return "{}, {:%B %d, %Y}".format(weekday, today)

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
            "and", "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
            "hundred", "thousand", "million", "billion", "trillion"]

        pat_op = re.compile(r'\s(plus|\+|minus|-|multiply|\*|divide|(divided\s+by)|/)\s',
                            re.IGNORECASE)
        pat_as = re.compile(r'((\sis\s)|=|(\sequals\s))', re.IGNORECASE)

        mat_op = re.search(pat_op, sentence)
        mat_as = re.search(pat_as, sentence)
        if mat_op and mat_as:  # contains an arithmetic operator and an assign operator
            number_rx = r'(?:{})'.format('|'.join(numbers))
            pat_num = re.compile(r'\b{0}(?:[\s-]{0})*\b|\d+'.format(number_rx), re.IGNORECASE)
            ind_list = [(m.start(0), m.end(0)) for m in re.finditer(pat_num, sentence)]
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
        for word in text.split():
            if word not in num_words:
                return -1

            scale, increment = num_words[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0

        return result + current


def call_function(func_info, tokenized_data=None, para_list=None):
    func_data = FunctionData(tokenized_data)

    func_dict = {
        'get_story_any': func_data.get_story_any,
        'get_story_name': func_data.get_story_name,
        'get_joke_any': func_data.get_joke_any,

        'get_date_time': FunctionData.get_date_time,
        'get_time': FunctionData.get_time,
        'get_today': FunctionData.get_today,
        'get_today_weekday': FunctionData.get_today_weekday,

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
    knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')

    td = TokenizedData(dict_file=dict_file, knbase_dir=knbs_dir, corpus_dir=corp_dir,
                       augment=False)

    sentences = [
        "How much is 12 + 14?",
        "How much is twelve thousand three hundred four plus two hundred fifty six?",
        "What is five hundred eighty nine multiply six?",
        "What is five hundred eighty nine divided by 89?",
        "What is five hundred and seventy nine divided by 89?",
        "How much is twelve thousand three hundred and four divided by two hundred fifty six?",
        "What is seven billion five million and four thousand three hundred and four plus "
        "five million and four thousand three hundred and four?",
        "How much is 16 - 23?",
        "How much is 144 * 12?",
        "How much is 23 / 26?",
        "99 + 19 = ?",
        "178 - 27 = ?",
        "99 + 19 =?",
        "178 - 27 =?",
        "99 + 19=",
        "178 - 27=",
        "99 + 19 equals?",
        "178 - 27 equals?",
        "What is 49 / 77?",
        "If x=12 and y=14, how much is x + y?",
        "If x=55 and y=19, how much is y - x?",
        "Let x=99, y=9, how much is x / y?",
        "What is 16 + 24 equals to?",
        "What is 16 + 24?"
    ]

    for sentence in sentences:
        print(sentence)
        print(FunctionData.check_arithmetic_pattern_and_replace(sentence))
        print()

    # print(call_function("get_story_name_para1_the_three_little_pigs", td))
    # print(call_function("get_story_any", td))
    # print(call_function("get_joke_any", td))
    # print(call_function("get_date_time"))
    # print("Today is {}.".format(call_function("get_today")))
    # print("It is {} now.".format(call_function("get_time")))
    # print("It is {} today.".format(call_function("get_today_weekday")))
    # print("Test {}.".format(call_function("something")))
