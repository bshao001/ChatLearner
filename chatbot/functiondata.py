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
        return "{} + {} = {}".format(num1, num2, res)

    @staticmethod
    def get_number_minus(num1, num2):
        res = num1 - num2
        return "{} - {} = {}".format(num1, num2, res)

    @staticmethod
    def get_number_multiply(num1, num2):
        res = num1 * num2
        return "{} * {} = {}".format(num1, num2, res)

    @staticmethod
    def get_number_divide(num1, num2):
        if num2 == 0:
            return "Sorry, but that does not make sense as the divisor cannot be zero."
        else:
            res = num1 / num2
            if isinstance(res, int):
                return "{} / {} = {}".format(num1, num2, res)
            else:
                return "{} / {} = {:.2f}".format(num1, num2, res)

    @staticmethod
    def check_arithmetic_pattern_and_replace(sentence):
        pat_matched, ind_list, num_list = FunctionData.contains_arithmetic_pattern(sentence)
        if pat_matched:
            s1, e1 = ind_list[0]
            s2, e2 = ind_list[1]
            new_sentence = sentence[:s1] + '_num1_' + sentence[e1:s2] + '_num2_' + sentence[e2:]
            return True, new_sentence, num_list
        else:
            return False, sentence, num_list

    @staticmethod
    def contains_arithmetic_pattern(sentence):
        pat_op = re.compile(r'\s+(plus|\+|minus|-|multiply|\*|divide|(divided\s+by)|/)\s+')
        pat_as = re.compile(r'\s(is|=|equals)\s')

        mat_op = re.search(pat_op, sentence)
        mat_as = re.search(pat_as, sentence)
        if mat_op and mat_as:  # contains an arithmetic operator and an assign operator
            ind_list = [(m.start(0), m.end(0)) for m in re.finditer(r'\d+', sentence)]
            num_list = []
            if len(ind_list) == 2:  # contains exactly two numbers
                for start, end in ind_list:
                    str = sentence[start:end]
                    num_list.append(int(str))

                return True, ind_list, num_list

        return False, [], []


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

    return func_info

if __name__ == "__main__":
    import os
    from settings import PROJECT_ROOT
    from chatbot.tokenizeddata import TokenizedData

    dict_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'dicts.pickle')
    knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')

    td = TokenizedData(dict_file=dict_file, knbase_dir=knbs_dir, corpus_dir=corp_dir,
                       augment=False)

    print(call_function("get_story_name_para1_the_three_little_pigs", td))
    print(call_function("get_story_any", td))
    print(call_function("get_joke_any", td))
    print(call_function("get_date_time"))
    print("Today is {}.".format(call_function("get_today")))
    print("It is {} now.".format(call_function("get_time")))
    print("It is {} today.".format(call_function("get_today_weekday")))
    print("Test {}.".format(call_function("something")))
