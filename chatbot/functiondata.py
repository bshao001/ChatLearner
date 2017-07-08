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

    def get_joke_any(self):
        jokes = self.tokenized_data.jokes
        max_joke_id = len(jokes) - 1
        joke_id = random.randint(0, max_joke_id)
        return jokes[joke_id]

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


def call_function(func_name, tokenized_data=None):
    func_data = FunctionData(tokenized_data)

    func_dict = {
        'get_story_any': func_data.get_story_any,
        'get_joke_any': func_data.get_joke_any,
        'get_date_time': FunctionData.get_date_time,
        'get_time': FunctionData.get_time,
        'get_today': FunctionData.get_today,
        'get_today_weekday': FunctionData.get_today_weekday
    }

    if func_name in func_dict:
        return func_dict[func_name]()
    else:
        return func_name

if __name__ == "__main__":
    import os
    from settings import PROJECT_ROOT
    from chatbot.tokenizeddata import TokenizedData

    dict_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'dicts.pickle')
    knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')

    td = TokenizedData(dict_file=dict_file, knbase_dir=knbs_dir, corpus_dir=corp_dir,
                       augment=False)

    print(call_function("get_story_any", td))
    print(call_function("get_joke_any", td))
    print(call_function("get_date_time"))
    print("Today is {}.".format(call_function("get_today")))
    print("It is {} now.".format(call_function("get_time")))
    print("It is {} today.".format(call_function("get_today_weekday")))
    print("Test {}.".format(call_function("something")))
