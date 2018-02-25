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
"""This class is only used at inference time."""
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
    ask_howru_list = [
        "And you?",
        "How are you?",
        "How about yourself?"
    ]
    ask_name_list = [
        "May I also have your name, please?",
        "Would you also like to tell me your name, please?",
        "And, how should I call you, please?",
        "And, what do you want me to call you, dear sir or madam?"
    ]

    def __init__(self, knowledge_base, chat_session, html_format):
        """
        Args:
            knowledge_base: The knowledge base data needed for prediction.
            chat_session: The chat session object that can be read and written.
            html_format: Whether out_sentence is in HTML format.
        """
        self.knowledge_base = knowledge_base
        self.chat_session = chat_session
        self.html_format = html_format

    """
    # Rule 2: Date and Time
    """
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

    """
    # Rule 3: Stories and Jokes, and last topic
    """
    def get_story_any(self):
        self.chat_session.last_topic = "STORY"
        self.chat_session.keep_topic = True

        stories = self.knowledge_base.stories
        _, content = random.choice(list(stories.items()))
        if not self.html_format:
            content = re.sub(r'_np_', '', content)
        return content

    def get_story_name(self, story_name):
        self.chat_session.last_topic = "STORY"
        self.chat_session.keep_topic = True

        stories = self.knowledge_base.stories
        content = stories[story_name]
        if not self.html_format:
            content = re.sub(r'_np_', '', content)
        return content

    def get_joke_any(self):
        self.chat_session.last_topic = "JOKE"
        self.chat_session.keep_topic = True

        jokes = self.knowledge_base.jokes
        content = random.choice(jokes)
        if not self.html_format:
            content = re.sub(r'_np_', '', content)
        return content

    def continue_last_topic(self):
        if self.chat_session.last_topic == "STORY":
            self.chat_session.keep_topic = True
            return self.get_story_any()
        elif self.chat_session.last_topic == "JOKE":
            self.chat_session.keep_topic = True
            return self.get_joke_any()
        else:
            return "Sorry, but what topic do you prefer?"

    """
    # Rule 4: Arithmetic ops
    """
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

    """
    # Rule 5: User name, call me information, and last question and answer
    """
    def ask_howru_if_not_yet(self):
        howru_asked = self.chat_session.howru_asked
        if howru_asked:
            return ""
        else:
            self.chat_session.howru_asked = True
            return random.choice(FunctionData.ask_howru_list)

    def ask_name_if_not_yet(self):
        user_name = self.chat_session.user_name
        call_me = self.chat_session.call_me
        if user_name or call_me:
            return ""
        else:
            return random.choice(FunctionData.ask_name_list)

    def get_user_name_and_reply(self):
        user_name = self.chat_session.user_name
        if user_name and user_name.strip() != '':
            return user_name
        else:
            return "Did you tell me your name? Sorry, I missed that."

    def get_callme(self, punc_type):
        call_me = self.chat_session.call_me
        user_name = self.chat_session.user_name

        if call_me and call_me.strip() != '':
            if punc_type == 'comma0':
                return ", {}".format(call_me)
            else:
                return call_me
        elif user_name and user_name.strip() != '':
            if punc_type == 'comma0':
                return ", {}".format(user_name)
            else:
                return user_name
        else:
            return ""

    def get_last_question(self):
        # Do not record this pair as the last question and answer
        self.chat_session.update_pair = False

        last_question = self.chat_session.last_question
        if last_question is None or last_question.strip() == '':
            return "You did not say anything."
        else:
            return "You have just said: {}".format(last_question)

    def get_last_answer(self):
        # Do not record this pair as the last question and answer
        self.chat_session.update_pair = False

        last_answer = self.chat_session.last_answer
        if last_answer is None or last_answer.strip() == '':
            return "I did not say anything."
        else:
            return "I have just said: {}".format(last_answer)

    def update_user_name(self, new_name):
        return self.update_user_name_and_call_me(new_name=new_name)

    def update_call_me(self, new_call):
        return self.update_user_name_and_call_me(new_call=new_call)

    def update_user_name_and_call_me(self, new_name=None, new_call=None):
        user_name = self.chat_session.user_name
        call_me = self.chat_session.call_me
        # print("{}; {}; {}; {}".format(user_name, call_me, new_name, new_call))

        if user_name and new_name and new_name.strip() != '':
            if new_name.lower() != user_name.lower():
                self.chat_session.update_pending_action('update_user_name_confirmed', None, new_name)
                return "I am confused. I have your name as {}. Did I get it correctly?".format(user_name)
            else:
                return "You told me your name already. Thank you, {}, for assuring me.".format(user_name)

        if call_me and new_call and new_call.strip() != '':
            if new_call.lower() != call_me.lower():
                self.chat_session.update_pending_action('update_call_me_confirmed', new_call, None)
                return "You wanted me to call you {}. Would you like me to call you {} now?"\
                    .format(call_me, new_call)
            else:
                return "Thank you for letting me again, {}.".format(call_me)

        if new_call and new_call.strip() != '':
            if new_name and new_name.strip() != '':
                self.chat_session.user_name = new_name

            self.chat_session.call_me = new_call
            return "Thank you, {}.".format(new_call)
        elif new_name and new_name.strip() != '':
            self.chat_session.user_name = new_name
            return "Thank you, {}.".format(new_name)

        return "Sorry, I am confused. I could not figure out what you meant."

    def update_user_name_enforced(self, new_name):
        if new_name and new_name.strip() != '':
            self.chat_session.user_name = new_name
            return "OK, thank you, {}.".format(new_name)
        else:
            self.chat_session.user_name = None  # Clear the existing user_name, if any.
            return "Sorry, I am lost."

    def update_call_me_enforced(self, new_call):
        if new_call and new_call.strip() != '':
            self.chat_session.call_me = new_call
            return "OK, got it. Thank you, {}.".format(new_call)
        else:
            self.chat_session.call_me = None  # Clear the existing call_me, if any.
            return "Sorry, I am totally lost."

    def update_user_name_and_reply_papaya(self, new_name):
        user_name = self.chat_session.user_name

        if new_name and new_name.strip() != '':
            if user_name:
                if new_name.lower() != user_name.lower():
                    self.chat_session.update_pending_action('update_user_name_confirmed', None, new_name)
                    return "I am confused. I have your name as {}. Did I get it correctly?".format(user_name)
                else:
                    return "Thank you, {}, for assuring me your name. My name is Papaya.".format(user_name)
            else:
                self.chat_session.user_name = new_name
                return "Thank you, {}. BTW, my name is Papaya.".format(new_name)
        else:
            return "My name is Papaya. Thanks."

    def correct_user_name(self, new_name):
        if new_name and new_name.strip() != '':
            self.chat_session.user_name = new_name
            return "Thank you, {}.".format(new_name)
        else:
            # Clear the existing user_name and call_me information
            self.chat_session.user_name = None
            self.chat_session.call_me = None
            return "I am totally lost."

    def clear_user_name_and_call_me(self):
        self.chat_session.user_name = None
        self.chat_session.call_me = None

    def execute_pending_action_and_reply(self, answer):
        func = self.chat_session.pending_action['func']
        if func == 'update_user_name_confirmed':
            if answer.lower() == 'yes':
                reply = "Thank you, {}, for confirming this.".format(self.chat_session.user_name)
            else:
                new_name = self.chat_session.pending_action['No']
                self.chat_session.user_name = new_name
                reply = "Thank you, {}, for correcting me.".format(new_name)
        elif func == 'update_call_me_confirmed':
            if answer.lower() == 'yes':
                new_call = self.chat_session.pending_action['Yes']
                self.chat_session.call_me = new_call
                reply = "Thank you, {}, for correcting me.".format(new_call)
            else:
                reply = "Thank you. I will continue to call you {}.".format(self.chat_session.call_me)
        else:
            reply = "OK, thanks."  # Just presents a reply that is good for most situations

        # Clear the pending action anyway
        self.chat_session.clear_pending_action()
        return reply

    """
    # Other Rules: Client Code
    """
    def client_code_show_picture_randomly(self, picture_name):
        if not self.html_format:  # Ignored in the command line interface
            return ''
        else:
            return ' _cc_start_show_picture_randomly_para1_' + picture_name + '_cc_end_'


def call_function(func_info, knowledge_base=None, chat_session=None, para_list=None,
                  html_format=False):
    func_data = FunctionData(knowledge_base, chat_session, html_format=html_format)

    func_dict = {
        'get_date_time': FunctionData.get_date_time,
        'get_time': FunctionData.get_time,
        'get_today': FunctionData.get_today,
        'get_weekday': FunctionData.get_weekday,

        'get_story_any': func_data.get_story_any,
        'get_story_name': func_data.get_story_name,
        'get_joke_any': func_data.get_joke_any,
        'continue_last_topic': func_data.continue_last_topic,

        'get_number_plus': FunctionData.get_number_plus,
        'get_number_minus': FunctionData.get_number_minus,
        'get_number_multiply': FunctionData.get_number_multiply,
        'get_number_divide': FunctionData.get_number_divide,

        'ask_howru_if_not_yet': func_data.ask_howru_if_not_yet,
        'ask_name_if_not_yet': func_data.ask_name_if_not_yet,
        'get_user_name_and_reply': func_data.get_user_name_and_reply,
        'get_callme': func_data.get_callme,
        'get_last_question': func_data.get_last_question,
        'get_last_answer': func_data.get_last_answer,

        'update_user_name': func_data.update_user_name,
        'update_call_me': func_data.update_call_me,
        'update_user_name_and_call_me': func_data.update_user_name_and_call_me,
        'update_user_name_enforced': func_data.update_user_name_enforced,
        'update_call_me_enforced': func_data.update_call_me_enforced,
        'update_user_name_and_reply_papaya': func_data.update_user_name_and_reply_papaya,

        'correct_user_name': func_data.correct_user_name,
        'clear_user_name_and_call_me': func_data.clear_user_name_and_call_me,

        'execute_pending_action_and_reply': func_data.execute_pending_action_and_reply,

        'client_code_show_picture_randomly': func_data.client_code_show_picture_randomly
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
            if func_para == '_name_' and para_list is not None and len(para_list) >= 1:
                return func_dict[func_name](para_list[0])
            elif func_para == '_callme_' and para_list is not None and len(para_list) >= 2:
                return func_dict[func_name](para_list[1])
            else:  # The parameter value was embedded in the text (part of the string) of the training example.
                return func_dict[func_name](func_para)
        else:
            func_para1 = func_info[para1_index+7:para2_index]
            func_para2 = func_info[para2_index+7:]
            if para_list is not None and len(para_list) >= 2:
                para1_val = para_list[0]
                para2_val = para_list[1]

                if func_para1 == '_num1_' and func_para2 == '_num2_':
                    return func_dict[func_name](para1_val, para2_val)
                elif func_para1 == '_num2_' and func_para2 == '_num1_':
                    return func_dict[func_name](para2_val, para1_val)
                elif func_para1 == '_name_' and func_para2 == '_callme_':
                    return func_dict[func_name](para1_val, para2_val)

    return "You beat me to it, and I cannot tell which is which for this question."


# if __name__ == "__main__":
#     import os
#     from settings import PROJECT_ROOT
#     from chatbot.knowledgebase import KnowledgeBase
#     from chatbot.sessiondata import ChatSession
#
#     knbs = KnowledgeBase()
#     knbs.load_knbase(os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase'))
#
#     cs = ChatSession(1)
#
#     print(call_function('get_story_any', knbs, cs, html_format=True))
#     print(call_function('get_story_any', knbs, cs, html_format=False))
#     print(call_function('get_joke_any', knbs, cs, html_format=True))
#     print(call_function('get_joke_any', knbs, cs, html_format=False))
#     print(call_function('get_weekday_para1_d_2'))
#     print(call_function('get_weekday_para1_d_1'))
#     print(call_function('get_weekday_para1_d0'))
#     print(call_function('get_weekday_para1_d1'))
#     print(call_function('get_weekday_para1_d2'))
