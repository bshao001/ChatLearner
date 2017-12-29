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
    ask_name_list = [
        "May I also have your name, please?",
        "And, how should I call you, please?",
        "And, What do you want me to call you, dear sir or madam?"
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

        pat_op1 = re.compile(
            r'\s(plus|add|added|\+|minus|subtract|subtracted|-|times|multiply|multiplied|\*|divide|(divided\s+by)|/)\s',
            re.IGNORECASE)
        pat_op2 = re.compile(r'\s((sum\s+of)|(product\s+of))\s', re.IGNORECASE)
        pat_as = re.compile(r'((\bis\b)|=|(\bequals\b)|(\bget\b))', re.IGNORECASE)

        mat_op1 = re.search(pat_op1, sentence)
        mat_op2 = re.search(pat_op2, sentence)
        mat_as = re.search(pat_as, sentence)
        if (mat_op1 or mat_op2) and mat_as:  # contains an arithmetic operator and an assign operator
            # Replace all occurrences of word "and" with 3 whitespaces before feeding to
            # the pattern matcher.
            pat_and = re.compile(r'\band\b', re.IGNORECASE)
            if mat_op1:
                tmp_sentence = pat_and.sub('   ', sentence)
            else:  # Do not support word 'and' in the English numbers any more as that can be ambiguous.
                tmp_sentence = pat_and.sub('_T_', sentence)

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

    """
    # Rule 5: User name, call me information, and last question and answer
    """
    @staticmethod
    def check_username_callme_pattern_and_replace(sentence):
        import nltk

        tokens = nltk.word_tokenize(sentence)
        tmp_sentence = ' '.join(tokens[:]).strip()

        pat_name = re.compile(r'(\s|^)my\s+name\s+is\s+(.+?)(\s\.|\s,|\s!|$)', re.IGNORECASE)
        pat_call = re.compile(r'(\s|^)call\s+me\s+(.+?)(\s(please|pls))?(\s\.|\s,|\s!|$)', re.IGNORECASE)

        mat_name = re.search(pat_name, tmp_sentence)
        mat_call = re.search(pat_call, tmp_sentence)

        para_list = []
        found = 0
        if mat_name:
            user_name = mat_name.group(2).strip()
            para_list.append(user_name)
            new_sentence = sentence.replace(user_name, ' _name_ ', 1)
            # print("User name is: {}.".format(user_name))
            found += 1
        else:
            para_list.append('')  # reserve the slot
            new_sentence = sentence
            # print("User name not found.")

        if mat_call:
            call_me = mat_call.group(2).strip()
            para_list.append(call_me)
            new_sentence = new_sentence.replace(call_me, ' _callme_ ')
            # print("Call me {}.".format(call_me))
            found += 1
        else:
            para_list.append('')
            # print("call me not found.")

        if found >= 1:
            return True, new_sentence, para_list
        else:
            return False, sentence, para_list

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
        if call_me and call_me.strip() != '':
            if punc_type == 'comma0':
                return ", {}".format(call_me)
            else:
                return call_me
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

        'execute_pending_action_and_reply': func_data.execute_pending_action_and_reply
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
#
#     knbs = KnowledgeBase()
#     knbs.load_knbase(os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase'))
#
#     print(call_function('get_story_any', knbs, html_format=True))
#     print(call_function('get_story_any', knbs, html_format=False))
#     print(call_function('get_joke_any', knbs, html_format=True))
#     print(call_function('get_joke_any', knbs, html_format=False))
#     print(call_function('get_weekday_para1_d_2'))
#     print(call_function('get_weekday_para1_d_1'))
#     print(call_function('get_weekday_para1_d0'))
#     print(call_function('get_weekday_para1_d1'))
#     print(call_function('get_weekday_para1_d2'))
#
#     sentence = "My name is jack brown. Please call me Mr. Brown."
#     print("# {}".format(sentence))
#     _, ns, _ = FunctionData.check_username_callme_pattern_and_replace(sentence)
#     print(ns)
#
#     sentence = "My name is Bo Shao."
#     print("# {}".format(sentence))
#     _, ns, _ = FunctionData.check_username_callme_pattern_and_replace(sentence)
#     print(ns)
#
#     sentence = "You can call me Dr. Shao."
#     print("# {}".format(sentence))
#     _, ns, _ = FunctionData.check_username_callme_pattern_and_replace(sentence)
#     print(ns)
#
#     sentence = "Call me Ms. Tailor please."
#     print("# {}".format(sentence))
#     _, ns, _ = FunctionData.check_username_callme_pattern_and_replace(sentence)
#     print(ns)
#
#     sentence = "My name is Mark. Please call me Mark D."
#     print("# {}".format(sentence))
#     _, ns, _ = FunctionData.check_username_callme_pattern_and_replace(sentence)
#     print(ns)
