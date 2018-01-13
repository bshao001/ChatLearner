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
"""This file is only used at inference time."""
import re


def check_patterns_and_replace(question):
    pat_matched, new_sentence, para_list = _check_arithmetic_pattern_and_replace(question)

    if not pat_matched:
        pat_matched, new_sentence, para_list = _check_not_username_pattern_and_replace(new_sentence)

    if not pat_matched:
        pat_matched, new_sentence, para_list = _check_username_callme_pattern_and_replace(new_sentence)

    return pat_matched, new_sentence, para_list


def _check_arithmetic_pattern_and_replace(sentence):
    pat_matched, ind_list, num_list = _contains_arithmetic_pattern(sentence)
    if pat_matched:
        s1, e1 = ind_list[0]
        s2, e2 = ind_list[1]
        # Leave spaces around the special tokens so that NLTK knows they are separate tokens
        new_sentence = sentence[:s1] + ' _num1_ ' + sentence[e1:s2] + ' _num2_ ' + sentence[e2:]
        return True, new_sentence, num_list
    else:
        return False, sentence, num_list


def _contains_arithmetic_pattern(sentence):
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
                text_int = _text2int(text)
                if text_int == -1:
                    return False, [], []
                num_list.append(text_int)

            return True, ind_list, num_list

    return False, [], []


def _text2int(text):
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


def _check_not_username_pattern_and_replace(sentence):
    import nltk

    tokens = nltk.word_tokenize(sentence)
    tmp_sentence = ' '.join(tokens[:]).strip()

    pat_not_but = re.compile(r'(\s|^)my\s+name\s+is\s+(not|n\'t)\s+(.+?)(\s\.|\s,|\s!)\s*but\s+(.+?)(\s\.|\s,|\s!|$)',
                             re.IGNORECASE)
    mat_not_but = re.search(pat_not_but, tmp_sentence)

    pat_not = re.compile(r'(\s|^)my\s+name\s+is\s+(not|n\'t)\s+(.+?)(\s\.|\s,|\s!|$)', re.IGNORECASE)
    mat_not = re.search(pat_not, tmp_sentence)

    para_list = []
    found = 0
    if mat_not_but:
        wrong_name = mat_not_but.group(3).strip()
        correct_name = mat_not_but.group(5).strip()
        para_list.append(correct_name)
        new_sentence = sentence.replace(wrong_name, ' _ignored_ ', 1).replace(correct_name, ' _name_ ', 1)
        # print("User name is not: {}, but {}.".format(wrong_name, correct_name))
        found += 1
    elif mat_not:
        wrong_name = mat_not.group(3).strip()
        new_sentence = sentence.replace(wrong_name, ' _ignored_ ', 1)
        # print("User name is not: {}.".format(wrong_name))
        found += 1
    else:
        new_sentence = sentence
        # print("Wrong name not found.")

    if found >= 1:
        return True, new_sentence, para_list
    else:
        return False, sentence, para_list


def _check_username_callme_pattern_and_replace(sentence):
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


if __name__ == "__main__":
    sentence = "My name is jack brown. Please call me Mr. Brown."
    print("# {}".format(sentence))
    _, ns, _ = _check_username_callme_pattern_and_replace(sentence)
    print(ns)

    sentence = "My name is Bo Shao."
    print("# {}".format(sentence))
    _, ns, _ = _check_username_callme_pattern_and_replace(sentence)
    print(ns)

    sentence = "You can call me Dr. Shao."
    print("# {}".format(sentence))
    _, ns, _ = _check_username_callme_pattern_and_replace(sentence)
    print(ns)

    sentence = "Call me Ms. Tailor please."
    print("# {}".format(sentence))
    _, ns, _ = _check_username_callme_pattern_and_replace(sentence)
    print(ns)

    sentence = "My name is Mark. Please call me Mark D."
    print("# {}".format(sentence))
    _, ns, _ = _check_username_callme_pattern_and_replace(sentence)
    print(ns)

    sentence = "My name is not just Shao, but Bo Shao."
    print("# {}".format(sentence))
    _, ns, _ = _check_not_username_pattern_and_replace(sentence)
    print(ns)

    sentence = "My name is not just Shao."
    print("# {}".format(sentence))
    _, ns, _ = _check_not_username_pattern_and_replace(sentence)
    print(ns)