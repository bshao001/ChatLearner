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
"""
This class is only used at inference time.
In the case of a production system, the SessionData has to be maintained so that ChatSession objects
can expire and then be cleaned from the memory.
"""


class SessionData:
    def __init__(self):
        self.session_dict = {}

    def add_session(self):
        items = self.session_dict.items()
        if items:
            last_id = max(k for k, v in items)
        else:
            last_id = 0
        new_id = last_id + 1

        self.session_dict[new_id] = ChatSession(new_id)
        return new_id

    def get_session(self, session_id):
        return self.session_dict[session_id]


class ChatSession:
    def __init__(self, session_id):
        """
        Args:
            session_id: The integer ID of the chat session.
        """
        self.session_id = session_id

        self.howru_asked = False

        self.user_name = None
        self.call_me = None

        self.last_question = None
        self.last_answer = None
        self.update_pair = True

        self.last_topic = None
        self.keep_topic = False

        # Will be storing the information of the pending action:
        # The action function name, the parameter for answer yes, and the parameter for answer no.
        self.pending_action = {'func': None, 'Yes': None, 'No': None}

    def before_prediction(self):
        self.update_pair = True
        self.keep_topic = False

    def after_prediction(self, new_question, new_answer):
        self._update_last_pair(new_question, new_answer)
        self._clear_last_topic()

    def _update_last_pair(self, new_question, new_answer):
        """
        Last pair is updated after each prediction except in a few cases.
        """
        if self.update_pair:
            self.last_question = new_question
            self.last_answer = new_answer

    def _clear_last_topic(self):
        """
        Last topic is cleared after each prediction except in a few cases.
        """
        if not self.keep_topic:
            self.last_topic = None

    def update_pending_action(self, func_name, yes_para, no_para):
        self.pending_action['func'] = func_name
        self.pending_action['Yes'] = yes_para
        self.pending_action['No'] = no_para

    def clear_pending_action(self):
        """
        Pending action is, and only is, cleared at the end of function: execute_pending_action_and_reply.
        """
        self.pending_action['func'] = None
        self.pending_action['Yes'] = None
        self.pending_action['No'] = None
