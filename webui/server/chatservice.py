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
import tensorflow as tf

import tornado.httpserver
import tornado.ioloop

from webui.server.tornadows import soaphandler
from webui.server.tornadows import webservices
from webui.server.tornadows import complextypes
from webui.server.tornadows import xmltypes
from webui.server.tornadows.soaphandler import webservice

from settings import PROJECT_ROOT
from chatbot.tokenizeddata import TokenizedData
from chatbot.botpredictor import BotPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SessionSentence(complextypes.ComplexType):
    sessionId = int
    sentence = str


class ChatService(soaphandler.SoapHandler):
    def initialize(self, **kwargs):
        self.predictor = kwargs.pop('predictor')

    @webservice(_params=[xmltypes.Integer, xmltypes.String], _returns=SessionSentence)
    def reply(self, sessionId, question):
        """
        Args:
            sessionId: Reserved for future use. 
            question: The sentence given by the end user who chats with the ChatLearner. 
        Returns:
            outputSentence: The sessionId is the same as in the input for validation purpose. 
            The answer is the response from the ChatLearner.
        """
        answer = self.predictor.predict(question, html_format=True)

        outputSentence = SessionSentence()
        outputSentence.sessionId = sessionId
        outputSentence.sentence = answer
        return outputSentence

if __name__ == "__main__":
    print("Loading saved dictionaries for words and IDs ... ")
    dict_file = os.path.join(PROJECT_ROOT, 'Data', 'Result', 'dicts.pickle')
    td = TokenizedData(dict_file=dict_file)

    print("Creating TF session ...")
    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')

    with tf.Session() as sess:
        predictor = BotPredictor(sess, td, res_dir, 'basic')
        # Predict one and discard the output, as the very first one is slower.
        predictor.predict("Hello")

        service = [('ChatService', ChatService, {'predictor': predictor})]
        app = webservices.WebService(service)
        ws = tornado.httpserver.HTTPServer(app)
        ws.listen(8080)
        print("Web service started.")
        tornado.ioloop.IOLoop.instance().start()
