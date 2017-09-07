# ChatLearner

![](https://img.shields.io/badge/python-3.5.2-brightgreen.svg)  ![](https://img.shields.io/badge/tensorflow-1.3.0-yellowgreen.svg?sanitize=true)

A chatbot implemented in TensorFlow based on the new sequence to sequence model, with certain rules integrated.

## Notes and Highlights:
1. This implementation was based on the new seq2seq model (dynamic RNN based) in TensorFlow version 1.3 (require 1.2.1 and later). The code was largely referenced on the tutorial of the new NMT model (https://github.com/tensorflow/nmt).
2. The repository also contains a chatbot implementation based on the legacy seq2seq model. In case you are interested in that, please check the Legacy_Chatbot branch at https://github.com/bshao001/ChatLearner/tree/Legacy_Chatbot.
3. As the new model is based on the dynamic RNN, a GPU (or CPU) can afford to train it with a larger batch size. With the legacy one, I had to train the model with batch size 64, while this one in batch size 128, therefore cutting the training time from about 10 hours to 5 hours based on the Papaya dataset (described below).
4. Although the implementation supports multiple GPU training (kind of model parallel), my experience with 2 GPU (NVIDIA GeForce GTX 1080 Ti) for one epoch took about 415 seconds, while a single GPU took about 384 seconds. Therefore, single GPU setting is preferred. In case you don't have a good GPU, CPU training for one epoch takes roughly one hour or 70 minutes.
5. Attention mechanism and beam search are both supported in this implementation.
6. Some rules are integrated to demo how to combine traditional rule-based chatbots with new deep learning models. If you are not interested, you can easily remove those line related to knowledgebase.py and functiondata.py. However, with these rules, the chatbot can answer certain categories of questions that a normal deep learning model cannot do. For example:
   * "What time is it now?" or "What day is it today?" or "What's the date yesterday?"
   * "Read me a story please." or "Tell me a joke." It can then present stories and jokes randomly and not being limited by the sequence length.
   * "How much is twelve thousand three hundred four plus two hundred fifty six?" or "What is the sum of five and six?" or "How much is twelve thousand three-hundred and four divided by two-hundred-fifty-six?" or "If x=55 and y=19, how much is y - x?" or "How much do you get if you subtract eight from one hundred?" or even "If x = 99 and y = 228 / x, how much is y?"

## Training Data (Papaya Data Set)
1. The training data are composed of two sets: the first set was handcrafted, and we created the samples in order to maintain a consistent role of the chatbot, who can therefore be trained to be polite, patient, humorous, and aware that he is a robot, but pretend to be a 9-year old boy named Papaya; the second set was cleaned from some online resources, including the scenario conversations designed for training robots, and the Cornell movie dialogs.
2. The training data set is split into three categories: two subsets will be augmented during the training, with different levels or times, while the third will not. The augmented subsets are to train the model with rules to follow, and some knowledge and common senses, while the third subset is just to help to train the language model.
3. The scenario conversations were extracted and reorganized from http://www.eslfast.com/robot/. If your model can support context, it would work much better by utilizing these conversations.
4. The original Cornell data set can be found at http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html. We cleaned it using a Python script (the script can also be found in the Corpus folder); we then cleaned it manually by quickly searching certain patterns. The final data is available here: https://github.com/bshao001/ChatLearner/blob/master/Data/Corpus/Augment0/cornell_cleaned_new.txt
5. The data files were already preprocessed with NLTK tokenizer so that they are ready to feed into the model using new dataset API in TensorFlow.

## Training
Other than Python 3.5.2, Numpy, and TensorFlow 1.3. You also need NLTK (Natural Language Toolkit) version 3.2.4, including its data.

Training is straightforward. Remember to create a folder named Result under the Data folder first. Then just run the following commands:

```bash
cd chatbot
python bottrainer.py
```

A good GPU is highly recommended for the training as it can be very time-consuming. To train the model using the existing parameters and the Papaya data set with a single GPU (NVIDIA GeForce GTX 1080 Ti), it will take about 5 hours to get the desired perplexity. You can modify the model parameters based on your available computing resources. You will be able to see the training results under Data/Result/ folder. Make sure the following 2 files exist as all these will be required for testing and prediction (the .meta file is optional as the inference model will be created independently): 

1. basic.data-00000-of-00001
2. basic.index

## Testing / Inference
For testing and prediction, we provide a simple command interface and a web-based interface. In order to quickly check how the trained model performs, use the following command interface:

```bash
cd chatbot
python botui.py
```

Wait until you get the command prompt "> ".

A demo test result is provided as well. Please check it to see how this chatbot behaves now (will be updated very soon): https://github.com/bshao001/ChatLearner/blob/master/Data/Test/responses.txt

## Web Interface
A SOAP-based web service architecture is implemented, with a Python server and a Java client. A nice GUI is also included for your reference. For details, please check: https://github.com/bshao001/ChatLearner/tree/master/webui

## References and Credits:
1. The new NMT model: https://github.com/tensorflow/nmt
2. Tornado Web Service: https://github.com/rancavil/tornado-webservices
