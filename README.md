# ChatLearner

![](https://img.shields.io/badge/python-3.5.2-brightgreen.svg) ![](https://img.shields.io/badge/tensorflow-1.2.0-yellowgreen.svg)

A chatbot implemented in TensorFlow based on the sequence to sequence model, with certain rules integrated.

## Notes and Highlights:
1. This implementation was created and tested under TensorFlow 1.2 GPU version. As there were significant changes among different versions of TensorFlow, especially in RNN-related areas, you may find it not work in earlier versions.

2. Because the copy.deepcopy method does not support placeholders, the TensorFlow seq2seq.py file was copied and slightly modified, so that encoder_cell and decoder_cell can be passed into embedding_attention_seq2seq method separately. Therefore, placeholders for dropout can be used.

3. The model_with_buckets method was not directly invoked. Instead, the source code was copied and then split into two parts. As a result, the inference graph and training graph can be expressed in two separate methods, making the code better organized. Another benefit of doing this is that the saver does not have to save the training graph, therefore, the model and data can be loaded more quickly at prediction time.

4. Different from most other open source implementations of the seq2seq model (for chatbot or for translation), the model is restored from a saved meta file instead of being created from scratch, leading a much cleaner and more readable code.

5. The idea of data augmentation was inspired by the way people perform data augmentation when training a CNN. If you want to let a CNN know what a cat is, you perform some kind of affine transformation so that no matter how big a cat is, or where the cat locates in an image, the CNN model knows it is cat (or not). Here, we introduce extra paddings in the opposite side into the source input to help the RNN model filter out the noises introduced by the required padding.

8. The knowledge base is introduced here to address the case problem (to certain extent) experienced in the prediction output in the end user interface. I believe it also makes sense to treat multi-word terms/names/phrases as atomic entities in both training and prediction. For example, “United States”, “New York”, and “James Gosling” are all treated as single words with the idea of so-called knowledge base here. This would be extremely helpful for a domain-specific chatbot.

8. Simple rule functions are embedded into the training data so that certain categories of questions can be answered. For example:
   * "What time is it now?" or "What day is it today?" or "What's the date yesterday?"
   * "Read me a story please." or "Tell me a joke." It can then present stories and jokes randomly and not being limited by the sequence length. 
   * "How much is twelve thousand three hundred four plus two hundred fifty six?" or "What is the sum of five and six?" or "How much is twelve thousand three-hundred and four divided by two-hundred-fifty-six?" or "If x=55 and y=19, how much is y - x?" or "How much do you get if you subtract eight from one hundred?" or even "If x = 99 and y = 228 / x, how much is y?"

## Training Data (Papaya Data Set)
1. The training data are composed of two sets: the first set was handcrafted, and we created the samples in order to maintain a consistent role of the chatbot, who can therefore be trained to be polite, patient, humorous, and aware that he is a robot, but pretend to be a 9-year old boy named Papaya; the second set was cleaned from some online resources, including the scenario conversations designed for training robots, and the Cornell movie dialogs.

2. The training data set is split into three categories: two subsets will be augmented during the training, with different levels or times, while the third will not. The augmented subsets are to train the model with rules to follow, and some knowledge and common senses, while the third subset is just to help to train the language model.

3. The scenario conversations were extracted and reorganized from http://www.eslfast.com/robot/. If your model can support context, it would work much better by utilizing these conversations.

4. The original Cornell data set can be found at http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html. We cleaned it using a Python script (the script can also be found in the Corpus folder); we then cleaned it manually by quickly searching certain patterns. The final data is available here: https://github.com/bshao001/ChatLearner/blob/master/Data/Corpus/Augment0/cornell_cleaned.txt

## Training
Other than Python 3.5.2, Numpy, and TensorFlow 1.2. You also need NLTK (Natural Language Toolkit) version 3.2.4, including its data.

Training is straightforward. Remember to create a folder named Result under the Data folder first. Then just run the following commands:

```bash
cd chatbot
python basicmodel.py
```

A good GPU is highly recommended for the training as it can be very time-consuming. To train the model using the existing parameters and the Papaya data set with a single GPU (NVIDIA GeForce GTX 1080 Ti), it will take about 9 hours to get the desired perplexity. You can modify the model parameters based on your available computing resources. You will be able to see the training results under Data/Result/ folder. Make sure the following 4 files exist as all these will be required for testing and prediction: 

1. basic.data-00000-of-00001
2. basic.index
3. basic.meta
4. dicts.pickle

## Testing
For testing and prediction, we provide a simple command interface and a web-based interface. In order to quickly check how the trained model performs, use the following command interface:

```bash
cd chatbot
python botui.py
```

Wait until you get the command prompt "> ".

A demo test result is provided as well. Please check it to see how this chatbot behaves now: https://github.com/bshao001/ChatLearner/blob/master/Data/Test/responses.txt

## Web Interface
A SOAP-based web service architecture is implemented, with a Python server and a Java client. A nice GUI is also included for your reference. For details, please check: https://github.com/bshao001/ChatLearner/tree/master/webui

## References and Credits:
1. Deep QA: https://github.com/Conchylicultor/DeepQA
2. Easy seq2seq: http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
3. Tornado Web Service: https://github.com/rancavil/tornado-webservices
