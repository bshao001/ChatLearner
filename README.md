# ChatLearner

![](https://img.shields.io/badge/python-3.5.2-brightgreen.svg) ![](https://img.shields.io/badge/tensorflow-1.2.0-yellowgreen.svg)

A chatbot implemented in TensorFlow based on the sequence to sequence model.

## Notes and Highlights:
1. This implementation was created and tested under TensorFlow 1.2 GPU version. As there were significant changes among different versions of TensorFlow, especially in RNN-related areas, you may find it not work in earlier versions.

2. Because the copy.deepcopy method does not support placeholders, the TensorFlow seq2seq.py file was copied and slightly modified, so that encoder_cell and decoder_cell can be passed into embedding_attention_seq2seq method separately. Therefore, placeholders for dropout can be used.

3. The model_with_buckets method was not directly invoked. Instead, the source code was copied and then split into two parts. As a result, the inference graph and training graph can be expressed in two separate methods, making the code better organized. Another benefit of doing this is that the saver does not have to save the training graph, therefore, the model and data can be loaded more quickly at prediction time.

4. Different from most other open source implementations of the seq2seq model (for chatbot or for translation), the model is restored from a saved meta file instead of being created from scratch, leading a much cleaner and more readable code.

5. Common techniques, such as word embedding, attention mechanism, bucketing for training, output projection are all implemented in this model. However, bucketing approach does not sound suitable for a chatbot as the length ratios of source sentences and target sentences can vary significantly, causing some kind of mess at prediction time.

6. The idea of data augmentation was inspired by the way people perform data augmentation when training a CNN. If you want to let a CNN know what a cat is, you perform some kind of affine transformation so that no matter how big a cat is, or where the cat locates in an image, the CNN model knows it is cat (or not). Here, we introduce extra paddings in the opposite side into the source input to help the RNN model filter out the noises introduced by the required padding. This significantly cleared the mess introduced by bucketing.

7. The knowledge base is introduced here to address the case problem (to certain extent) experienced in the prediction output in the end user interface. I believe it also makes sense to treat multi-word terms/names/phrases as atomic entities in both training and prediction. For example, “United States”, “New York”, and “James Gosling” are all treated as single words with the idea of so-called knowledge base here. This would be extremely helpful for a domain-specific chatbot.

8. Simple rule functions are embedded into the training data so that simple questions can be answered, such as:
   * "What time is it now?" or "What day is it today?"
   * "Read me a story please." or "Tell me a joke." It can then present stories and jokes randomly and not being limited by the sequence length. 
   * "How much is twelve thousand three hundred four plus two hundred fifty six?" or "How much is twelve thousand three-hundred and four divided by two-hundred-fifty-six?" or "If x=55 and y=19, how much is y - x?" or even "If x = 99 and y = 228 / x, how much is y?"

## Training Data (Papaya Data Set)
1. The training data, although still toy-sized, are mostly handcrafted. They were created to maintain a consistent role of the chatbot, who can therefore be trained to be polite, patient, humorous, and aware that he is a robot, but pretends to be a 9-year old boy named Papaya. Most of the sentence pairs were written by my son, Kenny Shao, who will be going to high school fall 2017. Please do give us credits by linking to this page in case you make use of the training data for any purposes.

2. Some of the scenario conversations were extracted and reorganized from http://www.eslfast.com/robot/. If your model can support context, it would work much better by utilizing these conversations.

## Training
Training is simple. Remember to create a folder named Result under the Data folder first. Then just run the following commands:

```bash
cd chatbot
python basicmodel.py
```

With the existing parameters in the file and the current Papaya training data set, it will be very easy to get to a perplexity of 1.06 at around epoch 30. It would be better if you let it run until it terminates by itself, i.e., reaching the point with the perplexity less than 1.04 or until the maximum epoch. You will be able to see the training results under Data/Result/ folder. Make sure they are 4 files exist: 
1. basic.data-00000-of-00001
2. basic.index
3. basic.meta
4. dicts.pickle

All these will be required for test and prediction.

# Test and Prediction
For testing and prediction, we don't have a web interface available yet. Try to use the following command interface for now:

```bash
cd chatbot
python botui.py
```

Wait until you get the command prompt "> ".

A demo test result is provided as well. Please check this to see how this chatbot behaves now: https://github.com/bshao001/ChatLearner/blob/master/Data/Test/responses.txt

## References and Credits:
1. Deep QA: https://github.com/Conchylicultor/DeepQA
2. Easy seq2seq: http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
