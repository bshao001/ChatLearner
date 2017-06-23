# ChatLearner

A chatbot implemented in TensorFlow based on the seq2seq model.

Notes:
1. Because the copy.deepcopy method does not support placeholders, the TensorFlow seq2seq.py file was copied and slightly modified, so that encoder_cell and decoder_cell can be passed into embedding_attention_seq2seq method separately.
2. The model_with_buckets method was not directly invoked. Instead, the source code was copied and then split into two parts. As a result, the inference graph and training graph can be expressed in two separate methods. Another benefit of doing this is that the saver does not have to save the training graph, therefore, the model and data can be loaded more quickly at prediction time.
3. Different from most other open source implementations of the seq2seq model (for chatbot or for translation), the model is restored from a saved meta file instead of being created from scratch, leading a much cleaner and more readable code.
4. Common techniques, such as word embedding, attention mechanism, bucketing for training, output projection are all implemented in this model. However, bucketing approach does not sound suitable for a chatbot as the length ratios of source sentences and target sentences can vary significantly, causing some kind of mess at prediction time.
5. The training data, although still very small, are all handcrafted. They were created to maintain a consistent role of the chatbot, who is trained to be aware that he is a robot, but pretends to be a 9-year old boy. Most of the sentence pairs were written by my son, Kenny Shao, who will be going to high school fall 2017. Please do give us credits by linking to this page in case you make use of the training data for any purposes.

References:
1. Deep QA: https://github.com/Conchylicultor/DeepQA
2. Easy seq2seq: http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
