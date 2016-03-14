# do dises: 
# http://radimrehurek.com/gensim/tut1.html#corpus-streaming-one-document-at-a-time
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
# https://radimrehurek.com/gensim/models/doc2vec.html

import os
import gensim, logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on sentences
model = gensim.models.Word2Vec(sentences, min_count=1)
print(model)

# print(gensim.models.word2vec.Text8Corpus(os.getcwd()+'/data/text8'))


# class MySentences(object):
#   def __init__(self, dirname):
#     self.dirname = dirname

#   def __iter__(self):
#     for fname in os.listdir(self.dirname):
#       for line in open(os.path.join(self.dirname, fname)):
        # yield line.split()

# sentences = MySentences(os.getcwd()+'/data/gensim')
# model = gensim.models.Word2Vec(sentences)
# print(model)
# print(sentences)

# model = gensim.models.Word2Vec(sentences, min_count=10, size=200, workers=4)


# print(model)