# skflow

## Care while building:

- StandardScaler, fit_transform, Pipeline
- categorization of data
- one-hot matrix for feature space expansion
- hyperparameter fine tuning, best done parallelized and automatically
- model save and load
- for small dataset, do kfold cross validation: http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html#cross-validation-generators


## NLP with deep conv net

- see le paper: conv neural net for sentence classification

## Installation

```shell
sudo apt-get -y install build-essential python3-dev python3-setuptools python3-numpy python3-scipy python3-pip libatlas-dev libatlas3gf-base
pip3 install scikit-learn
sudo pip3 install git+git://github.com/tensorflow/skflow.git
pip3 install pandas
pip3 install matplotlib
sudo apt-get install python-matplotlib
```

debug install, try to install protobuf 3 beta: `sudo pip3 install protobuf==3.0.0b2`

##### Data download trick: 
`wget https://googledrive.com/host/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M/ag_news_csv.tar.gz`

skflow tutorial 1 on [here](https://medium.com/@ilblackdragon/tensorflow-tutorial-part-1-c559c63c0cb1#.feohjl3hx)

>Note on the parameters for the model —I’ve been putting some for an example, but learning rate, optimizer and how many steps you train a model can make a big difference. Usually, in real scenarios one would run hyper-parameter search to find an optimal set which improves cost or accuracy on the validation set.


Use the DNN to train sentence to intent, use embedding, word2vec, and the categorizer for intents

## Progression:

##### Today:

- gensim eg: https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec, https://radimrehurek.com/gensim/models/doc2vec.html, http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim, https://gist.github.com/balajikvijayan/9f7ab00f9bfd0bf56b14
- broken gensim tut: http://rare-technologies.com/word2vec-tutorial/
- finish skflow tutorials
- more hyperopt
- word2vec: https://github.com/piskvorky/gensim, https://github.com/danielfrg/word2vec, https://linanqiu.github.io/2015/10/07/word2vec-sentiment/, http://nbviewer.jupyter.org/github/danielfrg/word2vec/blob/master/examples/doc2vec.ipynb
- ok word2vec: . Go with gensim first, since it's more powerful. Otherwise fallback to the apache word2vec, 
- datejs tut
- see seq2seq
https://www.tensorflow.org/versions/r0.7/tutorials/image_recognition/index.html

scikit load csv:
```
import numpy as np
data = np.loadtxt('XXX.csv', delimiter=',')
```

##### Next:
- embeddings, word2vec, distr rep https://github.com/tensorflow/skflow/issues/68
- nlp data categorization
- TF tasks: understand loss number, play with shape, init tensor, reshape, conv2d

- learn scikit http://scikit-learn.org/stable/
- learn pandas http://pandas.pydata.org/pandas-docs/stable/10min.html
