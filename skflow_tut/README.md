# skflow

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

- finish skflow tutorials
- read skflow api doc
- see seq2seq


##### Next:
- embeddings, word2vec, distr rep https://github.com/tensorflow/skflow/issues/68
- nlp data categorization
- TF tasks: understand loss number, play with shape, init tensor, reshape, conv2d
- hyperparameter selection
- tensorflow serving https://tensorflow.github.io/serving/serving_basic

- learn scikit http://scikit-learn.org/stable/
- learn pandas http://pandas.pydata.org/pandas-docs/stable/10min.html
