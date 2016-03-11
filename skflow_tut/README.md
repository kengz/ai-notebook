# skflow

## Installation

```shell
sudo apt-get install build-essential python3-dev python3-setuptools python3-numpy python3-scipy python3-pip libatlas-dev libatlas3gf-base
pip3 install scikit-learn
pip3 install skflow
pip3 install pandas matplotlib
```

skflow tutorial 1 on [here](https://medium.com/@ilblackdragon/tensorflow-tutorial-part-1-c559c63c0cb1#.feohjl3hx)

>Note on the parameters for the model —I’ve been putting some for an example, but learning rate, optimizer and how many steps you train a model can make a big difference. Usually, in real scenarios one would run hyper-parameter search to find an optimal set which improves cost or accuracy on the validation set.


Use the DNN to train sentence to intent, use embedding, word2vec, and the categorizer for intents

bug fix: http://stackoverflow.com/questions/35789666/tensorflow-with-skflow-attributeerror-module-object-has-no-attribute-saver

## Steps:

finish skflow tutorials
https://drive.google.com/folderview?id=0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M#grid

embeddings, word2vec, https://github.com/tensorflow/skflow/issues/68
read skflow api doc
TF tasks: understand loss number, play with shape, init tensor, reshape, conv2d
nlp data categorization
hyperparameter selection

skflow nlp, distributed rep
tensorflow serving https://tensorflow.github.io/serving/serving_basic

learn scikit http://scikit-learn.org/stable/
learn pandas http://pandas.pydata.org/pandas-docs/stable/10min.html
