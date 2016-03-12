from sklearn import metrics
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import skflow

# Download n load data
mnist = input_data.read_data_sets('MNIST_data')


# Linear classifier

classifier = skflow.TensorFlowLinearClassifier(n_classes=10, batch_size=100, steps=1000, learning_rate=0.01)
classifier.fit(mnist.train.images, mnist.train.labels)
score = metrics.accuracy_score(classifier.predict(mnist.test.images), mnist.test.labels)
print('Accuracy: {0:f}'.format(score))

