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


# Conv net

def max_pool_2x2(tensor_in):
  return tf.nn.max_pool(tensor_in, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_model(X,y):
  # reshape to 1D higher for input batch
  X =tf.reshape(X, [-1, 28,28,1])
  # for conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5,5], bias=True, activation=tf.nn.rely)
    h_pool1 = max_pool_2x2(h_conv1)

  # second convs layer will compute 64 features for each 5x5 patch
  with tf.variable_scope('conv_layer2'):
    h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5,5], bias=True, activation=tf.nn.relu)
    h_pool2 = max_pool_2x2(h_conv2)
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

  # densely conn layer with 1024 neurons
  h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.rely, keep_prob=0.5)
  return skflow.models.logistic_regression(h_fc1, y)

# Training, predicing
classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=10, batch_size=100, steps=20000, learning_rate=0.001)
classifier.fit(mnist.train.images, mnist.train.labels)
score = metrics.accuracy_score(classifier.predict(mnist.test.images), mnist.test.labels)
print('Accuracy: {0:f}'.format(score))
