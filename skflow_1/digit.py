import random
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

import tensorflow as tf

import skflow

random.seed(42)

digits = datasets.load_digits()
X = digits.images
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def conv_model(X, y):
  X = tf.expand_dims(X, 3)
  features = tf.reduce_max(skflow.ops.conv2d(X, 12, [3,3]), [1,2])
  features = tf.reshape(features, [-1, 12])
  return skflow.models.logistic_regression(features, y)

classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=10, steps=500, learning_rate=0.05, batch_size=128)
classifier.fit(X_train, y_train)
print(accuracy_score(classifier.predict(X_test), y_test))