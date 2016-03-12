from sklearn import datasets, cross_validation, metrics

import skflow
import tensorflow as tf

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# setup exponential decay function
def exp_decay(global_step):
  return tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=100, decay_rate=0.001)

# use customized decay function in learning_rate
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10,20,10], n_classes=3, steps=800, learning_rate=exp_decay)

# Fit and predict.
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(classifier.predict(X_test), y_test)
print('Accuracy: {0:f}'.format(score))
