import skflow
import tensorflow as tf
from sklearn import datasets, cross_validation, metrics

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

def my_model(X, y):
  """
  This is DNN with 10, 20, 10 hidden layers, and dropout of 0.5 probability.

  Note: If you want to run this example with multiple GPUs, Cuda Toolkit 7.0 and 
  CUDNN 6.5 V2 from NVIDIA need to be installed beforehand. 
  """
  with tf.device('/cpu:0'):
    layers = skflow.ops.dnn(X, [10,20,10], keep_prob=0.5)
  with tf.device('/cpu:1'):
    return skflow.models.logistic_regression(layers, y)

classifier = skflow.TensorFlowEstimator(model_fn=my_model, n_classes=3)
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(classifier.predict(X_test), y_test)
print('Accuracy: {0:f}'.format(score))
