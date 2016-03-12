from sklearn import datasets, cross_validation, metrics
import skflow

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

def my_model(X, y):
  """This is DNN with 10, 20, 10 hidden layers, and dropout of 0.9 probability."""
  layers = skflow.ops.dnn(X, [10,20,10], keep_prob=0.9)
  return skflow.models.logistic_regression(layers, y)

classifier = skflow.TensorFlowEstimator(model_fn=my_model, n_classes=3, steps=1000)
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(classifier.predict(X_test), y_test)
print('Accuracy: {0:f}'.format(score))
