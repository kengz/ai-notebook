import shutil

import skflow
from sklearn import datasets, metrics, cross_validation

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, train_size=0.2, random_state=42)

classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10,20,10], n_classes=3, steps=800)
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(classifier.predict(X_test), y_test)
print('Accuracy: {0:f}'.format(score))

# Clean checkpoint folder if exists
try:
  shutil.rmtree('./models/iris_custom_model')
except OSError:
  pass

# Save model, parameters and learned variables.
classifier.save('./models/iris_custom_model')
classifier = None

## Restore everything
new_classifier = skflow.TensorFlowEstimator.restore('./models/iris_custom_model')
score = metrics.accuracy_score(y_test, new_classifier.predict(X_test))
print('Accuracy: {0:f}'.format(score))
