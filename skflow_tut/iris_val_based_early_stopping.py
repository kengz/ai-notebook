from tensorflow.python.platform import googletest
from sklearn import datasets, cross_validation, metrics

import skflow

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_train, y_train, train_size=0.2, random_state=42)

val_monitor = skflow.monitors.ValidationMonitor(X_val, y_val, early_stopping_rounds=200, n_classes=3)

# classifier with early stopping on training data
classifier1 = skflow.TensorFlowDNNClassifier(hidden_units=[10,20,10], n_classes=3, steps=2000)

# Fit and predict.
classifier1.fit(X_train, y_train)
score1 = metrics.accuracy_score(classifier1.predict(X_test), y_test)

# classifier with early stopping on validation data
classifier2 = skflow.TensorFlowDNNClassifier(hidden_units=[10,20,10], n_classes=3, steps=2000)
classifier2.fit(X_train, y_train, val_monitor)
score2 = metrics.accuracy_score(classifier2.predict(X_test), y_test)

# in many applications, the score is improved by using early stopping on val data
print('Accuracy: {0:f}'.format(score1))
print('Accuracy: {0:f}'.format(score2))
