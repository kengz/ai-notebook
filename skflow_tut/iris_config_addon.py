from sklearn import datasets, cross_validation, metrics

import skflow

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# You can define you configurations by providing a ConfigAddon object to
# estimator to control session configurations, e.g. num_cores and gpu_memory_fraction
config_addon = skflow.addons.ConfigAddon(num_cores=3, gpu_memory_fraction=0.6)

# Build 3 layer DNN with 10, 20, 10 units respecitvely.
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10,20,10], n_classes=3, steps=200, config_addon=config_addon)

# Fit and predict.
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(classifier.predict(X_test), y_test)
print('Accuracy: {0:f}'.format(score))
