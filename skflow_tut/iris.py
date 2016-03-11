# import shutil

# import skflow
# from sklearn import datasets, metrics, cross_validation

# iris = datasets.load_iris()
# # print(iris.data)
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
# classifier.fit(X_train, y_train)
# score = metrics.accuracy_score(classifier.predict(X_test), y_test)
# print('Accuracy: {0:f}'.format(score))

# # Clean checkpoint folder if exists
# try:
#     shutil.rmtree('/tmp/skflow_examples/iris_custom_model')
# except OSError:
#     pass

# # Save model, parameters and learned variables.
# classifier.save('/tmp/skflow_examples/iris_custom_model')
# classifier = None

# ## Restore everything
# new_classifier = skflow.TensorFlowEstimator.restore('/tmp/skflow_examples/iris_custom_model')
# score = metrics.accuracy_score(y_test, new_classifier.predict(X_test))
# print('Accuracy: {0:f}'.format(score))