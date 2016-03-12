from sklearn.pipeline import Pipeline
from sklearn import datasets, cross_validation, metrics
from sklearn.preprocessing import StandardScaler

import skflow

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, train_size=0.2, random_state=42)

# It's useful to scale to ensure Stochastic Gradient Descent will do the right thing
scaler = StandardScaler()

# Build 3 layer DNN with 10, 20, 10 units respecitvely.
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10,20,10], n_classes=3, steps=800)

# All estimators in a pipeline, except the last one, must be transformers (i.e. must have a transform method).
pipeline = Pipeline([('scaler', scaler), ('DNNclassifier', classifier)])

pipeline.fit(X_train, y_train)

# Fit and predict.
score = metrics.accuracy_score(pipeline.predict(X_test), y_test)
print('Accuracy: {0:f}'.format(score))
