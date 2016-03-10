import random
import pandas
import numpy as np
import tensorflow as tf
from sklearn import metrics, cross_validation

import skflow

random.seed(42)

data = pandas.read_csv('./data/titanic_train.csv')
X = data[['Embarked']]
y = data[['Survived']]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2, random_state=42)

embarked_classes = X_train['Embarked'].unique()
print('Embarked classes for training are', embarked_classes)

cat_processor = skflow.preprocessing.CategoricalProcessor()
X_train = np.array(list(cat_processor.fit_transform(X_train)))
X_test = np.array(list(cat_processor.transform(X_test)))

n_classes = len(cat_processor.vocabularies_[0])

EMBEDDING_SIZE = 3

def categorical_model(X, y):
  features = skflow.ops.categorical_variable(
    X, n_classes, embedding_size=EMBEDDING_SIZE, name='embarked')
  return skflow.models.logistic_regression(tf.squeeze(features, [1]), y)

classifier = skflow.TensorFlowEstimator(model_fn=categorical_model, n_classes=2)
classifier.fit(X_train, y_train)

print(metrics.accuracy_score(classifier.predict(X_test), y_test))
print(metrics.roc_auc_score(classifier.predict(X_test), y_test))

def one_hot_categorical_model(X, y):
  features = skflow.ops.one_hot_matrix(X, n_classes)
  return skflow.models.logistic_regression(tf.squeeze(features, [1]), y)

classifier = skflow.TensorFlowEstimator(model_fn=one_hot_categorical_model, n_classes=2, steps=1000, learning_rate=0.01)
classifier.fit(X_train, y_train)

print(metrics.accuracy_score(classifier.predict(X_test), y_test))
print(metrics.roc_auc_score(classifier.predict(X_test), y_test))
