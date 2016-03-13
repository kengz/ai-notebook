# Hyperparameter optimization

# - [scikit](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.grid_search). 
# builtin with scikit rmb. algo: grid search, random search.
# e.g.: http://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
# skflow e.g.: https://github.com/tensorflow/skflow/blob/master/skflow/tests/test_grid_search.py
# and: http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search and here http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
# - [hyperopt](https://github.com/hyperopt/hyperopt). algo: rand search, TPE
# - [optunity](http://optunity.readthedocs.org/en/latest/_modules/optunity/api.html#minimize). algo: search tree

# - Useful scikit examples: http://hyperopt.github.io/hyperopt-sklearn/

# hyperopt-sklearn algo:
# Random Search
# Tree of Parzen Estimators (TPE)
# Annealing
# Tree
# Gaussian Process Tree


from sklearn import datasets
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error

import tensorflow as tf
import skflow

iris = datasets.load_iris()

classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10], n_classes=3, steps=200)
grid_search = GridSearchCV(classifier, {
  'hidden_units': [[5,10,5], [10, 20, 10]],
  'learning_rate': [0.1, 0.01]
  }, n_jobs=4)

# fit and search
gs = grid_search.fit(iris.data, iris.target)
print(gs)
print(gs.best_estimator_)
# save the best model
gs.best_estimator_.save('./models/iris_grid_search')

classifier2 = skflow.TensorFlowEstimator.restore('./models/iris_grid_search')
score = accuracy_score(classifier2.predict(iris.data), iris.target)

print('Accuracy: {0:f}'.format(score))