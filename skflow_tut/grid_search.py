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
  })

# fit and search
gs = grid_search.fit(iris.data, iris.target)
print(gs)
print(gs.best_estimator_)
# save the best model
gs.best_estimator_.save('./models/iris_grid_search')

classifier2 = skflow.TensorFlowEstimator.restore('./models/iris_grid_search')
score = accuracy_score(classifier2.predict(iris.data), iris.target)

print('Accuracy: {0:f}'.format(score))