import pandas
from sklearn import datasets
import tensorflow as tf
import skflow

# # Titanic model
# train = pandas.read_csv('data/titanic_train.csv')

# # print sample of the data type used for training
# print(train[['Age', 'SibSp', 'Fare']].fillna(0)[:1])
# print(train['Survived'].fillna(0)[:2])

# classifier = skflow.TensorFlowEstimator.restore('./models/titanic_dnn_1')

# outcome = classifier.predict(train[['Age', 'SibSp', 'Fare']].fillna(0)[:2])
# print(outcome)


# iris model
classifier = skflow.TensorFlowEstimator.restore('./models/iris_dnn')

iris = datasets.load_iris()
outcome = classifier.predict(iris.data[:1])
print(outcome)
print(iris.data[:1], iris.target[:1])