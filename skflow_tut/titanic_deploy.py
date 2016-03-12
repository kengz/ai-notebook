import pandas
import tensorflow as tf
import skflow

train = pandas.read_csv('data/titanic_train.csv')

print(train[['Age', 'SibSp', 'Fare']].fillna(0)[:1])
print(train['Survived'].fillna(0)[:2])

classifier = skflow.TensorFlowEstimator.restore('./models/titanic_dnn_1')

res = classifier.predict(train[['Age', 'SibSp', 'Fare']].fillna(0)[:2])
print(res)

# feed_dict = {x: train[['Age', 'SibSp', 'Fare']].fillna(0)[:1]}
# classification = tf.run(classifier, feed_dict)

# print(classification)