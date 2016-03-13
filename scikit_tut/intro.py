# http://scikit-learn.org/stable/tutorial/index.html

# import and modeling
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)
print(digits.target)

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
res = clf.predict(digits.data[-1:])
print(res)


# model saving
import pickle
# dump to string
# s = pickle.dumps(clf)
# print(s)
# clf2 = pickle.loads(s)
# If wanna save to file, use these:
pickle.dump( clf, open( "./models/clf.p", "wb" ) )
clf2 = pickle.load( open( "./models/clf.p", "rb" ) )
res2 = clf2.predict(digits.data[0:1])
print(res2)
print(digits.target[0])


# type casting
import numpy as np
from sklearn import random_projection

# default is float64
rng = np.random.RandomState(0)
X = rng.rand(10,2000)
X = np.array(X, dtype='float32')
print(X.dtype)

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)


clf = svm.SVC()
clf.fit(iris.data, iris.target)
print(list(clf.predict(iris.data[:3])))
# whereas use the actual name (unmapped)
clf.fit(iris.data, iris.target_names[iris.target])
print(list(clf.predict(iris.data[:3])))



# Refitting, update parameters
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

clf = svm.SVC()
clf.set_params(kernel='linear').fit(X, y)
print(clf.predict(X_test))

# update params, refit
clf.set_params(kernel='rbf').fit(X, y)
print(clf.predict(X_test))
