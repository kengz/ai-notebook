from sklearn import datasets, cross_validation, metrics
from sklearn import preprocessing

import skflow

boston = datasets.load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2, random_state=42)

# always rescale input to 0 mean, unit stdev
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)

# 2-layer fully conn DNN Regressor, 10, 10 units resp
regressor = skflow.TensorFlowDNNRegressor(hidden_units=[10, 20, 10], steps=5000, learning_rate=0.01, batch_size=1)

regressor.fit(X_train, y_train)

score = metrics.mean_squared_error(regressor.predict(scaler.fit_transform(X_test)), y_test)

print('MSE: {0:f}'.format(score))