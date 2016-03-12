import skflow

classifier = skflow.TensorFlowEstimator.restore('./models/mnist_convnet')


## First Convolutional Layer
print('1st Convolutional Layer weights and Bias')
print(classifier.get_tensor_value('conv_layer1/convolution/filters:0'))
print(classifier.get_tensor_value('conv_layer1/convolution/bias:0'))

## Second Convolutional Layer
print('2nd Convolutional Layer weights and Bias')
print(classifier.get_tensor_value('conv_layer2/convolution/filters:0'))
print(classifier.get_tensor_value('conv_layer2/convolution/bias:0'))

## Densely Connected Layer
print('Densely Connected Layer weights')
print(classifier.get_tensor_value('dnn/layer0/Linear/Matrix:0'))

## Logistic Regression weights
print('Logistic Regression weights')
print(classifier.get_tensor_value('logistic_regression/weights:0'))
