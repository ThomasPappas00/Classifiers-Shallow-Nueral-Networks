# Handwritten Digit Classifiers
Developing three types of shallow Neural Network classifiers from scratch for distinguishing handwritten images (28x28 pixels) of the digits zero (0) and eight (8) for the _Data Processing and Learning Algorithms_ class.

## Description
This is an attempt to compare three methods of designing NNs, namely **Cross Entropy**, **Exponential** and **Hinge**, based on the cumulative error rate of each network. Every NN has a dimention of 784 × 300 × 1 (28*28=784 inputs, 300 hidden layer and 1 output [the decision]) and is trained on the MNIST training data (5500 in total) using the _Stochastic Gradient Descent_ algorithm and tested on the MNIST testing data (973 in total).
