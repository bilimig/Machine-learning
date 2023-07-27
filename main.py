# A binary classifier that recognizes one of the digits in MNIST.

import numpy as np
import matplotlib.pyplot as plt
# Applying Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Basically doing prediction but named forward as its
# performing Forward-Propagation
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

# Calling the predict() function
def classify(X, w):
    return np.round(forward(X, w))


# Computing Loss over using logistic regression
def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


# calculating gradient
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

# calling the training function for desired no. of iterations
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print('Iteration %4d => Loss: %.20f' % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

# Doing inference to test our model
def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" %
          (correct_results, total_examples, success_percent))

# Test it
import mnist as data

DIGIT = 5

# X = data.load_images("train-images-idx3-ubyte.gz")
# Y = data.load_labels("train-labels-idx1-ubyte.gz").flatten()
# digits = X[Y == DIGIT]
# np.random.shuffle(digits)
#
# rows, columns = 3, 15
# fig = plt.figure()
# for i in range(rows * columns):
#     ax = fig.add_subplot(rows, columns, i + 1)
#     ax.axis('off')
#     ax.imshow(digits[i].reshape((28, 28)), cmap="Greys")
# plt.show()

w = train(data.X_train, data.Y_train, iterations=100, lr=1e-5)
test(data.X_test, data.Y_test, w)

