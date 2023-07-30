import numpy as np
import mnist as data


# Applying Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(logits):
    exponential = np.exp(logits)
    return exponential/np.sum(exponential, axis=1).reshape(-1,1)


# Basically doing prediction but named forward as its
# performing Forward-Propagation
def forward(X, w1, w2):
    h = sigmoid(np.matmul(data.prepend_bias(X), w1))
    y_hat = softmax(np.matmul(data.prepend_bias(h), w2))
    return y_hat


# Calling the predict() function
def classify(X, w1, w2):
    y_hat = forward(X, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


# Computing Loss over using logistic regression
def loss(Y, y_hat):
  return -np.sum(Y * np.log(y_hat)) / Y.shape[0]


# calculating gradient
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]


# Printing results to the terminal screen
def report(iteration, X_train, Y_train, X_test, Y_test, w1, w2):
    y_hat = forward(X_train, w1 ,w2)
    training_loss = loss(Y_train, y_hat)
    classifications = classify(X_test,w1 ,w2)
    accuracy = np.average(classifications == Y_test)*100.0
    print("Iteration: %5d, Loss: %.6f, Accuracy: %.2f%%" %
          (iteration, training_loss, accuracy))


# calling the training function for desired no. of iterations
def train(X_train, Y_train, X_test, Y_test, iterations, lr):
    w = np.zeros((X_train.shape[1], Y_train.shape[1]))
    for i in range(iterations):
        report(i, X_train, Y_train, X_test, Y_test, w)
        w -= gradient(X_train, Y_train, w) * lr
    report(iterations, X_train, Y_train, X_test, Y_test, w)
    return w


