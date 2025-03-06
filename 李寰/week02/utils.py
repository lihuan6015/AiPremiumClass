import numpy as np

def forward(X_train, theta, bias):
    z = np.dot(theta, X_train.T) + bias
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat