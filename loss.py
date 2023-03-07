import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def cross_entropy(y_true, y_pred):
    return -np.mean(y_true*np.log(y_pred))

def cross_entropy_prime(y_true, y_pred):
    return -y_true/y_pred

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

def mae_prime(y_true, y_pred):
    return np.sign(y_true-y_pred)/y_true.size

def mbe(y_true, y_pred):
    return np.mean(y_true-y_pred)

def mbe_prime(y_true, y_pred):
    return 1/y_true.size

def svm(y_true, y_pred):
    return np.mean(np.maximum(0, 1-y_true*y_pred))

def svm_prime(y_true, y_pred):
    return -y_true*np.maximum(0, 1-y_true*y_pred)/y_true.size

