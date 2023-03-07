import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x>0).astype(float)

def x_cube(x):
    return x**3

def x_cube_prime(x):
    return 3*x**2

def leaky_relu(x):
    return np.maximum(0.01*x, x)

def leaky_relu_prime(x):
    return (x>0).astype(float) + 0.01*(x<=0).astype(float)

def elu(x):
    return np.where(x>0, x, np.exp(x)-1)

def elu_prime(x):
    return np.where(x>0, 1, np.exp(x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_prime(x):
    return softmax(x)*(1-softmax(x))

def swish(x):
    return x*sigmoid(x)

def swish_prime(x):
    return sigmoid(x) + x*sigmoid_prime(x)

def linear(x):
    return x

def linear_prime(x):
    return 1


