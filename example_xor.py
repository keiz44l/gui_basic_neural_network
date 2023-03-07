import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer

def xor(epochs, learning_rate, activation, activation_prime, loss, loss_prime):
# training data
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # network
    net = Network()
    net.add(FCLayer(2, 3))
    net.add(ActivationLayer(activation, activation_prime))
    net.add(FCLayer(3, 1))
    net.add(ActivationLayer(activation, activation_prime))

    # train
    net.use(loss, loss_prime)
    net.fit(x_train, y_train, epochs, learning_rate)

    # test
    out = net.predict(x_train)
    print(out)
    x = int(input("Entrez une valeur de test: "))
    y = int(input("Entrez une valeur de test: "))
    out = net.predict(np.array([[[x,y]]]))
    out_string = str(out[0])
    out_string = out_string.replace("[", "").replace("]", "")
    out= float(out_string)
    if out > 0.5:
        out_string = "1"
    else:
        out_string = "0"
    print(out_string)
    
