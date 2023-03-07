import tkinter as tk
import activation as Activation
import loss as Loss
from  example_xor import xor

def build_gui():
    # Créer la fenêtre principale
    root = tk.Tk()
    root.title("Neural Network Menu")
    root.geometry("300x300")

    # Créer une liste déroulante pour la fonction d'activation
    activation_var = tk.StringVar(value="tanh")
    activation_label = tk.Label(root, text="Activation function:")
    activation_label.pack()
    activation_dropdown = tk.OptionMenu(root, activation_var, "tanh", "relu", "sigmoid", "leaky_relu", "elu", "softmax", "swish", "linear")
    activation_dropdown.pack()

    # Créer une zone de saisie pour l'alpha
    alpha_var = tk.StringVar(value="0.1")
    alpha_label = tk.Label(root, text="Alpha:")
    alpha_label.pack()
    alpha_entry = tk.Entry(root, textvariable=alpha_var)
    alpha_entry.pack()

    # Créer une liste déroulante pour la fonction de perte
    loss_var = tk.StringVar(value="mse")
    loss_label = tk.Label(root, text="Loss function:")
    loss_label.pack()
    loss_dropdown = tk.OptionMenu(root, loss_var, "mse", "cross_entropy", "mae", "mbe", "svm")
    loss_dropdown.pack()

    # Créer un curseur pour les epochs
    epoch_var = tk.IntVar(value=0)
    epoch_label = tk.Label(root, text="Epochs:")
    epoch_label.pack()
    epoch_scale = tk.Scale(root, variable=epoch_var, from_=1, to=10000, resolution=10,orient=tk.HORIZONTAL)
    epoch_scale.pack()

    # Créer une liste déroulante pour l'exemple à lancer
    example_var = tk.StringVar(value="xor")
    example_label = tk.Label(root, text="Example:")
    example_label.pack()
    example_dropdown = tk.OptionMenu(root, example_var, "xor")
    example_dropdown.pack()

    # Créer un bouton pour lancer le réseau de neurones
    run_button = tk.Button(root, text="Run", command=lambda: run_network(activation_var.get(), float(alpha_var.get()), loss_var.get(), epoch_var.get(), example_var.get(), root))
    run_button.pack()

    # Afficher la fenêtre principale
    root.mainloop()

# Définir une fonction qui lance le réseau de neurones avec les paramètres sélectionnés
def run_network(activation, alpha, loss, epochs, example, root):

    activation_prime = None
    loss_prime = None

    # Définir la fonction de perte
    match loss:
        case "mse":
            loss = Loss.mse
            loss_prime = Loss.mse_prime
        case "cross_entropy":
            loss = Loss.cross_entropy
            loss_prime = Loss.cross_entropy_prime
        case "mae":
            loss = Loss.mae
            loss_prime = Loss.mae_prime
        case "mbe":
            loss = Loss.mbe
            loss_prime = Loss.mbe_prime
        case "svm":
            loss = Loss.svm
            loss_prime = Loss.svm_prime
        case _:
            raise ValueError("Invalid loss function")

    # Définir la fonction d'activation
    match activation:
        case "tanh":
            activation = Activation.tanh
            activation_prime = Activation.tanh_prime
        case "relu":
            activation = Activation.relu
            activation_prime = Activation.relu_prime
        case "sigmoid":
            activation = Activation.sigmoid
            activation_prime = Activation.sigmoid_prime
        case "leaky_relu":
            activation = Activation.leaky_relu
            activation_prime = Activation.leaky_relu_prime
        case "elu":
            activation = Activation.elu
            activation_prime = Activation.elu_prime
        case "softmax":
            activation = Activation.softmax
            activation_prime = Activation.softmax_prime
        case "swish":
            activation = Activation.swish
            activation_prime = Activation.swish_prime
        case "linear":
            activation = Activation.linear
            activation_prime = Activation.linear_prime
        case _:
            raise ValueError("Invalid activation function")
        

    # Définir l'exemple à lancer
    match example:
        case "xor":
            xor(epochs, alpha, activation, activation_prime, loss, loss_prime)
            root.destroy()
            build_gui()
        case _:
            raise ValueError("Invalid example")




build_gui()
    
