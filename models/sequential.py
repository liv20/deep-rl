import torch.nn as nn

def mlp(input_size, output_size, hidden_depth, hidden_size, activation=nn.ReLU):
    """
    Creates multi-layer perceptron. Output layer is linear.

    Parameters:
    input_size : int
     - size of input space
    output_size : int
     - size of output space
    hidden_depth : int
     - depth of hidden network
    hidden_size : int
     - size of hidden layers
    activation : nn
     - activation function in hidden layers (default: nn.ReLU)
    
    Returns:
    mlp : nn.Sequential
     - multilayer perceptron with specified dimensions
    """
    assert hidden_depth >= 0

    layer_sizes = [input_size] + [hidden_size for _ \
         in range(hidden_depth)] + [output_size]

    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 1:
            layers.append(activation())
    return nn.Sequential(*layers)
