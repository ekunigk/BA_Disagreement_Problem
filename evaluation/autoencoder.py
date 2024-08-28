import torch 
from torch import nn
from torch import optim
import numpy as np

"""
definition of the autoencoder used for translations
"""

class Encoder(nn.Module):

    def __init__(self, layers_encode):
        super(Encoder, self).__init__()
        activation_func = nn.Tanh()
        layer_list_encode = []
        for i in range(len(layers_encode)-1):
            layer_list_encode.append(nn.Linear(layers_encode[i], layers_encode[i+1]))
            layer_list_encode.append(activation_func)
        self.encoder = nn.Sequential(*layer_list_encode)
                      

    def forward(self, x):
        return self.encoder(x)
    

class Decoder(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.Tanh(),
            nn.Linear(16, output_dim),
            nn.Tanh()
        )


    def __init__(self, layers_decode):
        super(Decoder, self).__init__()
        activation_func = nn.Tanh()
        layer_list_decode = []
        for i in range(len(layers_decode)-1):
            layer_list_decode.append(nn.Linear(layers_decode[i], layers_decode[i+1]))
            layer_list_decode.append(activation_func)
        self.decoder = nn.Sequential(*layer_list_decode)


    def forward(self, z):
        return self.decoder(z)
    

class Autoencoder(nn.Module):
    
    def __init__(self, layers_encode, layers_decode):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(layers_encode)
        self.decoder = Decoder(layers_decode)
    
    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.decoder(z)
        return y_pred
        