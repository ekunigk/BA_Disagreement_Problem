import torch 
from torch import nn
from torch import optim
import numpy as np


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, hidden_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)
    

class Decoder(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.Tanh(),
            nn.Linear(8, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.decoder(z)
    

class Autoencoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.decoder(z)
        return y_pred
        