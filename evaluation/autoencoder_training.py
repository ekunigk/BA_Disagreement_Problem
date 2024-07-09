import torch 
from torch import nn
from torch import optim
import numpy as np


def train_autoencoder(autoencoder, explanation1, explanation2, num_epochs=10, lr=0.001, batch_size=32):

    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # for epoch in range(num_epochs):
        


