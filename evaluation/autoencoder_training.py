import torch 
from torch import nn
from torch import optim
import numpy as np


def train_autoencoder(autoencoder, explanation1, explanation2, num_epochs=20, lr=0.001, batch_size=32):

    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_x, batch_y in create_batches(explanation1, explanation2, batch_size):
            outputs = autoencoder(batch_x)
            loss = loss_fn(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch}, loss: {loss.item():.4f}")


def create_batches(explanation1, explanation2, batch_size):

    length = len(explanation1)
    for i in range(0, length, batch_size):
        yield explanation1[i:min(i+batch_size, length)], explanation2[i:min(i+batch_size, length)]

        


