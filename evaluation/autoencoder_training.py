import torch 
from torch import nn
from torch import optim
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from evaluation.autoencoder import Autoencoder
import matplotlib.pyplot as plt

from evaluation.linear_translator import calculate_masked_mse


def train_autoencoder(autoencoder, explanation1, explanation2, num_epochs=10, lr=0.001, batch_size=32):

    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_x, batch_y in create_batches(explanation1, explanation2, batch_size):
            outputs = autoencoder(batch_x)
            loss = loss_fn(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if epoch+1 % 10 == 0:
        # print(f"epoch: {epoch}, loss: {loss.item():.4f}")


def train_evaluate_autoencoder(autoencoder, ex1, ex2, num_epochs=10, lr=0.001, batch_size=32):

    ex1_train, ex1_val, ex2_train, ex2_val = train_test_split(ex1, ex2, test_size=0.2, random_state=49)

    train_losses = []
    val_losses = []
    
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0
        for batch_x, batch_y in create_batches(ex1_train, ex2_train, batch_size):
            outputs = autoencoder(batch_x)
            loss = loss_fn(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(ex1_train)*batch_size)

        autoencoder.eval()
        val_loss = 0.0
        for batch_x, batch_y in create_batches(ex1_val, ex2_val, batch_size):
            outputs = autoencoder(batch_x)
            loss = loss_fn(outputs, batch_y)
            val_loss += loss.item()

        val_losses.append(val_loss / len(ex1_val)*batch_size)

        print(f"epoch: {epoch}, train loss: {train_losses[-1]:.4f}, val loss: {val_losses[-1]:.4f}")

    plt.figure(figsize=(12, 4))

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    plt.show()



def create_batches(explanation1, explanation2, batch_size):

    length = len(explanation1)
    for i in range(0, length, batch_size):
        yield explanation1[i:min(i+batch_size, length)], explanation2[i:min(i+batch_size, length)]

        
def translate_with_autoencoder(autoencoder, explanation_set, non_zero_explanation_set, num_epochs=20, lr=0.001, batch_size=32, masked=False, masked_indices=None, non_zero_masked_indices=None):

    number_method = {0: 'IG', 1: 'KS', 2: 'LI', 3: 'SG', 4: 'VG'}
    ex_length = int(len(explanation_set) / 5)
    ex_length_non_zero = int(len(non_zero_explanation_set) / 5)

    mse = {}
    mse_kfold_values = {}

    for i in range(5):
        for j in range(5):
            if i != j: 
                if i == 2 or j == 2:
                    explanation1 = non_zero_explanation_set[i*ex_length_non_zero:(i+1)*ex_length_non_zero, :-1] 
                    explanation2 = non_zero_explanation_set[j*ex_length_non_zero:(j+1)*ex_length_non_zero, :-1] 
                    if masked:
                        masked_ind = non_zero_masked_indices[j*ex_length_non_zero:(j+1)*ex_length_non_zero]
                else:
                    explanation1 = explanation_set[i*ex_length:(i+1)*ex_length, :-1] 
                    explanation2 = explanation_set[j*ex_length:(j+1)*ex_length, :-1]
                    if masked:
                        masked_ind = masked_indices[j*ex_length:(j+1)*ex_length]
            
                # X_train, X_test, y_train, y_test = train_test_split(explanation1, explanation2, test_size=0.2, random_state=49)
                # train_autoencoder(autoencoder, X_train, y_train, num_epochs, lr, batch_size)
                mse_temp = []
                kf = KFold(n_splits=10, random_state=44, shuffle=True)
                for train_idx, test_idx in kf.split(explanation1):

                    X_train = torch.index_select(explanation1, 0, torch.tensor(train_idx))
                    X_test = torch.index_select(explanation1, 0, torch.tensor(test_idx))
                    y_train, y_test = explanation2[train_idx], explanation2[test_idx]
                
                    train_autoencoder(autoencoder, X_train, y_train, num_epochs, lr, batch_size)
                    autoencoder.eval()
                    y_pred = autoencoder(X_test)
                    if masked:
                        mse_temp.append(calculate_masked_mse(masked_ind[test_idx], y_pred.detach().numpy(), y_test))
                    else:
                        mse_temp.append(mean_squared_error(y_test, y_pred.detach().numpy()))
                mse[number_method[i] + '_' + number_method[j]] = np.mean(mse_temp)
                mse_kfold_values[number_method[i] + '_' + number_method[j]] = mse_temp

                autoencoder.eval()
                y_pred = autoencoder(X_test)
                # mse[number_method[i] + '_' + number_method[j]] = mean_squared_error(y_test, y_pred.detach().numpy())
        
    return mse, mse_kfold_values


def translate_with_ae_simple(autoencoder, explanation_set, non_zero_explanation_set, num_epochs=20, lr=0.001, batch_size=32, masked=False, masked_indices=None, non_zero_masked_indices=None):

    number_method = {0: 'IG', 1: 'KS', 2: 'LI', 3: 'SG', 4: 'VG'}
    ex_length = int(len(explanation_set) / 5)
    ex_length_non_zero = int(len(non_zero_explanation_set) / 5)

    mse = {}  

    for i in range(5):
        for j in range(5):
            if i != j: 
                if i == 2 or j == 2:
                    explanation1 = non_zero_explanation_set[i*ex_length_non_zero:(i+1)*ex_length_non_zero, :-1] 
                    explanation2 = non_zero_explanation_set[j*ex_length_non_zero:(j+1)*ex_length_non_zero, :-1] 
                    if masked:
                        masked_ind = non_zero_masked_indices[j*ex_length_non_zero:(j+1)*ex_length_non_zero]
                else:
                    explanation1 = explanation_set[i*ex_length:(i+1)*ex_length, :-1] 
                    explanation2 = explanation_set[j*ex_length:(j+1)*ex_length, :-1]
                    if masked:
                        masked_ind = masked_indices[j*ex_length:(j+1)*ex_length]

                X_train, X_test, y_train, y_test = train_test_split(explanation1, explanation2, test_size=0.2, random_state=33)
                train_autoencoder(autoencoder, X_train, y_train, num_epochs, lr, batch_size)

                autoencoder.eval()
                y_pred = autoencoder(X_test)
                # if masked:
                    # mse[number_method[i]+'_'+number_method[j]] = calculate_masked_mse(masked_ind[test_idx], y_pred.detach().numpy(), y_test)
                # else:
                mse[number_method[i]+'_'+number_method[j]] = mean_squared_error(y_test, y_pred.detach().numpy())
        
    return mse, 0

def test_ae_architectures(dc, list_of_layers):
    mse_dict = {}

    for key in list_of_layers:
        layers_encode = list_of_layers[key][0]
        layers_decode = list_of_layers[key][1]
        autoencoder = Autoencoder(layers_encode, layers_decode)
        mse, kfold = translate_with_ae_simple(autoencoder, dc.scaled_explanations, dc.non_zero_explanations, num_epochs=20, lr=0.01, batch_size=16)
        mse_dict[key] = mse
        print(f"Autoencoder with {key} mse: {mse}")

    return mse_dict
                
