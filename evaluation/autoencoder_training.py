import torch 
from torch import nn
from torch import optim
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from evaluation.autoencoder import Autoencoder


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

        if epoch+1 % 10 == 0:
            print(f"epoch: {epoch}, loss: {loss.item():.4f}")


def create_batches(explanation1, explanation2, batch_size):

    length = len(explanation1)
    for i in range(0, length, batch_size):
        yield explanation1[i:min(i+batch_size, length)], explanation2[i:min(i+batch_size, length)]

        
def translate_with_autoencoder(autoencoder, explanation_set, non_zero_explanation_set, num_epochs=20, lr=0.001, batch_size=32):

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
                else:
                    explanation1 = explanation_set[i*ex_length:(i+1)*ex_length, :-1] 
                    explanation2 = explanation_set[j*ex_length:(j+1)*ex_length, :-1]
            
                X_train, X_test, y_train, y_test = train_test_split(explanation1, explanation2, test_size=0.2, random_state=46)
                train_autoencoder(autoencoder, X_train, y_train, num_epochs, lr, batch_size)
                # mse_temp = []
                # kf = KFold(n_splits=10, random_state=44, shuffle=True)
                # for train_idx, test_idx in kf.split(explanation1):

                #     X_train = torch.index_select(explanation1, 0, torch.tensor(train_idx))
                #     X_test = torch.index_select(explanation1, 0, torch.tensor(test_idx))
                #     y_train, y_test = explanation2[train_idx], explanation2[test_idx]
                
                #     train_autoencoder(autoencoder, X_train, y_train, num_epochs, lr, batch_size)
                #     autoencoder.eval()
                #     y_pred = autoencoder(X_test)
                #     mse_temp.append(mean_squared_error(y_test, y_pred.detach().numpy()))
                # mse[number_method[i] + '_' + number_method[j]] = np.mean(mse_temp)

                autoencoder.eval()
                y_pred = autoencoder(X_test)
                mse[number_method[i] + '_' + number_method[j]] = mean_squared_error(y_test, y_pred.detach().numpy())
        
    return mse


def test_ae_architectures(dc, list_of_layers):
    mse_dict = {}

    for key in list_of_layers:
        layers_encode = list_of_layers[key][0]
        layers_decode = list_of_layers[key][1]
        autoencoder = Autoencoder(layers_encode, layers_decode)
        mse = translate_with_autoencoder(autoencoder, dc.scaled_explanations, dc.non_zero_explanations, num_epochs=30, lr=0.001, batch_size=16)
        mse_dict[key] = mse

    return mse_dict
                
