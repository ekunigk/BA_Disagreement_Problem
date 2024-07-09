from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def translate(explanation1, explanation2, pred=False):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(explanation1, explanation2, test_size=0.2, random_state=43)
    X_train, X_test, y_train, y_test = X_train.numpy(), X_test.numpy(), y_train.numpy(), y_test.numpy()
    model.fit(X_train, y_train)
    B = model.coef_
    y_pred = model.predict(X_test)
    scores = model.score(X_test, y_test)
    # analyze_residuals(y_test, y_pred)
    if pred:
        mse = mean_squared_error(y_test, y_pred)
        return mse, scores
    else:
        return 0, scores
    

def translate_pairwise(explanation_set, non_zero_explanation_set, pred=True, kfold=True):
    length_explanations = int(len(explanation_set) / 5)
    r2 = {}
    mse = {}
    mean_mse = {}
    number_method = {0: 'IG', 1: 'KS', 2: 'LI', 3: 'SG', 4: 'VG'}
    for i in range(5):
        for j in range(5):
            if i != j:
                if i == 2 or j == 2:
                    ex_length = int(len(non_zero_explanation_set) / 5)
                    explanation1 = non_zero_explanation_set[i*ex_length:(i+1)*ex_length, :-1] # .numpy()
                    explanation2 = non_zero_explanation_set[j*ex_length:(j+1)*ex_length, :-1] # .numpy()
                else:
                    explanation1 = explanation_set[i*length_explanations:(i+1)*length_explanations, :-1] # .numpy()
                    explanation2 = explanation_set[j*length_explanations:(j+1)*length_explanations, :-1] # .numpy()
                if kfold:
                    mse_score, r2_score = translate_kfold(explanation1, explanation2, pred=pred)
                else:
                    mse_score, r2_score = translate(explanation1, explanation2, pred)
                r2[number_method[i] + '_' + number_method[j]] = r2_score
                mse[number_method[i] + '_' + number_method[j]] = mse_score
                mean_mse[number_method[i] + '_' + number_method[j]] = compare_to_mean_baseline(explanation2)
    return r2, mse, mean_mse


def translate_kfold(explanation1, explanation2, k=10, random_state=44, pred=True):
    X = explanation1
    y = explanation2
    model = LinearRegression()
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    kf.get_n_splits(X)
    scores = []
    mse_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train = torch.index_select(X, 0, torch.tensor(train_idx)).numpy()
        X_test = torch.index_select(X, 0, torch.tensor(test_idx)).numpy()
        y_train, y_test = y[train_idx].numpy(), y[test_idx].numpy()

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        if pred:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)

        scores.append(score)

    return np.mean(mse_scores), np.mean(scores)


def compare_to_mean_baseline(explanation2):
    mean = np.mean(explanation2.numpy(), axis=0)
    mean_array = np.zeros(explanation2.shape)
    mean_array[:,:] = mean
    mean_mse = mean_squared_error(explanation2, mean_array)
    return mean_mse
 


def analyze_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    sns.histplot(residuals[:, 0])
    plt.title('Residuals')
    plt.show()

        

