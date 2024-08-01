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
    

def translate_pairwise(explanation_set, non_zero_explanation_set, pred=True, kfold=True, masked=False, masked_indices=None, non_zero_masked_indices=None):
    length_explanations = int(len(explanation_set) / 5)
    r2 = {}
    mse = {}
    mean_mse = {}
    variances = {}
    number_method = {0: 'IG', 1: 'KS', 2: 'LI', 3: 'SG', 4: 'VG'}
    ex_masked_indices = None
    for i in range(5):
        for j in range(5):
            if i != j:
                if i == 2 or j == 2:
                    ex_length = int(len(non_zero_explanation_set) / 5)
                    explanation1 = non_zero_explanation_set[i*ex_length:(i+1)*ex_length, :-1] # .numpy()
                    explanation2 = non_zero_explanation_set[j*ex_length:(j+1)*ex_length, :-1] # .numpy()
                    if masked:
                        ex_masked_indices = non_zero_masked_indices[j*ex_length:(j+1)*ex_length]
                else:
                    explanation1 = explanation_set[i*length_explanations:(i+1)*length_explanations, :-1] # .numpy()
                    explanation2 = explanation_set[j*length_explanations:(j+1)*length_explanations, :-1] # .numpy()
                    if masked:
                        ex_masked_indices = masked_indices[j*length_explanations:(j+1)*length_explanations]
                if kfold:
                    mse_score, r2_score, variance = translate_kfold(explanation1, explanation2, pred=pred, masked=masked, masked_indices=ex_masked_indices)
                else:
                    mse_score, r2_score = translate(explanation1, explanation2, pred, masked=masked)
                    variance = 0
                r2[number_method[i] + '_' + number_method[j]] = r2_score
                mse[number_method[i] + '_' + number_method[j]] = mse_score
                mean_mse[number_method[i] + '_' + number_method[j]] = compare_to_mean_baseline(explanation2)
                variances[number_method[i] + '_' + number_method[j]] = variance
    return r2, mse, mean_mse, variances


def translate_kfold(explanation1, explanation2, k=10, random_state=44, pred=True, masked=False, masked_indices=None):
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
            # print(y_pred[:10])
            # print(f'prediction: {y_pred[0]}, actual: {y_test[0]}')
            if masked:
                mse = calculate_masked_mse(masked_indices[test_idx], y_pred, y_test)
            else:
                mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)
            
        scores.append(score)
    if pred:
        variance = np.var(mse_scores)

    return np.mean(mse_scores), np.mean(scores), variance


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


def create_index_matrix(masked_indices, number_of_features):
    index_matrix = np.ones((len(masked_indices), number_of_features))
    for i, indices in enumerate(masked_indices):
        for index in indices:
            index = int(index)
            index_matrix[i][index] = 0
    
    return index_matrix


def calculate_masked_mse(masked_indices, y_pred, y_true):
    number_of_features = len(y_pred[0])
    index_matrix = create_index_matrix(masked_indices, number_of_features)
    # print(y_pred[:10])

    masked_y_pred = np.multiply(y_pred, index_matrix)
    masked_mse = mean_squared_error(y_true, masked_y_pred)


    return masked_mse


def calculate_distance_to_baseline(mse_scores, mse_baseline):
    distances = {}
    for key, value in mse_scores.items():
        distances[key] = mse_baseline[key] - value
    return distances


def calculate_percentage_of_baseline(mse_scores, mse_baseline):
    percentages = {}
    for key, value in mse_scores.items():
        percentages[key] = mse_baseline[key] / value
    return percentages
        

