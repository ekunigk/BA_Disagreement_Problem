import torch 
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression

# logistic regression classification to see if methods are separable

def split_regression_data(two_explanation_set, test_size=0.2):
    X = two_explanation_set[:, :-1]
    y = two_explanation_set[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=44)

    return X_train, X_test, y_train, y_test


def test_on_kfold(model, two_explanation_set, k=10, random_state=44):
    X = two_explanation_set[:, :-1]
    y = two_explanation_set[:, -1]

    model = LogisticRegression(random_state=10, max_iter=100)

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    kf.get_n_splits(X)

    scores = []

    for train_idx, test_idx in kf.split(X):

        X_train = torch.index_select(X, 0, torch.tensor(train_idx))
        X_test = torch.index_select(X, 0, torch.tensor(test_idx))
        y_train, y_test = y[train_idx], y[test_idx]

        # accuracy berechnen
        
        clf = model.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        scores.append(score)

    variance = np.var(scores)

    coefficients = clf.coef_

    return scores, coefficients


def concatenate_data_explanation(data_collector, two_explanation_set, dataset_name):
    X, y = data_collector.collect_original_dataset()

    X_doubled = torch.cat((X, X))
    X_cat = torch.cat((two_explanation_set[:, :-1], X_doubled), 1)
    y_cat = two_explanation_set[:, -1]

    return X_cat, y_cat

# prepare classification data

def get_pairwise_explanations(explanations_all, non_zero_explanations):
    pairs = {}
    method_list = ['IG', 'KS', 'LI', 'SG', 'VG']

    for i in range(4):
        method1 = method_list[0]
        for j in range(len(method_list)-1):
            method2 = method_list[j+1]
            dataset = separate_into_pairs(explanations_all, non_zero_explanations, method1, method2)
            pairs[f'{method1}_{method2}'] = dataset
        method_list.pop(0)

    return pairs


def separate_into_pairs(explanations_all, non_zero_explanations, method1, method2):

    explanation_length = int(len(explanations_all)/5)
    non_zero_length = int(len(non_zero_explanations)/5)

    dataset = torch.ones((1, len(explanations_all[0])))

    if method1 == 'LI' or method2 == 'LI':
        explanations = non_zero_explanations
        ex_length = non_zero_length
    else:
        explanations = explanations_all
        ex_length = explanation_length

    for i in range(2):
        method = [method1, method2][i]

        if method == 'IG':
            dataset = torch.vstack((dataset, explanations[0:ex_length]))
        elif method == 'KS':
            dataset = torch.vstack((dataset, explanations[ex_length:ex_length*2]))
        elif method == 'LI':
            dataset = torch.vstack((dataset, explanations[ex_length*2:ex_length*3]))
        elif method == 'SG':
            dataset = torch.vstack((dataset, explanations[ex_length*3:ex_length*4]))
        elif method == 'VG':
            dataset = torch.vstack((dataset, explanations[ex_length*4:ex_length*5]))

    dataset = dataset[1:]
    return dataset 
    
# method employed in preliminary analysis

def pairwise_kfold(explanations_all, non_zero_explanations, k=10, random_state=44):
    
    model = LogisticRegression(random_state=10, max_iter=100)
    pairs = get_pairwise_explanations(explanations_all, non_zero_explanations)

    scores = {}
    variances = {}
    coefficients = {}

    for pair in pairs:
        two_explanation_set = pairs[pair]
        score, coefs = test_on_kfold(model, two_explanation_set, k, random_state=random_state)
        scores[pair] = score
        # variances[pair] = variance
        coefficients[pair] = coefs


    return scores, coefficients



