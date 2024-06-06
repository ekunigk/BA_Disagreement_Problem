import torch 
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

# from data.evaluation_prep import collect_regression_data, collect_original_dataset


def split_regression_data(two_explanation_set, test_size=0.2):
    X = two_explanation_set[:, :-1]
    y = two_explanation_set[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=44)

    return X_train, X_test, y_train, y_test


def test_on_kfold(model, two_explanation_set, k=10):
    X = two_explanation_set[:, :-1]
    y = two_explanation_set[:, -1]

    kf = KFold(n_splits=k, shuffle=True, random_state=44)
    kf.get_n_splits(X)

    scores = []

    for train_idx, test_idx in kf.split(X):
        # X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        # y_train, y_test = y[train_idx], y[test_idx]

        X_train = torch.index_select(X, 0, torch.tensor(train_idx))
        X_test = torch.index_select(X, 0, torch.tensor(test_idx))
        y_train, y_test = y[train_idx], y[test_idx]

        clf = model.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        scores.append(score)

    # print("all scores: ", scores)
    # print("mean score: ", np.mean(scores))

    return np.mean(scores)


def concatenate_data_explanation(data_collector, two_explanation_set, dataset_name):
    X, y = data_collector.collect_original_dataset(dataset_name)

    X_doubled = torch.cat((X, X))
    X_cat = torch.cat((two_explanation_set[:, :-1], X_doubled), 1)
    y_cat = two_explanation_set[:, -1]

    return X_cat, y_cat


def get_pairwise_explanations(data_collector, explanation_set, model_number=1):
    pairs = {}
    method_list = ['ig', 'ks', 'li', 'sg', 'vg']

    for i in range(4):
        method1 = method_list[0]
        for j in range(len(method_list)-1):
            method2 = method_list[j+1]
            dataset = data_collector.collect_regression_data(explanation_set, method1, method2, model_number)
            pairs[f'{method1}_{method2}'] = dataset
        method_list.pop(0)

    # print(list(pairs.keys()))
    return pairs 


def pairwise_kfold(data_collector, explanation_set, model_number=1, k=10):
    pairs = get_pairwise_explanations(data_collector, explanation_set, model_number)
    model = LogisticRegression(random_state=10, max_iter=100)
    scores = {}

    for pair in pairs:
        # print(pair)
        two_explanation_set = pairs[pair]
        score = test_on_kfold(model, two_explanation_set, k)
        scores[pair] = score

    return scores


def multiple_pairwise_kfold(data_collector, explanation_set, model_number=1, k=10, n=3):
    compared_scores = np.array([])

    for i in range(n):
        scores = pairwise_kfold(data_collector, explanation_set, model_number, k)
        for score in scores.values():
            compared_scores = np.append(compared_scores, score)
    
    compared_scores = compared_scores.reshape(n, -1)

    df = pd.DataFrame(compared_scores, columns=list(scores.keys()))

    # compared_scores = {}
    # for i in range(n):
    #     scores = pairwise_kfold(data_collector, explanation_set, model_number, k)
    #     compared_scores[i] = scores

    return df



