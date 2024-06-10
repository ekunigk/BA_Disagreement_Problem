import torch
import numpy as np

def feature_agreement(ex_1, ex_2, k):
    indices_1 = set(top_features(ex_1, k).numpy())
    indices_2 = set(top_features(ex_2, k).numpy())
    count = len(indices_1.intersection(indices_2))
    return count / k


def top_features(ex, k):
    ex_abs = torch.abs(ex)
    values, indices = torch.topk(ex_abs, k, largest=True)
    return indices


def euclidean(ex_1, ex_2):
    return torch.cdist(ex_1, ex_2, p=2)


def fa_pairwise(explanations, k):
    fa_matrix = np.zeros((len(explanations), len(explanations)))
    for i in range(len(explanations)):
        for j in range(i+1, len(explanations)):
            fa_matrix[i, j] = feature_agreement(explanations[i], explanations[j], k)
            fa_matrix[j, i] = fa_matrix[i, j]
        fa_matrix[i, i] = 1
    return fa_matrix
    

def fa_average_pairwise(explanation_set, keys, feature_amount, n, k):
    explanation_keys = keys
    size = len(explanation_set[explanation_keys[0]])
    
    fa_matrix = np.zeros((len(explanation_keys), len(explanation_keys)))

    indices = np.random.randint(0, size, size=n)
    for i in indices:
        explanations = np.ones((1, feature_amount))
        for key in explanation_keys:
            explanations = np.vstack((explanations, explanation_set[key][i]))
        explanation_fa = torch.tensor(explanations[1:])
        fa_matrix = fa_matrix + fa_pairwise(explanation_fa, k)
    fa_matrix = fa_matrix / n 

    return fa_matrix 