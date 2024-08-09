import torch
import numpy as np

def feature_agreement(ex_1, ex_2, k):
    indices_1 = set(top_features(ex_1, k).numpy())
    indices_2 = set(top_features(ex_2, k).numpy())
    count = len(indices_1.intersection(indices_2))
    return count / k


def top_features(ex, k):
    ex_abs = torch.abs(ex)
    values, indices = torch.topk(ex, k, largest=True)
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
    

def fa_average_pairwise2(dc, n, k, model_number=1):
    keys = dc.get_keys(model_number)
    explanation_set = dc.explanation_set
    size = len(explanation_set[keys[0]])

    feature_amount = len(explanation_set[keys[0]][0])
    
    fa_matrix = np.zeros((len(keys), len(keys)))

    indices = np.random.randint(0, size, size=n)
    
    for i in indices:
        explanations = np.ones((1, feature_amount))
        for key in keys:
            explanations = np.vstack((explanations, explanation_set[key][i]))
        explanation_fa = torch.tensor(explanations[1:])
        fa_matrix = fa_matrix + fa_pairwise(explanation_fa, k)
    fa_matrix = fa_matrix / n 

    return fa_matrix 


def fa_average_pairwise(dc, n, k, model_number=1):
    explanation_set = dc.scaled_explanations
    size = int(len(explanation_set)/5)
    feature_amount = len(explanation_set[0]) -1

    fa_matrix = np.zeros((5, 5))

    for i in range(size):
        explanations = np.ones((1, feature_amount))
        for j in range(5):
            explanations = np.vstack((explanations, explanation_set[j*size+i, :-1]))
        explanation_fa = torch.tensor(explanations[1:])
        fa_matrix = fa_matrix + fa_pairwise(explanation_fa, k)
    fa_matrix = fa_matrix / size

    return fa_matrix


def fa_difference(explanation_set, keys, n, start=0, stop=4, steps=1, random_comparison=False):
    fa_diff = np.ones((1, 10))
    k_list = []
    for k in range(start, stop, steps):
        k_list.append(k)
        fa_matrix = fa_average_pairwise(explanation_set, keys, n, k)
        fa_diff = np.vstack((fa_diff, [fa_matrix[[0, 0, 0, 0, 1, 1, 1, 2, 2, 3], [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]]]))

    if random_comparison:
        random_results = [1]
        for i in k_list:
            first_set = np.random.randint(0, len(explanation_set[keys[0]][0]), size=i)
            second_set = np.random.randint(0, len(explanation_set[keys[0]][0]), size=i)
            count = len(set(first_set).intersection(set(second_set)))
            random_results = np.append(random_results, (count/i))
        fa_diff = np.hstack((fa_diff, random_results.reshape(-1, 1)))

    return fa_diff[1:], k_list


# def fa_model_comparison(explanation_set, k, model_amount=3):
#     explanation_length = int(len(explanation_set)/model_amount)
#     fa_matrix = np.zeros((model_amount, model_amount))

        

