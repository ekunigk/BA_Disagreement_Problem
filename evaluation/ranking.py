import numpy as np
import torch


def create_rankings(dict_scores):
    scores_sorted = {k : v for k, v in sorted(dict_scores.items(), key=lambda item: item[1]) }
    rankings = {k : i+1 for i, k in enumerate(scores_sorted.keys())}
    return rankings


def merge_dictionaries(dict1, dict2, dict3):
    merged_dict = {}
    for key in dict1.keys():
        merged_dict[key] = [dict1[key], dict2[key], dict3[key]]

    return merged_dict


def merge_rankings(dict_of_dicts):
    ranking_dict = {}
    
    for key in list(dict_of_dicts.keys()):
        ranking_dict[key] = create_rankings(dict_of_dicts[key])

    merged_dict = {}
    for key in ranking_dict[list(ranking_dict.keys())[0]].keys():
        merged_dict[key] = [ranking_dict[dict_key][key] for dict_key in ranking_dict.keys()]

    return merged_dict


def separate_concepts(merged_dict):
    grad_pairs = ['IG_VG', 'IG_SG', 'SG_VG', 'SG_IG', 'VG_IG', 'VG_SG']
    perturb_pairs = ['KS_LI', 'LI_KS']
    grad_dict = {}
    perturb_dict = {}
    mixed_dict = {}

    for key in merged_dict.keys():
        if key in grad_pairs:
            grad_dict[key] = merged_dict[key]
        elif key in perturb_pairs:
            perturb_dict[key] = merged_dict[key]
        else:
            mixed_dict[key] = merged_dict[key]

    return grad_dict, perturb_dict, mixed_dict


def sum_mses(mse_dict):
    sum_dict = {}
    for key in mse_dict.keys():
        sum_temp = 0
        for key2 in mse_dict[key].keys():
            sum_temp += mse_dict[key][key2]
        sum_dict[key] = sum_temp
    
    return sum_dict


def find_best_architecture(mse_dict):
    best_dict = {}
    key_list = list(mse_dict.keys())
    for key in mse_dict[key_list[0]].keys():
        temp_mse = 10000
        for key1 in key_list:
            if mse_dict[key1][key] < temp_mse:
                temp_mse = mse_dict[key1][key]
                best_dict[key] = key1
    
    return best_dict


def merge_two_dicts(x, y):
    z = {**x, **y}
    return z
