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





