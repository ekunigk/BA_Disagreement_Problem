import numpy as np
import torch
import pandas as pd


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
        print(key)
        ranking_dict[key] = create_rankings(dict_of_dicts[key])

    merged_dict = {}
    for key in ranking_dict[list(ranking_dict.keys())[0]].keys():
        merged_dict[key] = [ranking_dict[dict_key][key] for dict_key in ranking_dict.keys()]

    return merged_dict

def create_rankings_kfold(dict_scores):
    rankings = {}
    for i in range(10):
        scores_sorted = {k : v for k, v in sorted(dict_scores[i].items(), key=lambda item: item[1]) }
        rankings[i] = {k : i+1 for i, k in enumerate(scores_sorted.keys())}
    return rankings


def normalize_df(df, baseline):
    # print(df.head())
    for col in df.columns:
        for idx in df.index:
            df[col][idx] = df[col][idx] / baseline[col][idx]

    return df


def rank_entries(df, baseline=0):

    if baseline != 0:
        df[:] = normalize_df(df, baseline)

    ranks_array = np.zeros((20, 3, 10))
    
    for col_idx, col in enumerate(df.columns):

        col_data = np.array(df[col].tolist())

        ranks = np.argsort(np.argsort(col_data, axis=0), axis=0) + 1  

        ranks_array[:, col_idx, :] = ranks

    rank_df = pd.DataFrame([list(row) for row in ranks_array], columns=df.columns, index=df.index)
    
    return rank_df


def separate_concepts_df(merged_df):

    grad_pairs = ['IG_VG', 'IG_SG', 'SG_VG', 'SG_IG', 'VG_IG', 'VG_SG']
    perturb_pairs = ['KS_LI', 'LI_KS']
    mixed_pairs = ['IG_KS', 'IG_LI', 'KS_IG', 'KS_VG', 'LI_IG', 'LI_SG', 'SG_LI', 'SG_KS', 'VG_KS', 'VG_LI']

    grad_df = pd.DataFrame(index=grad_pairs, columns=merged_df.columns)
    perturb_df = pd.DataFrame(index=perturb_pairs, columns=merged_df.columns)
    mixed_df = pd.DataFrame(index=mixed_pairs, columns=merged_df.columns)

    for idx in merged_df.index:
        if idx in grad_pairs:
            grad_df[idx] = merged_df.loc[idx]
        elif idx in perturb_pairs:
            perturb_df[idx] = merged_df.loc[idx]
        else:
            mixed_df[idx] = merged_df.loc[idx]

    # grad_df = merged_df.loc(grad_pairs)
    # perturb_df = merged_df.loc(perturb_pairs)
    # mixed_df = merged_df.drop(grad_pairs + perturb_pairs, axis=1)

    return grad_df, perturb_df, mixed_df



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

def separate_kfold(merged_dict):
    grad_dict = {}
    perturb_dict = {}
    mixed_dict = {}
    for key in merged_dict.keys():
        grad, perturb, mixed = separate_concepts(merged_dict[key])
        grad_dict[key] = grad
        perturb_dict[key] = perturb
        mixed_dict[key] = mixed

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
