import torch
import numpy as np

def count_feature_amount(explanation_set, with_label=True):
    if with_label:
        feature_count = torch.zeros(len(explanation_set[0])-1)
        explanations = explanation_set[:, :-1].clone()
    else:
        feature_count = torch.zeros(len(explanation_set[0]))
        explanations = explanation_set.clone()
    
    for explanation in explanations:
        for i in range(len(explanation)):
            if explanation[i] != 0:
                feature_count[i] += 1
    
    return feature_count


def count_features_per_method(explanation_set, with_label=True):
    method_length = int(len(explanation_set)/5)
    method_feature_count = torch.zeros((5, len(explanation_set[0])-1))
    for i in range(5):
        explanation_range = explanation_set[i*method_length:(i+1)*method_length]
        method_feature_count[i] = count_feature_amount(explanation_range, with_label)

    return method_feature_count


def count_lime_features(method_explanation_set):
    lime_feature_count = torch.zeros(len(method_explanation_set[0]))
    ex_counter = 0
    index_list = []
    for explanation in method_explanation_set[:, :-1]:
        counter_temp = 0
        for i in range(len(explanation)):
            if explanation[i] != 0:
                counter_temp += 1
        if counter_temp == 0:
            index_list.append(ex_counter)
        lime_feature_count[counter_temp] += 1
        ex_counter += 1
    return lime_feature_count


def calculate_variance(explanation_set, exclude_zeros=False, all_methods=True):

    if all_methods:
        method_variance = {}
        method_length = int(len(explanation_set)/5)
        for i in range(5):
            if exclude_zeros:
                method_variance[i] = torch.var(explanation_set[i*method_length:(i+1)*method_length, :-1][explanation_set[i*method_length:(i+1)*method_length, :-1]!=0], dim=0)
            else:
                method_variance[i] = torch.var(explanation_set[i*method_length:(i+1)*method_length, :-1], dim=0)
    else: 
        if exclude_zeros:
            method_variance = torch.var(explanation_set[:, :-1][explanation_set[:, :-1] != 0], dim=0)
        else:
            method_variance = torch.var(explanation_set[:, :-1], dim=0)

    return method_variance




                


    


