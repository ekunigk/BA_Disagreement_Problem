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


def calculate_variance2(explanation_set, exclude_zeros=False, all_methods=True):

    if all_methods:
        method_variance = {}
        method_length = int(len(explanation_set)/5)
        for i in range(5):
            if exclude_zeros:
                var_set = explanation_set[i*method_length:(i+1)*method_length, :-1][explanation_set[i*method_length:(i+1)*method_length, :-1]!=0]
                print(var_set[0])
                method_variance[i] = torch.var(explanation_set[i*method_length:(i+1)*method_length, :-1][explanation_set[i*method_length:(i+1)*method_length, :-1]!=0], dim=0)
            else:
                method_variance[i] = torch.var(explanation_set[i*method_length:(i+1)*method_length, :-1], dim=0)
    else: 
        if exclude_zeros:
            method_variance = torch.var(explanation_set[:, :-1][explanation_set[:, :-1] != 0], dim=0)
        else:
            method_variance = torch.var(explanation_set[:, :-1], dim=0)

    return method_variance


def do_dimensional_analysis(dc):
    number_of_features = len(dc.scaled_explanations[0])-1
    method_size = int(len(dc.scaled_explanations)/5)
    non_zero_method_size = int(len(dc.non_zero_explanations)/5)

    explanation_dict = {'ig': dc.scaled_explanations[:method_size,:-1],
                        'ks': dc.scaled_explanations[method_size:(2*method_size),:-1],
                        'li': dc.non_zero_explanations[(2*non_zero_method_size):(3*non_zero_method_size), :-1],
                        'sg': dc.scaled_explanations[(3*method_size):(4*method_size), :-1],
                        'vg': dc.scaled_explanations[(4*method_size):, :-1]}
    
    dimensional_variance = torch.zeros((5, number_of_features))
    dimensional_mean = torch.zeros((5, number_of_features))

    for i, key in enumerate(explanation_dict):
        print(explanation_dict[key].shape)
        dimensional_variance[i] = torch.var(explanation_dict[key], dim=0)
        dimensional_mean[i] = torch.mean(explanation_dict[key], dim=0)

    explanation_variance = torch.zeros((method_size, 5))
    explanation_mean = torch.zeros((method_size, 5))
    top_features = torch.zeros((5, number_of_features))


    for i, key in enumerate(explanation_dict):
        if key == 'li':
            method_size_new = non_zero_method_size
        else:
            method_size_new = method_size
        for j in range(method_size_new):
            explanation_variance[j, i] = torch.var(explanation_dict[key][j])
            explanation_mean[j, i] = torch.mean(explanation_dict[key][j])
            values, top_feat_temp = torch.topk(explanation_dict[key][j], 2, largest=True)
            for k in top_feat_temp:
                top_features[i][top_feat_temp] += 1

    explanation_variance_per_method = torch.mean(explanation_variance, dim=0)


    return dimensional_variance, dimensional_mean, explanation_variance, explanation_mean, explanation_variance_per_method, top_features





                


    


