import numpy as np
import torch 

from data_management.data_collector import DataCollector

from evaluation.preliminary_evaluation import fa_average_pairwise
from evaluation.structural_regression import pairwise_kfold
from evaluation.umap import visualize_umap, project_umap

from visualization.preliminary_fig import visualize_fa
from visualization.regression_fig import visualize_scores_temp

"""
collective method for parts of the preliminary analysis
"""

def analyze_explanations(explanation_set, n_fa, k_fa, model_number=1, n_neighbors=15, min_dist=0.1, scaled=False, masked=False, k_mask=3, mask=0):

    data_collector = DataCollector(explanation_set, model_number)
    keys = data_collector.get_keys(model_number)

    fa_matrix = fa_average_pairwise(data_collector, n_fa, k_fa, model_number)
    visualize_fa(fa_matrix)

    if masked and scaled:
        data_collector.mask_features(k_mask, mask, scaled=True)
        umap_data = data_collector.masked_explanations
        regression_data = data_collector.masked_explanations
        non_zero_data = data_collector.non_zero_masked_explanations
    elif masked and not scaled:
        data_collector.mask_features(k_mask, mask, scaled=False)
        umap_data = data_collector.masked_explanations
        regression_data = data_collector.masked_explanations
        non_zero_data = data_collector.non_zero_masked_explanations
    elif scaled and not masked:
        umap_data = data_collector.scaled_explanations
        regression_data = data_collector.scaled_explanations
        non_zero_data = data_collector.non_zero_explanations
    else:
        umap_data = data_collector.explanations_all
        regression_data = data_collector.explanations_all
        non_zero_data = data_collector.non_zero_explanations

    embedding, method_length, non_zero_method_length = project_umap(umap_data, non_zero_data, n_neighbors=n_neighbors, min_dist=min_dist)
    visualize_umap(embedding, method_length, non_zero_method_length)

    scores = pairwise_kfold(regression_data, non_zero_data)
    visualize_scores_temp(scores)



