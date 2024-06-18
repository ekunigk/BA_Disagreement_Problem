import numpy as np
import torch 

from data.data_collection import DataCollector
from data.evaluation_prep import collect_regression_data, prepare_umap_data

from evaluation.preliminary_evaluation import fa_average_pairwise
from evaluation.structural_regression import pairwise_kfold
from evaluation.umap import visualize_umap, project_umap

from visualization.preliminary_fig import visualize_fa
from visualization.regression_fig import visualize_scores

def analyze_explanations(explanation_set, model_number, n_fa, k_fa, n_neighbors=15, min_dist=0.1):

    data_collector = DataCollector()
    explanations = data_collector.collect_data(explanation_set)
    keys = data_collector.get_keys(explanations, model_number)

    fa_matrix = fa_average_pairwise(explanations, keys, n_fa, k_fa)
    visualize_fa(fa_matrix)

    umap_data = prepare_umap_data(explanations, keys)
    embedding = project_umap(umap_data, n_neighbors=n_neighbors, min_dist=min_dist)
    visualize_umap(umap_data, embedding)

    scores = pairwise_kfold(data_collector, explanations, model_number)
    visualize_scores(scores)



