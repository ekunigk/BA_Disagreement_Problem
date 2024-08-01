from data.data_collector import DataCollector
from evaluation.linear_translator import translate_pairwise
from evaluation.autoencoder import Autoencoder
from evaluation.autoencoder_training import translate_with_autoencoder
from visualization.translator_fig import visualize_multiple_scores, show_rankings, show_rankings_bp
from evaluation.ranking import merge_rankings, create_rankings, separate_concepts

import numpy as np

def evaluate_two_datasets(model_number1=1, model_number2=1, eval=True):

    dc_bw = DataCollector('breastw', model_number=model_number1)
    dc_sb = DataCollector('spambase', model_number=model_number2)

    bw_r2, bw_mse, bw_mse_baseline, bw_var = translate_pairwise(dc_bw.scaled_explanations, dc_bw.non_zero_explanations)
    masked_indices6, masked_indices6_nonzero = dc_bw.mask_features(k=6, scaled=True)
    bw_r2_m6, bw_mse_m6, bw_mse_baseline_m6, bw_var_m6 = translate_pairwise(dc_bw.masked_explanations, dc_bw.non_zero_masked_explanations, masked=True, masked_indices=masked_indices6, non_zero_masked_indices=masked_indices6_nonzero)
    masked_indices3, masked_indices3_nonzero = dc_bw.mask_features(k=3, scaled=True)
    bw_r2_m3, bw_mse_m3, bw_mse_baseline_m3, bw_var_m3 = translate_pairwise(dc_bw.masked_explanations, dc_bw.non_zero_masked_explanations, masked=True, masked_indices=masked_indices3, non_zero_masked_indices=masked_indices3_nonzero)

    sb_r2, sb_mse, sb_mse_baseline, sb_var = translate_pairwise(dc_sb.scaled_explanations, dc_sb.non_zero_explanations)
    masked_indices38, masked_indices38_nonzero = dc_sb.mask_features(k=38, scaled=True)
    sb_r2_m38, sb_mse_m38, sb_mse_baseline_m338, sb_var_m38 = translate_pairwise(dc_sb.masked_explanations, dc_sb.non_zero_masked_explanations, masked=True, masked_indices=masked_indices38, non_zero_masked_indices=masked_indices38_nonzero)
    masked_indices19, masked_indices19_nonzero = dc_sb.mask_features(k=19, scaled=True)
    sb_r2_m19, sb_mse_m19, sb_mse_baseline_m19, sb_var_m19 = translate_pairwise(dc_sb.masked_explanations, dc_sb.non_zero_masked_explanations, masked=True, masked_indices=masked_indices19, non_zero_masked_indices=masked_indices19_nonzero)

    score_dict = {'bw_mse': bw_mse, 'bw_mse 6' : bw_mse_m6, 'bw_mse 3': bw_mse_m3, 'sb_mse': sb_mse, 'sb_mse 38': sb_mse_m38, 'sb_mse 19': sb_mse_m19}
    labels = ('bw_mse', 'bw_mse 6', 'bw_mse 3', 'sb_mse', 'sb_mse 38', 'sb_mse 19')

    title = 'MSE comparison between datasets'

    if eval:
        evaluate_translations(score_dict, labels, title)

    return score_dict



def evaluate_models(explanation_set='breastw', eval=True):
    dc1 = DataCollector(explanation_set , model_number=1)
    dc2 = DataCollector(explanation_set , model_number=2)
    dc3 = DataCollector(explanation_set , model_number=3)

    r2_1, mse_1, mse_baseline_1, var_1 = translate_pairwise(dc1.scaled_explanations, dc1.non_zero_explanations)
    r2_2, mse_2, mse_baseline_2, var_2 = translate_pairwise(dc2.scaled_explanations, dc2.non_zero_explanations)
    r2_3, mse_3, mse_baseline_3, var_3 = translate_pairwise(dc3.scaled_explanations, dc3.non_zero_explanations)

    if explanation_set=='breastw':
        mask_1_3 = 6
        mask_2_3 = 3
    else:
        mask_1_3 = 38
        mask_2_3 = 19

    masked_indices1, masked_indices1_nonzero = dc1.mask_features(k=mask_1_3, scaled=True)
    masked_indices2, masked_indices2_nonzero = dc2.mask_features(k=mask_1_3, scaled=True)
    masked_indices3, masked_indices3_nonzero = dc3.mask_features(k=mask_1_3, scaled=True)

    r2_m1, mse_m1, mse_baseline_m1, var_m1 = translate_pairwise(dc1.masked_explanations, dc1.non_zero_masked_explanations, masked=True, masked_indices=masked_indices1, non_zero_masked_indices=masked_indices1_nonzero)
    r2_m2, mse_m2, mse_baseline_m2, var_m2 = translate_pairwise(dc2.masked_explanations, dc2.non_zero_masked_explanations, masked=True, masked_indices=masked_indices2, non_zero_masked_indices=masked_indices2_nonzero)
    r2_m3, mse_m3, mse_baseline_m3, var_m3 = translate_pairwise(dc3.masked_explanations, dc3.non_zero_masked_explanations, masked=True, masked_indices=masked_indices3, non_zero_masked_indices=masked_indices3_nonzero)

    masked_indices12, masked_indices12_nonzero = dc1.mask_features(k=mask_2_3, scaled=True)
    masked_indices22, masked_indices22_nonzero = dc2.mask_features(k=mask_2_3, scaled=True)
    masked_indices32, masked_indices32_nonzero = dc3.mask_features(k=mask_2_3, scaled=True)

    r2_m12, mse_m12, mse_baseline_m12, var_m12 = translate_pairwise(dc1.masked_explanations, dc1.non_zero_masked_explanations, masked=True, masked_indices=masked_indices12, non_zero_masked_indices=masked_indices12_nonzero)
    r2_m22, mse_m22, mse_baseline_m22, var_m22 = translate_pairwise(dc2.masked_explanations, dc2.non_zero_masked_explanations, masked=True, masked_indices=masked_indices22, non_zero_masked_indices=masked_indices22_nonzero)
    r2_m32, mse_m32, mse_baseline_m32, var_m32 = translate_pairwise(dc3.masked_explanations, dc3.non_zero_masked_explanations, masked=True, masked_indices=masked_indices32, non_zero_masked_indices=masked_indices32_nonzero)

    score_dict = {'model 1 mse': mse_1, 'model 1 mse 6': mse_m1, 'model 1 mse 3': mse_m12, 'model 2 mse': mse_2, 'model 2 mse 6': mse_m2, 'model 2 mse 3': mse_m22, 'model 3 mse': mse_3, 'model 3 mse 6': mse_m3, 'model 3 mse 3': mse_m32}

    labels = ('model 1 mse', 'model 1 mse 6', 'model 1 mse 3', 'model 2 mse', 'model 2 mse 6', 'model 2 mse 3', 'model 3 mse', 'model 3 mse 6', 'model 3 mse 3')

    title = 'MSE comparison between models'

    if eval:
        evaluate_translations(score_dict, labels, title)

    return score_dict


def evaluate_autoencoder(explanation_set='breastw', model_number=1, layers_encode=[9, 16, 5], layers_decode=[5, 16, 9], num_epochs=10, lr=0.001, batch_size=32, eval=True):
    if explanation_set=='breastw':
        input_dim = 9
        mask_1_3 = 6
        mask_2_3 = 3
    else:
        input_dim = 57
        mask_1_3 = 38
        mask_2_3 = 19

    autoencoder = Autoencoder(layers_encode, layers_decode)
    dc = DataCollector(explanation_set, model_number=model_number)

    mse = translate_with_autoencoder(autoencoder, dc.scaled_explanations, dc.non_zero_explanations, num_epochs, lr, batch_size)

    masked_indices, masked_indices_nonzero = dc.mask_features(k=mask_1_3, scaled=True)
    mse_m = translate_with_autoencoder(autoencoder, dc.masked_explanations, dc.non_zero_masked_explanations, num_epochs, lr, batch_size)

    masked_indices2, masked_indices2_nonzero = dc.mask_features(k=mask_2_3, scaled=True)
    mse_m2 = translate_with_autoencoder(autoencoder, dc.masked_explanations, dc.non_zero_masked_explanations, num_epochs, lr, batch_size)

    score_dict = {'autoencoder mse': mse, 'autoencoder mse 6': mse_m, 'autoencoder mse 3': mse_m2}
    labels = ('autoencoder mse', 'autoencoder mse 6', 'autoencoder mse 3')

    title = 'MSE comparison between autoencoders'

    if eval:
        evaluate_translations(score_dict, labels, title)

    return score_dict
    

def evaluate_translations(score_dict, labels, title):
    visualize_multiple_scores(score_dict, labels, title)

    ranking_dict = merge_rankings(score_dict)
    grad_dict, perturb_dict, mixed_dict = separate_concepts(ranking_dict)
    
    show_rankings(grad_dict, labels, 'Gradient based methods')
    show_rankings(perturb_dict, labels, 'Perturbation based methods')
    show_rankings(mixed_dict, labels, 'Mixed methods')

    show_rankings_bp(grad_dict)
    show_rankings_bp(perturb_dict)
    show_rankings_bp(mixed_dict)

    print(np.mean(list(grad_dict.values())))
    print(np.mean(list(perturb_dict.values())))
    print(np.mean(list(mixed_dict.values())))










    




