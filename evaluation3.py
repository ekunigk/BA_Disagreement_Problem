from data_management.data_collector import DataCollector
from evaluation.linear_translator import translate_pairwise, calculate_percentage_of_baseline
from evaluation.autoencoder import Autoencoder
from evaluation.autoencoder_training import translate_with_autoencoder
from visualization.translator_fig import visualize_multiple_scores, show_rankings, show_rankings_bp
from evaluation.ranking import merge_rankings, create_rankings, separate_concepts, merge_two_dicts
from data_management.data_saving import load_dict, save_dict

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

    score_dict = {'model 1 mse': mse_1, 'model 1 mse 1_3': mse_m1, 'model 1 mse 2_3': mse_m12, 'model 2 mse': mse_2, 'model 2 mse 1_3': mse_m2, 'model 2 mse 2_3': mse_m22, 'model 3 mse': mse_3, 'model 3 mse 1_3': mse_m3, 'model 3 mse 2_3': mse_m32}

    labels = ('model 1 mse', 'model 1 mse 6', 'model 1 mse 3', 'model 2 mse', 'model 2 mse 6', 'model 2 mse 3', 'model 3 mse', 'model 3 mse 6', 'model 3 mse 3')

    title = 'MSE comparison between models'

    if eval:
        evaluate_translations(score_dict, labels, title)

    return score_dict, mse_baseline_1, mse_baseline_m1, mse_baseline_m12


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

    mse, mse_kfold = translate_with_autoencoder(autoencoder, dc.scaled_explanations, dc.non_zero_explanations, num_epochs, lr, batch_size)

    masked_indices, masked_indices_nonzero = dc.mask_features(k=mask_1_3, scaled=True)
    mse_m, mse_m_kfold = translate_with_autoencoder(autoencoder, dc.masked_explanations, dc.non_zero_masked_explanations, num_epochs, lr, batch_size, masked=True, masked_indices=masked_indices, non_zero_masked_indices=masked_indices_nonzero)

    masked_indices2, masked_indices2_nonzero = dc.mask_features(k=mask_2_3, scaled=True)
    mse_m2, mse_m2_kfold = translate_with_autoencoder(autoencoder, dc.masked_explanations, dc.non_zero_masked_explanations, num_epochs, lr, batch_size, masked=True, masked_indices=masked_indices2, non_zero_masked_indices=masked_indices2_nonzero)

    score_dict = {'autoencoder mse': mse, 'autoencoder mse 1_3': mse_m, 'autoencoder mse 2_3': mse_m2}
    kfold_dict = {'autoencoder mse': mse_kfold, 'autoencoder mse 1_3': mse_m_kfold, 'autoencoder mse 2_3': mse_m2_kfold}
    labels = ('autoencoder mse', 'autoencoder mse 1_3', 'autoencoder mse 2_3')

    title = 'MSE comparison between autoencoders'

    if eval:
        evaluate_translations(score_dict, labels, title)

    return score_dict, kfold_dict
    

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



def evaluate_ae_lr_models(explanation_set='breastw', masking=0, show=True):

    if explanation_set=='breastw':
        path = 'saves/bw/'
    else:
        path = 'saves/sb/'

    sb_ae165 = load_dict(path+'score_dict_ae165_20_01_16.pkl')
    sb_ae165 = {key + ' 165': value for key, value in sb_ae165.items()}
    sb_ae85 = load_dict(path+'score_dict_ae84_20_01_16.pkl')
    sb_ae85 = {key + ' 85': value for key, value in sb_ae85.items()}
    sb_ae165_m2 = load_dict(path+'score_dict_ae165_20_01_16_m2.pkl')
    sb_ae165_m2 = {key + ' 165 m2': value for key, value in sb_ae165_m2.items()}
    sb_ae85_m2 = load_dict(path+'score_dict_ae84_20_01_16_m2.pkl')
    sb_ae85_m2 = {key + ' 85 m2': value for key, value in sb_ae85_m2.items()}
    sb_ae165_m3 = load_dict(path+'score_dict_ae165_20_01_16_m3.pkl')
    sb_ae165_m3 = {key + ' 165 m3': value for key, value in sb_ae165_m3.items()}
    sb_ae85_m3 = load_dict(path+'score_dict_ae84_20_01_16_m3.pkl')
    sb_ae85_m3 = {key + ' 85 m3': value for key, value in sb_ae85_m3.items()}

    score_dict, baseline = evaluate_models(explanation_set, eval=False)

    complete_1 = merge_two_dicts(sb_ae165, sb_ae85)
    complete_2 = merge_two_dicts(sb_ae165_m2, sb_ae85_m2)
    complete_3 = merge_two_dicts(sb_ae165_m3, sb_ae85_m3)
    c12 = merge_two_dicts(complete_1, complete_2)
    c123 = merge_two_dicts(c12, complete_3)
    	
    if masking==0:
        mask = {'165 m1': c123['autoencoder mse 165'],
             '85 m1': c123['autoencoder mse 85'],
             'lr m1': score_dict['model 1 mse'],
             '165 m2': c123['autoencoder mse 165 m2'],
             '85 m2': c123['autoencoder mse 85 m2'],
             'lr m2': score_dict['model 2 mse'],
             '165 m3': c123['autoencoder mse 165 m3'],
             '85 m3': c123['autoencoder mse 85 m3'],
             'lr m3': score_dict['model 3 mse'],
             'baseline': baseline}
    
        labels = ('165 m1', '84 m1', 'lr m1', '165 m2', '84 m2', 'lr m2', '165 m3', '84 m3', 'lr m3', 'baseline')
    title = 'MSE comparison between autoencoders and linear models'

    if masking==1:
        mask = {'165 m1': c123['autoencoder mse 1_3 165'],
             '85 m1': c123['autoencoder mse 1_3 85'],
             'lr m1': score_dict['model 1 mse 1_3'],
             '165 m2': c123['autoencoder mse 1_3 165 m2'],
             '85 m2': c123['autoencoder mse 1_3 85 m2'],
             'lr m2': score_dict['model 2 mse 1_3'],
             '165 m3': c123['autoencoder mse 1_3 165 m3'],
             '85 m3': c123['autoencoder mse 1_3 85 m3'],
             'lr m3': score_dict['model 3 mse 1_3'],
             'baseline': baseline}

        labels = ('1/3 165 m1', '1/3 84 m1', '1/3 lr m1', '1/3 165 m2', '1/3 84 m2', '1/3 lr m2', '1/3 165 m3', '1/3 84 m3', '1/3 lr m3', 'baseline')

    if masking==2:
        mask = {'165 m1': c123['autoencoder mse 2_3 165'],
             '85 m1': c123['autoencoder mse 2_3 85'],
             'lr m1': score_dict['model 1 mse 2_3'],
             '165 m2': c123['autoencoder mse 2_3 165 m2'],
             '85 m2': c123['autoencoder mse 2_3 85 m2'],
             'lr m2': score_dict['model 2 mse 2_3'],
             '165 m3': c123['autoencoder mse 2_3 165 m3'],
             '85 m3': c123['autoencoder mse 2_3 85 m3'],
             'lr m3': score_dict['model 3 mse 2_3'],
             'baseline': baseline}

        labels = ('2/3 165 m1', '2/3 84 m1', '2/3 lr m1', '2/3 165 m2', '2/3 84 m2', '2/3 lr m2', '2/3 165 m3', '2/3 84 m3', '2/3 lr m3', 'baseline')
    
    print(baseline)

    percentages_of_b = {}
    for key in mask:
        percentages_of_b[key] = calculate_percentage_of_baseline(mask[key], c123['autoencoder mse 165'])

    if show==True:
        evaluate_translations(mask, labels, title)

        evaluate_translations(percentages_of_b, labels, 'Percentage of baseline comparison between autoencoders and linear models')

    return mask, percentages_of_b






