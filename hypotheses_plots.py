from data_management.data_saving import load_dict, save_dict
from evaluation.ranking import rank_entries, separate_kfold, normalize_df
from evaluation3 import evaluate_models
from visualization.translator_fig import show_three, represent_values
import pandas as pd

"""
translation evaluation methods
"""

def show_mse_and_baseline(dataset='breastw', architecture='ae5',fig_size=(15,8), save_plt=False, path='mse.pdf'):

    """
    part 1 of translation analysis: mse values and baseline
    """
    
    if architecture == 'lr':
        no_mask_dict, b_dict = prepare_lr(dataset, mode='mse')
    else: 
        no_mask_dict, b_dict = generate_nomask_dicts(dataset, architecture)

    mse_df = pd.DataFrame(no_mask_dict)

    represent_values(mse_df, b_dict, figsize=fig_size, save_plt=save_plt, path=path)


def show_rankings_mse(dataset='breastw', architecture='ae5', fig_size1=(8,8), fig_size2=(12,8), save_plt=False, path1='ranking.pdf', path2='ranking2.pdf'):

    """
    part 2 of translation analysis: ranking of mse values
    """

    if architecture == 'lr':
        no_mask_dict, b_dict = prepare_lr(dataset, mode='mse')

    else:
        no_mask_dict, b_dict = generate_nomask_dicts(dataset, architecture)
    df = pd.DataFrame(no_mask_dict)

    ranked = rank_entries(df, b_dict)
    ranked_dict = ranked.to_dict()

    same, mixed = separate_kfold(ranked_dict)

    df_same = pd.DataFrame(same)
    df_mixed = pd.DataFrame(mixed)

    show_three(df_same, fig_size1, save_plt=save_plt, path=path1)
    show_three(df_mixed, fig_size2, save_plt=save_plt, path=path2)


def show_mse_with_mask(dataset='breastw', model_number=1, architecture='ae5', fig_size=(15,8), save_plt=False, path='mse.pdf'):

    """
    part 3 of translation analysis: mse values with masks
    """

    if architecture == 'lr':
        mse_dict, b_dict = prepare_lr(dataset, mode='mask', model_number=model_number)

    else:
        mse, baseline = generate_dicts_per_model(dataset, model_number, architecture)

        mse_dict = {'no mask': mse['autoencoder mse'], '1_3 mask': mse['autoencoder mse 1_3'], '2_3 mask': mse['autoencoder mse 2_3']}
        b_dict = {'no mask': baseline['m'+str(model_number)+' no mask'], '1_3 mask': baseline['m'+str(model_number)+' 1/3 mask'], '2_3 mask': baseline['m'+str(model_number)+' 2/3 mask']}
    # b_dict = {'no mask': baseline['m'+str(model_number)+' no mask'], '1_3 mask': baseline['m'+str(model_number)+' no mask'], '2_3 mask': baseline['m'+str(model_number)+' no mask']}
    
    df = pd.DataFrame(mse_dict)
    # normalized_df = normalize_df(df, b_dict)

    # show_three(normalized_df, (15,8), 3, ranking=False, ylabel='MSE', title='MSE with different masks')
    represent_values(df, b_dict, figsize=fig_size, save_plt=save_plt, path=path)



def generate_nomask_dicts(dataset='breastw', architecture='ae5'):

    """
    method for data generation from saved pickle files
    """


    if dataset == 'breastw':
        ds = 'bw'
    elif dataset == 'spambase':
        ds = 'sb'

    path1 = 'saves/' + ds + '/kfold/' + architecture + '_m1_20_01_16.pkl'
    b_path1 = 'saves/baseline/' + ds + '_baselines_m1.pkl'
    path2 = 'saves/' + ds + '/kfold/' + architecture + '_m2_20_01_16.pkl'
    b_path2 = 'saves/baseline/' + ds + '_baselines_m2.pkl'
    path3 = 'saves/' + ds + '/kfold/' + architecture + '_m3_20_01_16.pkl'
    b_path3 = 'saves/baseline/' + ds + '_baselines_m3.pkl'

    kfold1 = load_dict(path1)
    kfold2 = load_dict(path2)
    kfold3 = load_dict(path3)
    b1 = load_dict(b_path1)
    b2 = load_dict(b_path2)
    b3 = load_dict(b_path3)

    no_mask_dict = {'m1': kfold1['autoencoder mse'], 'm2': kfold2['autoencoder mse'], 'm3': kfold3['autoencoder mse']}
    b_dict = {'m1': b1['m1 no mask'], 'm2': b2['m2 no mask'], 'm3': b3['m3 no mask']}

    return no_mask_dict, b_dict


def generate_dicts_per_model(dataset='breastw', model_number=1, architecture='ae5'):
    if dataset == 'breastw':
        ds = 'bw'
    elif dataset == 'spambase':
        ds = 'sb'

    path = 'saves/' + ds + '/kfold/' + architecture + '_m' + str(model_number) + '_20_01_16.pkl'
    b_path = 'saves/baseline/' + ds + '_baselines_m' + str(model_number) + '.pkl'

    kfold = load_dict(path)
    b = load_dict(b_path)

    return kfold, b



def prepare_lr(dataset='breastw', mode='mse', model_number=1):
    score_dict, kfold_dict = evaluate_models(dataset, False)

    if dataset == 'breastw':
        ds = 'bw'
    elif dataset == 'spambase':
        ds = 'sb'

    b_path1 = 'saves/baseline/' + ds + '_baselines_m1.pkl'
    b_path2 = 'saves/baseline/' + ds + '_baselines_m2.pkl'
    b_path3 = 'saves/baseline/' + ds + '_baselines_m3.pkl'

    b1 = load_dict(b_path1)
    b2 = load_dict(b_path2)
    b3 = load_dict(b_path3)

    if model_number == 1:
        baseline = b1
    elif model_number == 2:
        baseline = b2
    elif model_number == 3:
        baseline = b3

    if mode == 'mse':
        no_mask_dict = {'m1': kfold_dict['model 1 mse'], 'm2': kfold_dict['model 2 mse'], 'm3': kfold_dict['model 3 mse']}
        b_dict = {'m1': b1['m1 no mask'], 'm2': b2['m2 no mask'], 'm3': b3['m3 no mask']}

        return no_mask_dict, b_dict

    elif mode == 'mask':
        mse_dict = {'no mask': kfold_dict['model '+str(model_number)+' mse'], '1_3 mask': kfold_dict['model '+str(model_number)+' mse 1_3'], '2_3 mask': kfold_dict['model '+str(model_number)+' mse 2_3']}
        b_dict = {'no mask': baseline['m'+str(model_number)+' no mask'], '1_3 mask': baseline['m'+str(model_number)+' 1/3 mask'], '2_3 mask': baseline['m'+str(model_number)+' 2/3 mask']}   

        return mse_dict, b_dict


