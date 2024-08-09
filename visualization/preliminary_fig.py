import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_fa(fa_matrix, title='Pairwise Feature Agreement', figsize=(10,10)):
    plt.figure(figsize=figsize)
    df = pd.DataFrame(fa_matrix)
    df.columns = ['IG', 'KS', 'LI', 'SG', 'VG']
    df.index = ['IG', 'KS', 'LI', 'SG', 'VG']
    sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True, cmap='crest')
    plt.title(title)
    plt.show()


def visualize_fa_differences(fa_diff, k_list, random_comparison=False):
    df = pd.DataFrame(fa_diff)
    if random_comparison:
        df.columns = ['ig_ks', 'ig_li', 'ig_sg', 'ig_vg', 'ks_li', 'ks_sg', 'ks_vg', 'li_sg', 'li_vg', 'sg_vg', 'random']
    else:
        df.columns = ['ig_ks', 'ig_li', 'ig_sg', 'ig_vg', 'ks_li', 'ks_sg', 'ks_vg', 'li_sg', 'li_vg', 'sg_vg'] 
    df.index = k_list
    plt.plot(df)
    plt.legend(df.columns, fontsize='6')
    plt.show()