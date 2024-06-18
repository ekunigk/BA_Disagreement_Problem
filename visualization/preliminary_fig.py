import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_fa(fa_matrix):
    df = pd.DataFrame(fa_matrix)
    df.columns = ['ig', 'ks', 'li', 'sg', 'vg']
    df.index = ['ig', 'ks', 'li', 'sg', 'vg']
    sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True, cmap='crest')
    plt.title('Pairwise Feature Agreement')
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