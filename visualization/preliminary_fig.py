import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

# feature agreement visualization in heatmap

def visualize_fa(fa_matrix, title='Pairwise Feature Agreement', figsize=(10,10), save_pgf=False):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=figsize)
    plt.rc('font', size=10.95)   

    df = pd.DataFrame(fa_matrix)
    df.columns = ['IG', 'KS', 'LI', 'SG', 'VG']
    df.index = ['IG', 'KS', 'LI', 'SG', 'VG']
    sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True, cmap='crest')
    plt.title(title)
    plt.show()


    if save_pgf:
        plt.savefig("fa_matrix.pdf", format='pdf', dpi=300)


def visualize_fa2(fa_matrix, figsize=(4, 4), save_plt=False, path='fa_matrix.pdf'):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=figsize)

    plt.rc('font', size=10.95)   

    df = pd.DataFrame(fa_matrix)
    df.columns = ['IG', 'KS', 'LI', 'SG', 'VG']
    df.index = ['IG', 'KS', 'LI', 'SG', 'VG']

    sns.heatmap(df, annot=True, fmt=".2f", annot_kws={"size": 10.95}, cmap='crest', 
                square=True, cbar_kws={"shrink": 0.8})
    
    plt.tight_layout()  

    plt.show()

    if save_plt:
        path_pdf = "figures/fa/" + path
        plt.savefig(path_pdf, format='pdf', dpi=300)

# visualization of tests with varying k

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