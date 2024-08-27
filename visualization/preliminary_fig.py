import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

# plt_default_backend = plt.get_backend()
# plt_default_params = plt.rcParams

# pgf_with_latex = {
#     "pgf.texsystem": "lualatex",
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": [],
#     "font.sans-serif": [],
#     "font.monospace": [],
#     "legend.fontsize": 10.95,
#     "axes.labelsize": 10.95,
#     "font.size": 10.95,
#     "legend.fontsize": 10.95,
#     "xtick.labelsize": 10.95,
#     "ytick.labelsize": 10.95,
#     "pgf.preamble": "\n".join([
#         r"\usepackage[utf8]{inputenc}",
#         r"\usepackage[T1]{fontenc}",
#         r"\usepackage{cmbright}",
#     ])
# }



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
    # plt.savefig('fa_matrix.pgf', format='pgf')

    if save_pgf:
        # plt.switch_backend('pgf')
        # plt.rcParams.update(pgf_with_latex)
        plt.savefig("fa_matrix.pdf", format='pdf', dpi=300)

        # plt.switch_backend(plt_default_backend)
        # plt.rcParams.update(plt_default_params)



def visualize_fa2(fa_matrix, figsize=(4, 4), save_plt=False, path='fa_matrix.pdf'):

    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=figsize)
    
    # Set overall font size
    plt.rc('font', size=10.95)   

    # Create a DataFrame for better handling
    df = pd.DataFrame(fa_matrix)
    df.columns = ['IG', 'KS', 'LI', 'SG', 'VG']
    df.index = ['IG', 'KS', 'LI', 'SG', 'VG']

    # vmin = 0. 
    # vmax = df.values.max()

    # Generate heatmap with smaller font for annotations and less padding
    sns.heatmap(df, annot=True, fmt=".2f", annot_kws={"size": 10.95}, cmap='crest', 
                square=True, cbar_kws={"shrink": 0.8})
    
    # Title and layout adjustments
    # plt.title(title, fontsize=10.95)
    plt.tight_layout()  # Reduce extra space around the plot
    
    # Show the plot
    plt.show()

    # Optionally save the plot as PDF
    if save_plt:
        path_pdf = "figures/fa/" + path
        plt.savefig(path_pdf, format='pdf', dpi=300)





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