from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.use('pdf')

import matplotlib.pyplot as plt
import seaborn as sns
import umap
import torch

"""
umap projection of the explanations for the preliminary analysis
"""


def project_umap(dataset, non_zero_dataset, n_neighbors=15, min_dist=0.1, scale=False):
    feature_set = dataset[:, :-1]

    method_length = int(len(feature_set)/5)
    non_zero_feature_set = non_zero_dataset[:, :-1]
    non_zero_method_length = int(len(non_zero_feature_set)/5)
    ig_ks = feature_set[:2*method_length]
    li = non_zero_feature_set[2*non_zero_method_length:3*non_zero_method_length]
    vg_sg = feature_set[3*method_length:]
    
    final_feature_set = torch.cat((ig_ks, li, vg_sg), 0)

    if scale:
        feature_set = StandardScaler().fit_transform(final_feature_set)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = reducer.fit_transform(final_feature_set)
    return embedding, method_length, non_zero_method_length



def visualize_umap(embedding, method_length, non_zero_method_length, figsize=(5,5), save_plt=False, path='umap.pdf'):

    """
    visualize umap projection
    """

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=figsize)
    
    plt.rc('font', size=10.95)   

    length = int(len(embedding) / 5)

    colors = ['#eda6c4', '#5fa777', '#164f7f', '#78184a', '#008080']  # 0b0b45 004aad

    ig = plt.scatter(embedding[0:method_length, 0], embedding[0:method_length, 1], color=colors[0], alpha=1)
    ks = plt.scatter(embedding[method_length:(2*method_length), 0], embedding[method_length:(2*method_length), 1], color=colors[1], alpha=1)
    li = plt.scatter(embedding[(2*method_length):(2*method_length+non_zero_method_length), 0], embedding[(2*method_length):(2*method_length+non_zero_method_length), 1], color=colors[2], alpha=1)
    sg = plt.scatter(embedding[(2*method_length+non_zero_method_length):(3*method_length+non_zero_method_length), 0], embedding[(2*method_length+non_zero_method_length):(3*method_length+non_zero_method_length), 1], color=colors[3], alpha=1)
    vg = plt.scatter(embedding[(3*method_length+non_zero_method_length):, 0], embedding[(3*method_length+non_zero_method_length):, 1], color=colors[4], alpha=1)

    # ks = plt.scatter(embedding[method_length:(2*method_length), 0], embedding[method_length:(2*method_length), 1], color=colors[1], linewidths=0.8, edgecolors='w', alpha=1)

    plt.legend((ig, ks, li, sg, vg),
               ('IG', 'KS', 'LI', 'SG', 'VG'),
               scatterpoints=1,
               loc='upper center',
               fontsize=10.95,
               bbox_to_anchor=(0.5, 1.18), ncol=5,  handletextpad=0.1, columnspacing=0.05)
    
    plt.tick_params(
        which='both',
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelbottom=False,
        labelleft=False)
    
    # plt.title('UMAP projection of the dataset')    
    plt.show()
    plt.tight_layout()

    if save_plt:
        path_pdf = "figures/umap/" + path
        plt.savefig(path_pdf, format='pdf', dpi=300)
