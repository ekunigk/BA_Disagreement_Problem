from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import umap

def project_umap(dataset, scale=False):
    feature_set = dataset[:, :-1]
    if scale:
        feature_set = StandardScaler().fit_transform(feature_set)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(feature_set)
    return embedding

def visualize_umap_nolegend(dataset, embedding, scale=False):

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in dataset[:, -1].numpy().astype(int)]
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the dataset', fontsize=24)

    plt.show()
    return embedding


def visualize_umap2(dataset, embedding, scale=False):

    label = dataset[:, -1].numpy().astype(int)


    sca = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=dataset[:, -1].numpy().astype(int),
        palette='deep'
    )
    sca.legend(loc='upper right', labels=['ig', 'ks', 'li', 'sg', 'vg'])
    sca.set()
    plt.show()
    return embedding

def visualize_umap(dataset, embedding):

    length = int(len(embedding) / 5)
    print(length)

    colors = ['blue', 'orange', 'green', 'red', 'purple']

    ig = plt.scatter(embedding[0:length, 0], embedding[0:length, 1], color=colors[0], linewidths=0.8, edgecolors='w', alpha=0.9)
    ks = plt.scatter(embedding[length:(2*length), 0], embedding[length:(2*length), 1], color=colors[1], linewidths=0.8, edgecolors='w', alpha=0.9)
    li = plt.scatter(embedding[(2*length):(3*length), 0], embedding[(2*length):(3*length), 1], color=colors[2], linewidths=0.8, edgecolors='w', alpha=0.9)
    sg = plt.scatter(embedding[(3*length):(4*length), 0], embedding[(3*length):(4*length), 1], color=colors[3], linewidths=0.8, edgecolors='w', alpha=0.9)
    vg = plt.scatter(embedding[(4*length):, 0], embedding[(4*length):, 1], color=colors[4], linewidths=0.8, edgecolors='w', alpha=0.9)

    plt.legend((ig, ks, li, sg, vg),
               ('ig', 'ks', 'li', 'sg', 'vg'),
               scatterpoints=1,
               loc='upper right',
               ncol=1,
               fontsize=8)
    
    plt.show()
