from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import umap


def visualize_umap(dataset, scale=False):
    
    feature_set = dataset[:, :-1]
    if scale:
        feature_set = StandardScaler().fit_transform(feature_set)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(feature_set)

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in dataset[:, -1].numpy().astype(int)]
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the dataset', fontsize=24)
    return embedding


