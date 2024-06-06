import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_fa(fa_matrix):
    df = pd.DataFrame(fa_matrix)
    df.columns = ['ig', 'ks', 'li', 'sg', 'vg']
    df.index = ['ig', 'ks', 'li', 'sg', 'vg']
    sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True, cmap='crest')
    plt.show()