import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def visualize_translation_scores(scores_1):
    x = scores_1.keys()
    y = scores_1.values()
    plt.figure(figsize=(18, 6))
    plt.scatter(x, y)
    plt.title('Translation Scores')
    plt.show()


def visualize_multiple_scores2(scores_1, scores_2, scores_3, labels, title, baseline=0):
    x = scores_1.keys()
    y1 = scores_1.values()
    y2 = scores_2.values()
    y3 = scores_3.values()
    
    plt.figure(figsize=(18, 6))

    if baseline != 0: 
        baseline = baseline.values()

    plt.scatter(x, y1, label=labels[0])
    plt.scatter(x, y2, label=labels[1])
    plt.scatter(x, y3, label=labels[2])
    if baseline != 0:
        plt.scatter(x, baseline, label='Baseline')
        labels = labels + ('mean baseline',)
    plt.title(title)
    plt.legend(labels)
    plt.show()



def visualize_multiple_scores(score_dict, labels, title, figsize=(18,6)):
    plt.figure(figsize=figsize)
    keys = score_dict.keys()
    for key in keys:
        plt.scatter(x=score_dict[key].keys(), y=score_dict[key].values(), alpha=0.6)
    plt.title(title)
    plt.legend(labels)
    plt.show()


def analyze_residuals(residuals, dim=0):
    residual = residuals[dim]
    sns.histplot(residual)
    plt.title('Residuals')
    plt.show()


def show_rankings(ranking_dict, label, title='Ranking of MSE of linear translation', figsize=(18, 6),):
    x = ranking_dict.keys()

    plt.figure(figsize=figsize)
    
    for i in range(len(ranking_dict[list(ranking_dict.keys())[0]])):
        y = [ranking[i] for ranking in ranking_dict.values()]
        plt.scatter(x, y, alpha=0.6)

    plt.ylim(0,22)
    plt.yticks(np.arange(0, 22, 2))
    
    plt.legend(label)
    plt.title(title)
    plt.show()


def show_rankings_bp(ranking_dict, title='Ranking of MSE of translation', figsize=(18, 6)):
    mse_array = np.array(list(ranking_dict.values()))
    mse_array = mse_array.T
    plt.boxplot(mse_array)
    plt.ylim(0, 22)
    plt.yticks(np.arange(0, 22, 2))
    plt.xticks(np.arange(1, len(ranking_dict.keys())+1), ranking_dict.keys())
    plt.title(title)
    plt.show()