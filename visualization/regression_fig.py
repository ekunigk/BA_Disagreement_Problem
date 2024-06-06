import seaborn as sns
import matplotlib.pyplot as plt

def visualize_scores(scores):
    method_list = list(scores.keys())
    score_list = list(scores.values())
    sns.scatterplot(x=method_list, y=score_list)
    plt.show()


def visualize_scores_all(scores_all):
    sns.boxplot(scores_all)
    plt.show()


def visualize_attempt(scores_all):
    a = sns.scatterplot(x=list(scores_all[0].keys()), y=list(scores_all[0].values()))
    b = sns.scatterplot(x=list(scores_all[1].keys()), y=list(scores_all[1].values()))
    c = sns.scatterplot(x=list(scores_all[2].keys()), y=list(scores_all[2].values()))
    plt.plot(a, b, c)
    plt.show()