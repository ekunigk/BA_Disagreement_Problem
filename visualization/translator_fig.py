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


def visualize_multiple_scores(scores_1, scores_2, scores_3, labels, title):
    x = scores_1.keys()
    y1 = scores_1.values()
    y2 = scores_2.values()
    y3 = scores_3.values()
    plt.figure(figsize=(18, 6))
    plt.scatter(x, y1, label=labels[0])
    plt.scatter(x, y2, label=labels[1])
    plt.scatter(x, y3, label=labels[2])
    plt.title(title)
    plt.legend(labels)
    plt.show()
