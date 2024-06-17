import seaborn as sns
import matplotlib.pyplot as plt

def visualize_scores(scores):
    method_list = list(scores.keys())
    score_list = list(scores.values())
    sns.scatterplot(x=method_list, y=score_list)
    plt.title('Logistic Regression Accuracy')
    plt.xlabel('Method Pairs')
    plt.ylabel('Accuracy')
    plt.show()


def visualize_scores_all(scores_all):
    sns.boxplot(scores_all)
    plt.show()


def visualize_attempt(scores_all, legend_names=('one', 'two', 'three')):

    color_list = ['blue', 'purple', 'red', 'orange', 'yellow']
    name_list = ['one', 'two', 'three', 'four', 'five']

    for i in range(len(scores_all)):
        name_list[i] = plt.scatter(x=list(scores_all[i].keys()), y=list(scores_all[i].values()), color=color_list[i])

    name_list = tuple(name_list[0:len(scores_all)])

    plt.legend(name_list, 
               legend_names,
               loc='upper right',
               fontsize='8')
    
    plt.title('Logistic Regression Accuracy')
    plt.xlabel('Method Pairs')
    plt.ylabel('Accuracy')

    # plt.plot(a, b, c)
    plt.show()