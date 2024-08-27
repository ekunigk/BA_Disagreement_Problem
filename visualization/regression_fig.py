import seaborn as sns
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

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


def visualize_attempt(scores_all, legend_names=('one', 'two', 'three'), figsize=(5,3), save_plt=False, path='lr.pdf'):

    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=figsize)
    
    # Set overall font size
    plt.rc('font', size=10.95)   

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

    if save_plt:
        path_pdf = "figures/logreg/" + path
        plt.savefig(path_pdf, format='pdf', dpi=300)


def visualize_scores_temp(scores, figsize=(5,3), save_plt=False, path='lr.pdf'):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=figsize)
    plt.rc('font', size=10.95)   

    score_array = np.array(list(scores.values())).T
    plt.boxplot(score_array)
    plt.xticks(range(1, len(scores)+1), ['IG-KS', 'IG-LI', 'IG-SG', 'IG-VG', 'KS-LI', 'KS-SG', 'KS-VG', 'LI-SG', 'LI-VG', 'SG-VG'], rotation=45)

    # plt.xlabel('Method Pairs')
    plt.ylabel('Accuracy')
    # plt.title('Classification Accuracy for Different Method Pairs')
    # plt.grid(True)
    plt.show()
    plt.tight_layout()

    if save_plt:
        path_pdf = "figures/logreg/" + path
        plt.savefig(path_pdf, format='pdf', dpi=300)

