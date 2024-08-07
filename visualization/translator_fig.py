import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

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


def show_three_bp(df):
    fig, ax = plt.subplots(figsize=(15, 8))

    # Iterate through each row to create boxplots
    count = 0
    for idx, row in df.iterrows():
        positions = [count * 3 + i for i in range(3)]  # positions for the boxplots
        box_data = [row[col] for col in df.columns]
        ax.boxplot(box_data, positions=positions, widths=0.6)
        count +=1

    # Customizing the plot
    ax.set_xticks([i * 3 + 1 for i in range(len(df))])
    ax.set_xticklabels(df.index)
    ax.set_xlabel('Row Index')
    ax.set_ylabel('Value Ranges')
    ax.set_title('Boxplots of MSE Values per Row')
    ax.grid(True)

    # Adding legend manually
    for i, col in enumerate(df.columns):
        ax.plot([], [], label=col, color='C'+str(i))
    ax.legend()

    plt.show()


def show_three(df):

    # Configure Matplotlib to use LaTeX
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

    # plt.rcParams.update(pgf_with_latex)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))

    # Decreased spacing for boxplots within the same row
    spacing_within_row = 0.8
    base_spacing = 3

    # Colors for the boxplots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Iterate through each row to create boxplots
    row_count = 0
    for idx, row in df.iterrows():
        positions = [row_count * base_spacing + i * spacing_within_row for i in range(3)]  # positions for the boxplots
        box_data = [row[col] for col in df.columns]
        row_count += 1
        
        # Create boxplots with colors
        bplots = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
        
        for patch, color in zip(bplots['boxes'], colors):
            patch.set_facecolor(color)

    # Customizing the plot
    ax.set_xticks([i * base_spacing + spacing_within_row for i in range(len(df))])
    ax.set_xticklabels(df.index)
    ax.set_xlabel('Row Index')
    ax.set_ylabel('Value Ranges')
    ax.set_title('Boxplots of MSE Values per Row')
    ax.grid(True)

    # Adding legend manually
    for color, col in zip(colors, df.columns):
        ax.plot([], [], label=col, color=color)
    ax.legend()

    # Save the plot as a vector graphic
    # plt.savefig('boxplots.svg', format='svg')  # You can also use 'pdf' or other vector formats

    plt.show()
