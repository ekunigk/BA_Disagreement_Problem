import seaborn as sns
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# visualization of translation scores in various versions

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

# visualization of residuals

def analyze_residuals(residuals, dim=0):
    residual = residuals[dim]
    sns.histplot(residual)
    plt.title('Residuals')
    plt.show()


def show_rankings(ranking_dict, label, title='Ranking of MSE of linear translation', figsize=(18, 6)):
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


# final visualization of rankings in boxplots

def show_three(df, figsize=(15, 8), base_spacing=2, ranking=True, ylabel=True, save_plt=False, path='box.pdf'):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=10.95)   

    fig, ax = plt.subplots(figsize=figsize)

    spacing_within_row = 0.5
    base_spacing = base_spacing
    box_width = 0.3

    colors = ['#3CB371','#6CA6CD', '#9F79EE']

    row_count = 0
    for idx, row in df.iterrows():
        positions = [row_count * base_spacing + i * spacing_within_row  for i in range(3)]  # positions for the boxplots
        box_data = [row[col] for col in df.columns]
 
        row_count += 1

        bplots = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True)
        
        for patch, color in zip(bplots['boxes'], colors):
            patch.set_facecolor(color)

    ax.set_xticks([i * base_spacing + spacing_within_row for i in range(len(df))])
    ax.set_xticklabels(df.index)
    ax.set_xlabel('Method Pair')
    if ranking:
        ax.set_ylim([0, 22])
        ax.set_yticks(np.arange(0, 20, 2))
    ax.set_ylabel('Rank')
    plt.xticks(rotation=45)

    for color, col in zip(colors, df.columns):
        ax.plot([], [], label=col, color=color)

    plt.tick_params(
        axis='y',         
        which='both',
        left=False,
        right=False,
        labelleft=False) 

    plt.legend(('M1', 'M2', 'M3'), loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3,  handletextpad=0.2, columnspacing=0.5)
    plt.tight_layout()

    plt.show()

    if save_plt:
        path_pdf = "figures/translation/ranks/" + path
        plt.savefig(path_pdf, format='pdf', dpi=300)


# final visualization of mses in scatter plot for first translations and masked version

def represent_values(df, baseline, figsize=(15, 8), save_plt=False, path='scatter.pdf'):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=10.95)   

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#3CB371','#6CA6CD', '#9F79EE']

    offsets = [-0.2, 0, 0.2]

    b_keys = list(baseline.keys())
    b1 = baseline[b_keys[0]]
    b2 = baseline[b_keys[1]]
    b3 = baseline[b_keys[2]]

    row_count = 0
    for idx, row in df.iterrows():
        for i, col in enumerate(df.columns):
            ax.scatter([row_count + offsets[i]] * 10, row[col], color=colors[i], label=col if row_count == 0 else "", alpha=0.7)
        
        ax.hlines(y=b1[idx], xmin=row_count -0.3, xmax=row_count -0.1, color='black')
        ax.hlines(y=b2[idx], xmin=row_count -0.1, xmax=row_count +0.1, color='black')
        ax.hlines(y=b3[idx], xmin=row_count +0.1, xmax=row_count +0.3, color='black')
        row_count += 1

    ax.set_xticks(range(len(df)))
    plt.xticks(rotation=45)
    ax.set_xticklabels(df.index, fontdict={'size': 10.95})
    plt.margins(x=0, y=0)
    ax.set_xlabel('Method Pair')
    ax.set_ylabel('MSE')

    # plt.tick_params(
    #     axis='y',          # changes apply to the x-axis
    #     which='both',
    #     left=False,
    #     right=False,
    #     labelleft=False) # labels along the bottom edge are

    # ax.legend(loc='upper center', bbox_to_anchor=(1, 1.15),
    #       ncol=3)

    all_values = np.concatenate(df.values.flatten())

    baseline_legend = Line2D([0], [0], color='black', linestyle='-', linewidth=1, label='Baseline')
    plt.legend(('No Mask', '1/3 Masked', '2/3 Masked', 'Baseline'), loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4,  handletextpad=0.2, columnspacing=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    if save_plt:
        path_pdf = "figures/translation/" + path
        plt.savefig(path_pdf, format='pdf', dpi=300)
