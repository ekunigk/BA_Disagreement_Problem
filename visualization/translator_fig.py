import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
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


# plt_default_backend = plt.get_backend()
# plt_default_params = plt.rcParams

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

def show_three(df, figsize=(15, 8), base_spacing=3, ranking=True, ylabel='Ranking among Methods', title='Ranking of Method Pairs based on Translation MSE'):

    # Configure Matplotlib to use LaTeX
    

    # plt.rcParams.update(pgf_with_latex)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)

    # Decreased spacing for boxplots within the same row
    spacing_within_row = 0.5
    base_spacing = base_spacing

    # Colors for the boxplots
    colors = ['#3CB371','#6CA6CD', '#9F79EE']

    # Iterate through each row to create boxplots
    row_count = 0
    for idx, row in df.iterrows():
        positions = [row_count * base_spacing + i * spacing_within_row for i in range(3)]  # positions for the boxplots
        box_data = [row[col] for col in df.columns]
        row_count += 1
        
        # Create boxplots with colors
        bplots = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True)
        
        for patch, color in zip(bplots['boxes'], colors):
            patch.set_facecolor(color)

        # ax.plot(positions, [baseline] * len(positions), 'k--', linewidth=1)

    # Customizing the plot
    ax.set_xticks([i * base_spacing + spacing_within_row for i in range(len(df))])
    ax.set_xticklabels(df.index)
    ax.set_xlabel('Method Pair')
    if ranking:
        ax.set_ylim([0, 21])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.grid(True)

    # Adding legend manually
    for color, col in zip(colors, df.columns):
        ax.plot([], [], label=col, color=color)
    ax.legend()

    # plt.switch_backend('pgf')
    # plt.rcParams.update(pgf_with_latex)
    # plt.savefig("boxplot_h1.png", format="png")  
    # plt.switch_backend(plt_default_backend)
    # plt.rcParams.update(plt_default_params)

    plt.show()


def represent_values(df, baseline):

    fig, ax = plt.subplots(figsize=(15, 8))

    # Colors for the scatter points
    colors = ['#3CB371','#6CA6CD', '#9F79EE']

    offsets = [-0.2, 0, 0.2]

    b_keys = list(baseline.keys())
    b1 = baseline[b_keys[0]]
    b2 = baseline[b_keys[1]]
    b3 = baseline[b_keys[2]]

    row_count = 0
    # Iterate through each row to create scatter plots
    for idx, row in df.iterrows():
        # Scatter the points for each column
        for i, col in enumerate(df.columns):
            ax.scatter([row_count + offsets[i]] * 10, row[col], color=colors[i], label=col if row_count == 0 else "", alpha=0.7)
        
        # ax.hlines(y=baseline[idx], xmin=row_count + offsets[0], xmax=row_count + offsets[2], colors='black', linestyles='--', linewidth=2)
        ax.hlines(y=b1[idx], xmin=row_count -0.3, xmax=row_count -0.1, color='black')
        ax.hlines(y=b2[idx], xmin=row_count -0.1, xmax=row_count +0.1, color='black')
        ax.hlines(y=b3[idx], xmin=row_count +0.1, xmax=row_count +0.3, color='black')
        row_count += 1

    # Customizing the plot
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index)
    ax.set_xlabel('Method Pair')
    ax.set_ylabel('Translation MSE')
    ax.set_title('Translation MSE of Pairs of Attribution Methods')

    # Set the y-axis to display a range based on the data
    all_values = np.concatenate(df.values.flatten())
    ax.set_ylim([all_values.min() - 0.02, all_values.max() + 0.05])

    baseline_legend = Line2D([0], [0], color='black', linestyle='-', linewidth=1, label='Baseline')

    # Add the legend to the plot
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [baseline_legend])

    # Save the plot as a vector graphic
    # plt.savefig('scatter_plot.svg', format='svg')  # You can also use 'pdf' or other vector formats

    plt.show()


def plot_boxplots(df, figsize=(15, 8)):
    fig, ax = plt.subplots(figsize=figsize)

    # Positions for the boxplots
    positions = np.arange(len(df))

    # Width of each box, set to a fixed small value
    box_width = 0.2

    # Plotting the boxplots
    row_count = 0
    for i, col in enumerate(df.columns):
        box = ax.boxplot(
            df[col].values.tolist(),
            positions=positions + row_count * box_width - box_width,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor=f'C{row_count}', color='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(markerfacecolor=f'C{row_count}', markeredgecolor='black', marker='o')
        )
        row_count += 1

    # Customizing the plot
    ax.set_xticks(positions)
    ax.set_xticklabels(df.index)
    ax.set_xlabel('Row Index')
    ax.set_ylabel('Values')
    ax.set_title('Boxplots of Values')

    # Set the y-axis to show the full range from 1 to 20
    ax.set_ylim([1, 20])

    # Adding legend manually
    handles = [plt.Line2D([0], [0], color=f'C{i}', lw=2, label=col) for i, col in enumerate(df.columns)]
    ax.legend(handles=handles)

    # Save the plot as a vector graphic
    # plt.savefig('boxplot_with_adjustable_size.svg', format='svg')  # You can also use 'pdf' or other vector formats

    plt.show()