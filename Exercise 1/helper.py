# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt


def boxplot_raw_data(data: pd.DataFrame, columns: list, style='seaborn-poster', save_fig_path=None, sharex=False) -> None:
    plt.style.use([style, 'fast'])

    # figsize is in inches, A4 = 8.27 x 11.69
    figure, ax = plt.subplots(len(columns), 1, sharex=sharex, tight_layout=True)

    for num, column in enumerate(columns):
        p = ax[num]
        p.boxplot(data.loc[:, column], vert=False)

        p.set_title(column)
        p.grid(False)

    if save_fig_path is not None:
        plt.savefig(save_fig_path, format='png')
        plt.close(figure)
    else:
        plt.show()


def plot_time_series_data(data: pd.DataFrame, columns: list, style='seaborn', save_fig_path=None, sharex=True) -> None:
    plot_args = {'linewidth': 0.7, 'alpha': 0.75}

    plt.style.use([style, 'fast'])

    # figsize is in inches, A4 = 8.27 x 11.69
    figure, ax = plt.subplots(len(columns), 1, sharex=sharex, tight_layout=True)

    for num, column in enumerate(columns):
        p = ax[num]
        p.plot(data.loc[:, column], **plot_args)

        p.set_title(column)
        p.grid(False)

    figure.align_ylabels()

    if save_fig_path is not None:
        plt.savefig(save_fig_path, format='png')
        plt.close(figure)
    else:
        plt.show()


def plotPie(dataFrame) -> None:
    labels = dataFrame.astype('category').cat.categories.tolist()
    counts = dataFrame.value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)  # autopct is show the % on plot
    ax1.axis('equal')
    plt.show()