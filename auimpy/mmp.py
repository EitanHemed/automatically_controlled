"""A module containing the functions and definitions to produce the plot of
the EZ-dm model results on all experiments.
"""
import sys
import glob
import re
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from constants import *

sns.set_context('paper')
sns.set_style("white")
plt.rcParams.update({'font.family': 'Times New Roman'})


TO_GRAB_COLS = ['participant', 'condition'] + MODEL_COLS
YTITLES = dict(zip(MODEL_COLS,
                   ['Drift Rate (ν)', 'Boundary Separation (a)',
                    'Non-Decision Time (S)']))

sns.set_palette(sns.color_palette(COLORS))


def plot_findings(filtered_data, save_fig):
    """"Run the plotting routine for the EZ-dm findings.

    Parameters
    ----------
    filtered_data: bool
        whether to load data files that went through extensive (True) or basic
        (False) screening of outlier participants and responses. See
        prepair.PrepAIR for details.
    save_fig: bool
        whether to save the generated figure (True) or not (False).

    Returns
    ------
    None.
    """
    df = _grab_exps(filtered_data)
    fig = _create_fig(df)
    if save_fig:
        fig.savefig(
            f'./plots/MMP_filtered_data_{filtered_data}.png',
            quality=95, dpi=350)


def _grab_exps(filtered_data=True):
    """Reads the pickle object of the dataframe for each experiment, returning
    the data to be plotted.

    Parameters
    ----------
    filtered_data: bool
        whether to load data files that went through extensive (True) or basic
        (False) screening of outlier participants and responses. See
        prepair.PrepAIR for details.

    Returns
    -------
    pd.DataFrame
        Containing the data to be plotted, identified by participant,
        condition and experiment name.
    """
    objs = glob.glob('../exp*/output/pickles/contrasts_filters_applied_'
                     f'{filtered_data}.p')

    try:
        return pd.concat([pd.read_pickle(o).assign(
            paper_name=re.search(r'exp.+?(?=\\)', o).group().replace(
                'exp', 'Experiment '))
            for o in objs])
    except Exception as e:
        print("Make sure to have all pickle files available by running the "
              "separate notebooks first.")
        raise(e)

def _create_fig(df):
    """

    Parameters
    ----------
    df: pd.DataFrame
        a dataframe containing the results of the EZ-dm model fitting for each
        participant, on the different experimental conditions. See air.AIR.

    Returns
    -------
    fig: plt.figure
        A figure of the plotted data.
    """

    fig, axs = plt.subplots(4, 3, figsize=(15, 15), sharey='col')
    # Loop over experiments and reocvered model paramters to plot the contrasts
    for (_, exper_df), axs_row in zip(
            df.groupby('paper_name'), axs):
        for var_name, ax in zip(MODEL_COLS, axs_row.flatten()):
            _plot(exper_df, var_name, ax)
    _finalize_fig(axs, fig)
    #fig.tight_layout()
    return fig


def _plot(df, var, ax):
    """

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe of the results of model fitting on a specific experiment.
    var: str
        Name of model parameter to be plotted.
    ax: matplotlib.axes
        Axes object for plotting.


    Returns
    -------
    None
    """
    ax.set_title(YTITLES[var], fontsize=16)
    # Plot mean of group
    sns.pointplot(data=df, x='condition',
                  y=var, ax=ax, join=False, color='black', capsize=.2)
    # Overlay scatterplot with non-overlapping data points
    sns.swarmplot(data=df, x='condition', y=var,
                  ax=ax, color=None, linewidth=1, alpha=0.5)
    # Increase font size
    for item in ([ax.yaxis.label, ax.xaxis.label] +
                 ax.get_yticklabels() + ax.get_xticklabels()):
        item.set_fontsize(19)
    ax.set_xticklabels([_anotate(vals[var])
                        for _, vals in df.groupby('condition')])
    ax.set_ylabel(df['paper_name'].iloc[0], fontsize=21)
    ax.axhline(0, color='darkgray', linestyle='--')


def _anotate(vals):
    """Runs a one-sample (2-tailed) against population mean of 0. The value
    is converted to a Cohen's d and returned with annotation of significance.

    Parameters
    ----------
    vals: np.array
        a 1-d np.array of the values obtained from the contrasts.

    Returns
    -------
    str
        Standardized effect size with significance test annotation.

    -------
    """
    t_test = sp.stats.ttest_1samp(vals, 0)
    cohen = (t_test.statistic / (len(vals) ** 0.5)).round(2)
    p_value_marker = np.where(t_test.pvalue >= .05,
                              '', np.where(t_test.pvalue >= 0.01, '*',
                                           np.where(t_test.pvalue >= 0.005,
                                                    '**',
                                                    '***')))
    return f'{cohen}{p_value_marker}'


def _finalize_fig(axs, fig):
    # Used mostly to remove the automatically generated axis labels.
    [ax.set_title('') for ax in axs.flatten()[3:]]
    [ax.set_xlabel('') for ax in axs.flatten()[:]]
    [ax.set_ylabel('') for ax in axs.T.flatten()[4:]]

    legend_elements = [Patch(
        facecolor=color, edgecolor='k', label=cond, alpha=0.5)
        for (cond, color) in zip(CONDS, COLORS)]

    axs.flatten()[1].legend(
        title="Contrasts vs. Baseline", ncol=3, loc='center',
        handles=legend_elements, fancybox=True,
        bbox_to_anchor=[0.5, 1.275], fontsize=15, title_fontsize=18)

    fig.subplots_adjust(wspace=0.275, hspace=0.15)