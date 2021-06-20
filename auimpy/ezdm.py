import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, patches
from scipy.special import logit

from constants import *

sns.set_context('paper')
sns.set_style("white")
plt.rcParams.update({'font.family': 'Times New Roman'})

"""Implementation of the EZ-DM Drift Diffusion model and validation of the
    model output."""


def ez_diffusion(proportion_correct, var_rt, mean_rt, edge_correction_n,
                 scale_param=0.1):
    """A Python implementation of the EZ-Diffusion model
    (Wagenmakers, Van der Maas & Grasman, 2007), based on the R code from the
    same paper.

    Parameters
    ---------
        proportion_correct: [Float, np.array]
            Proportion of correct responses.
        var_rt: [Float, np.array]
            Variance of response time (in seconds).
        mean_rt: [Float, np.array]
            Mean of response time (in seconds).
        edge_correction_n: int
            Number of trials per condition. Used to correct when a perfect
            accuracy is encountered (i.e., proportion_correct is 1).
        scale_param: float, optional
            Default is 0.1.

    Returns
    -------
        v: float
            Drift rate (rate of information accumulation).
        a: float
            Boundary separation (the level of information needed to reach
            a decision).
        t_er
            Non-decision time (time spent on non-decision related processes,
            such as motor planning and execution).
        mean_decision_time
            Time spent at reaching decision.
    """

    # Correct for excellent performance (logit(1) is infinite).
    proportion_correct[proportion_correct == 1] = 1 - 1 / (2 *
                                                           edge_correction_n)
    # Correct for performance at change level or below.
    proportion_correct[proportion_correct <= 0.5] = np.nan

    s_sq = scale_param ** 2

    logit_pc = logit(proportion_correct)
    x = logit_pc * (
            logit_pc * proportion_correct ** 2
            - logit_pc * proportion_correct
            + proportion_correct - 0.5) / var_rt

    # Drift rate
    v = np.sign(proportion_correct - 0.5) * scale_param * x ** 0.25
    # Boundary separation
    a = s_sq * logit(proportion_correct) / v
    # Mean decision time
    y = -v * a / s_sq
    mean_decision_time = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
    # Non decision time
    t_er = mean_rt - mean_decision_time

    return v, a, t_er, mean_decision_time


def validate_code():
    """A function to Validate the implementation of the code in Python.
    It produces the same results as shown in the original paper
    (Wagenmakers, Van Der Maas & Grasman, 2007).
    """

    # Table 1
    mean_rt = np.array([0.517, 0.467, 0.422, 0.372])
    var_rt = np.array([0.024, 0.024, 0.009, 0.009])
    proportion_correct = np.array([0.953, 0.953, 0.881, 0.881])
    participants = ['George', 'Rich', 'Amy', 'Mark']

    tbl1 = pd.DataFrame(np.array((mean_rt, var_rt, proportion_correct)).T,
                        columns=['Mean RT', 'RT Variance', 'Percent Correct'],
                        index=participants)

    print('Table 1 (Wagenmakers et al., 2007)')
    print(tbl1)
    # The printed results should be:
    #            Mean RT  RT Variance  Percent Correct
    # George    0.517        0.024            0.953
    # Rich      0.467        0.024            0.953
    # Amy       0.422        0.009            0.881
    # Mark      0.372        0.009            0.881

    model_results = ez_diffusion(
        *tbl1[['Percent Correct', 'RT Variance', 'Mean RT']].values.T,
        edge_correction_n=100)
    # edge_correction_n was set arbitrarily to be 100 as it is unspecified
    # in the paper and irrelevant to begin with, as there is no case
    # of perfect performance (consult tbl1).

    tbl2 = pd.DataFrame(
        np.array(model_results).T,
        columns=['Drift Rate', 'Boundary Seperation', 'Nondecision Time',
                 'Mean Decision Time'],
        index=participants)

    print('Table 2 (Wagenmakers et al., 2007)')
    print(tbl2.drop(columns=['Mean Decision Time']).round(2))
    # The printed results should be:
    #         Drift Rate  Boundary Seperation  Nondecision Time
    # George        0.25                 0.12              0.30
    # Rich          0.25                 0.12              0.25
    # Amy           0.25                 0.08              0.30
    # Mark          0.25                 0.08              0.25


def plot_diffusion(aggd_data, contrasts_data, raw_data,
                   filters_applied, save_figs=False):
    fig = plt.figure(figsize=(7, 7))

    # Main gridspec
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.01)
    # Secondary gridspecs
    gs0 = gridspec.GridSpecFromSubplotSpec(8, 6, subplot_spec=gs[0],
                                           hspace=0.3)
    gs1 = gridspec.GridSpecFromSubplotSpec(8, 6, subplot_spec=gs[1],
                                           hspace=0.3)
    gs2 = gridspec.GridSpecFromSubplotSpec(8, 6, subplot_spec=gs[2],
                                           hspace=0.3)
    gs_x = [gs0, gs1, gs2]

    _format_diffusion_plot_axes(fig)

    max_a = np.round(aggd_data.groupby('condition'
                                       )[
                         'boundary_separation'].mean().max() + 0.005, 3)

    # Fixing the boundary seperations to Baseline.
    boundary_on_baseline = aggd_data.loc[
        aggd_data['condition'] == 'Baseline', 'boundary_separation'
    ].mean()

    for cond_name, subplot, color in zip(
            contrasts_data['condition'].unique(), gs_x, COLORS):

        # As initial values
        cor_hist, main_plot, incor_hist = [None, None, None]

        for cond, c, b in zip(
                [cond_name, 'Baseline'],
                [color, 'black'], [False, True]):
            leftside_unbound = (
                    cond_name == "Compatible"  # First iteration of i
                    and cond == "Baseline")  # Second iteration of j

            v, a, ter, mean_decision_time = (
                aggd_data.loc[
                    aggd_data['condition'] == cond,
                    ['drift_rate', 'boundary_separation',
                     'non_decision_time',
                     'mean_decision_time']].mean().values)

            delta = (boundary_on_baseline - a)

            # Reassign instead of initial values
            cor_hist, main_plot, incor_hist = _draw_diffusion_plot(
                fig, subplot,
                raw_data.loc[raw_data['condition'] == cond],
                cond, c, v, a, ter,
                mean_decision_time,
                cor_hist=cor_hist, main_plot=main_plot,
                incor_hist=incor_hist,
                baseline=b, max_a=max_a, leftside_unbound=leftside_unbound,
                starting_point_delta=delta)

    # fig.tight_layout()

    if save_figs:
        fig.savefig(
            'output/figs/prepair/model_output - filters applied {}.png'.format(
                filters_applied), dpi=350, quality=95)


def _format_diffusion_plot_axes(fig):
    """Helper function to beautify """
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


def _create_diffusion_plots_subplots(fig, subplot):
    """Helper function to create subplots within a subplot."""
    return (
        fig.add_subplot(subplot[0:4, :]),
        fig.add_subplot(subplot[4:7, :]),
        fig.add_subplot(subplot[7:8, :])
    )


def _draw_diffusion_plot(
        fig, subplot, condition, cond_name, color, v, a, ter,
        mean_decision_time, baseline=False, cor_hist=None, main_plot=None,
        incor_hist=None, max_a=0.2, leftside_unbound=True,
        starting_point_delta=0):
    """
    Helper function to draw the fitted parameters of the ez diffusion
    model overlaid on the empirical distribution of response speed and
    error rate.

    Parameters
    ----------
    fig plt.figure.Figure
        A figure to draw the plot in.
    subplot: plt.subplots
        The current subplot in the figure.
    condition: pd.DataFrame
        Raw data. Must contain the columns 'rt' (in ms) and 'correct' (bool).
    cond_name: str
        For plot title.
    color: str
        A color name accepted by matplotlib.
    v: float
        Drift rate parameter (see ezdm module).
    a: float
        Boundary Separation parameter (see ezdm module).
    ter: float
        Non-Decision Time (see ezdm module).
    mean_decision_time: float
        Decision time out of response time (se ezdm module).
    baseline: bool
        Determines whether the data came from the baseline condition. Used as
        the baseline condition is overlaid on top of the other conditions
        as comparison.
    leftside_unbound: bool
        Determines whether the plot is located on the leftmost column of the
        figures subplots, as these contain y-axis labels and ticks, for example.
    starting_point_delta: float, optional
        Given a delta for the difference between the boundary of the currently
        plotted condition and the Baseline condition.

    Returns
    -------
    None.

    """

    if starting_point_delta != 0:
        a += starting_point_delta
    else:  # In case we enter the function with a Baseline condition and non-0𐤃
        starting_point_delta = 0

    if not baseline:
        cor_hist, main_plot, incor_hist = _create_diffusion_plots_subplots(
            fig, subplot)

    for (_n, g), ax in zip(condition.groupby('correct'),
                           (incor_hist, cor_hist)):
        vals = g.loc[(g['rt'] >= 0) & (g['rt'] <= SLOW_RT)].dropna()['rt']
        ax.hist(vals, bins=30, edgecolor=color, alpha=0.5,
                facecolor=[color, 'lightgrey'][baseline])
        main_plot.plot([ter * 1000, vals.mean()],
                       [a / 2, [0 + starting_point_delta / 2,
                                a - starting_point_delta / 2][_n]],
                       color=color, linewidth=1.5,
                       alpha=0.5)
        main_plot.plot([vals.mean() - 100, vals.mean() + 100],
                       np.repeat([0 + starting_point_delta / 2,
                                  a - starting_point_delta / 2][_n], 2),
                       color=color, linewidth=1.5,
                       alpha=0.5)

    if starting_point_delta is None or baseline:
        main_plot.axhline(y=a / 2, linewidth=1,
                          alpha=0.5, color=color,
                          linestyle=[':', '--'][baseline])
        main_plot.axhline(y=a,
                          linewidth=1, linestyle=[':', '--'][baseline],
                          alpha=0.5, color=color)

    if cond_name == "Compatible":
        for g, t in zip([cor_hist, main_plot, incor_hist], 'ABC'):
            g.annotate(f'({t})', (0.05, {'C': 0.4}.get(t, 0.825)),
                       fontsize=12, xycoords='axes fraction')
        main_plot.add_patch(
            patches.Rectangle((0, a * 0.3),
                              ter * 1000, a * 0.2,
                              linewidth=1, edgecolor='grey',
                              facecolor='grey', alpha=0.4))
        main_plot.annotate('Ter', (ter * 500, max_a * 0.35), fontsize=12,
                           ha='center')

    if cond_name == "Incompatible":
        main_plot.text(ter * 1000 + 100, a * 0.9,
                       DEP_VAR_LABEL_DICT['drift_rate'], ha='right',
                       rotation=0,
                       fontsize=10)

    if cond_name == "Irrelevant":
        main_plot.annotate('Correct\n Response', xy=[800, a],
                           xytext=[800, 0.6 * a], ha='center',
                           fontsize=10, color='k',
                           arrowprops=dict(facecolor='black', shrink=0.05,
                                           width=4, headwidth=12))
        main_plot.annotate('Incorrect\n Response', xy=[800, 0],
                           xytext=[800, 0.3 * a], ha='center',
                           fontsize=10, color='k',
                           arrowprops=dict(facecolor='black', shrink=0.05,
                                           width=4, headwidth=12))

    if baseline:
        main_plot.set(ylim=[0, max_a], xlim=[0, 1000], xticks=[],
                      ylabel=['', 'Boundary\nSeperation (a)'][leftside_unbound])
        if not leftside_unbound:
            main_plot.set(yticks=[], yticklabels=[])

        cor_hist.set(
            ylim=[0, 400], xticks=[],
            ylabel=['', 'Correct Response\n Density'][leftside_unbound])
        if not leftside_unbound:
            cor_hist.set(yticks=[], yticklabels=[], ylim=[0, 400])
        cor_hist.grid(False)

        incor_hist.set(
            xlabel='RT (ms)',
            ylabel=['', 'Incorrect Response\n Density'][leftside_unbound],
            ylim=[0, 100], xlim=[0, 1000],
            xticks=range(100, 1100, 200),
            xticklabels=range(100, 1100, 200))
        if not leftside_unbound:
            incor_hist.set(yticks=[], yticklabels=[], ylim=[0, 100])
        incor_hist.invert_yaxis()
        incor_hist.grid(False)

    if not baseline:
        cor_hist.set_title(cond_name, fontsize=12)

    main_plot.grid(False)

    return cor_hist, main_plot, incor_hist
