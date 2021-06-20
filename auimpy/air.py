# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:58:11 2019

@author: Eitan Hemed
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import patches

import pyrio
from constants import *

sns.set_context('paper')
sns.set_style("white")
plt.rcParams.update({'font.family': 'Times New Roman'})


def _gen_hypotheses_dict(dep_var):
    """Define null interval (H0 predicts) for the Bayesian analysis
       and whether contrasts should be one-or two-tailed including
       direction of hypothesis (for frequentist analysis) and
    """
    d = {
        'null_intervals': {
            'Incompatible': [0, np.inf],
            'Compatible': [-np.inf, 0],
            'Irrelevant': [-np.inf, np.inf]},
        'tails': {
            'Incompatible': 'greater',
            'Compatible': 'less',
            'Irrelevant': 'two.sided'}
    }

    # We have to reverse the hypotheses for drift rate as it is the opposite
    # of all other dependent variables. For example, we might predict lower
    # RT on the Compatible condition, but also greater drift rate, not
    # lesser drift rate.

    if dep_var == "drift_rate":
        for k in d.keys():
            d[k]['Compatible'], d[k]['Incompatible'] = (
                d[k]['Incompatible'], d[k]['Compatible'])

    return d


class AIR:
    # AIR - Automatic Imitation Report
    """The purpose of this class is generating a publication-ready report
    of the data for each one of several Automatic Imitation experiments.
    """

    def __init__(self,
                 agg_data,
                 contrasts_data,
                 agg_data2,
                 paper_name=None,
                 filters_applied=True,
                 save_figs=False,
                 write_text=False,
                 iters_num=10000,
                 **kwargs
                 ):
        """
        Initialize an AIR instance.

        Parameters
        ----------
        agg_data
        contrasts_data
        agg_data2
        standard_hypotheses
        paper_name
        filters_applied
        save_figs: bool
        informed_prior
        write_text
        figs_palette
        iters_num
        kwargs
        """
        self.agg_data = agg_data
        self.contrasts_data = contrasts_data
        self.agg_data2 = agg_data2
        self.save_figs = save_figs
        self.filters_applied = filters_applied
        self.write_text = write_text
        self.iters_num = iters_num
        self.colors = COLORS
        self.paper_name = paper_name
        self.figs_palette = sns.set_palette(sns.color_palette(self.colors))
        self.pyr = pyrio.PyRIO()

        stub = f"./output/texts/air_reports_filters_applied_{self.filters_applied}.txt"
        if os.path.exists(stub):
            with open(stub, 'w') as f:
                f.write('')

        self.results = self.analyze()
        self._save_contrasts()

    def analyze(self):
        """Run an analysis of Anova and matching T-tests for all
        dependent variables.
        """
        return dict(
            zip(DEP_VARS, [dict(
                zip(['anova_freq', 'margins', 't_tests'],
                    [*self.pyr.anova_within(self.contrasts_data,
                                            dep_var=dep_var,
                                            ind_var='condition',
                                            subject_id='participant'),
                     self._run_t_tests(dep_var)],
                    )) for dep_var in
                DEP_VARS]))

    def report_results(self, dep_var):
        """Report results for ANOVA, t-tests and plot data and estimates"""

        s = "\n".join([si for si in
                       [self.report_anova_for_dv(
                           dep_var)] + self.report_t_tests(
                           dep_var)])

        if self.write_text:
            with open("output/texts/air_reports_filters_applied_{}.txt".format(
                    self.filters_applied), "a", encoding='utf-8') as f:
                f.write(
                    'ANALYSIS REPORT FOR {} DATA - {}: \n***\n'.format(
                        {False: "UNFILTERED", True: "FILTERED"}[
                            self.filters_applied],
                        DEP_VAR_LABEL_DICT[dep_var]
                    )
                )
                f.write(s)
                f.write("\n***\n")
        print(s)
        self.plot_findings(dep_var)

    def report_anova_for_dv(self, dep_var):

        aov_res = self.results[dep_var]['anova_freq'].iloc[0].to_dict()
        return "Repeated-Measures ANOVA showed {}".format(
            self._report_anova_term(aov_res, dep_var))

    def _report_anova_term(self, aov_res, dep_var):
        """Format ANOVA report, returned as string"""

        aov_str = (
            "F({DF1:.0f}, {DF2:.0f}) = {F:.2f}, "
            "p {operator} {pvalue:.3f}, "
            "Partial-\u03B7 Squared = {pes:.3f}".format(
                **aov_res,
                operator={False: '=', True: '<'}[
                    aov_res['pvalue'] < 0.001])).replace('< 0.000', '< 0.001')

        return (
            '{sig} effect of '
            'experimental condition on {var} [{aov_str}]'.format(
                sig=np.where(
                    aov_res['pvalue'] < .05,
                    'a significant',
                    'an insignificant'
                ),
                var=dict(zip(DEP_VARS, TITALIZED_DEP_VARS))[dep_var],
                aov_str=aov_str)
        )

    def report_t_tests(self, dep_var):
        """Compare each condition to Baseline using one-sample t-test.
        Return results as list of formatted strings."""
        return [", ".join([
            ('The {cond} condition {alternative} t({parameter:.0f}) = '
             '{statistic:.2f}, p = {pvalue:.3f}, {dep_var} Change '
             + {'rt': '{estimate:.2f}', 'er': '{estimate:.2f}'
                }.get(dep_var, '{estimate:.4f}') + '{units}').format(
                **res['freq'].iloc[0].to_dict(),
                # Experimental condition
                cond=contrast,
                # label dep_var
                dep_var=dict(zip(DEP_VARS, TITALIZED_DEP_VARS))[dep_var],
                # label units of dep_var
                units=dict(zip(DEP_VARS, UNITS))[dep_var]
            ).replace('p = 0.000', 'p < 0.001'),
            # to handle the p value formatting
            'Cohen\'s d = {cohen:.2f} 95%-CI [{lower:.2f}, {upper:.2f}], '
            'BF1:0 = {bf:.3f}'.format(
                # Cohen's d (Standardized mean)
                **res['cohen'],
                # Bayes factor of comparison
                bf=res['BF'])
        ]) for contrast, res in self.results[dep_var]['t_tests'].items()]

    def plot_findings(self, dep_var, save_figs=False):
        """Generate the figure which holds the subplots for:
            - Empirical Data (Group means, individual means and 95%-CI)
            - Estimation results (Posterior distribution and 95%-HPD)
            - Sequential Bayes Factor analysis
        and call the matching functions for plotting each of these.

        Parameters
        ----------
        dep_var: str
            Name of the dependent variable to be analyzed ('rt', 'er' or
            'drift_rate').
        save_figs: bool
            Whether to save the generated figures (True) or not (False).
            Uses either the default argument or the one defined for the
            AIR instance.

        Returns
        -------
        None.
        """
        # TODO - use a 6-in-a-row subplot to get the correct proportions in
        #  gridspec, making the sequential BF plot slimmer
        fig, axs = plt.subplots(3, 2,
                                figsize=np.array((8.5, 5)),
                                gridspec_kw={'height_ratios': [6, 0.5, 6]})
        gs = axs[1, 1].get_gridspec()
        axs[1, 0].remove()
        leg_ax = fig.add_subplot(gs[1, 0:])
        axs[1, 1].remove()
        axs[-1, -1].set_title('(D)', loc='left')
        axs[-1, -1].set_axis_off()

        self._draw_posterior_distribution(dep_var, axs.flat[1])
        self._draw_contrasts(dep_var, axs.flat[0])
        self._draw_sequential_bf(dep_var, axs.flat[-2])
        self._add_legend(leg_ax, axs.flat[1])

        fig.tight_layout(pad=0.4, w_pad=0.55, h_pad=0.5)

        if self.save_figs or save_figs:
            fig.savefig('./output/figs/air/filters applied - {}/{}.png'.format(
                self.filters_applied, dep_var),
                quality=95, dpi=350)

    def _run_anova(self, dep_var):
        """Specify (DV) and fit model (Repeated Measures ANOVA)"""
        return self.pyr.anova_within(self.data, dep_var, 'condition',
                                     'participant')

    def _run_t_tests(self, dep_var):
        """Specify (DV) and run a series of paired-sample t-tests comparing to
         Baseline"""

        self.hypotheses_dict = _gen_hypotheses_dict(dep_var)

        return {
            name: self.pyr.t_test(
                y=self.agg_data.loc[
                    self.agg_data['condition'] == 'Baseline', dep_var].values,
                x=self.agg_data.loc[
                    self.agg_data['condition'] == name, dep_var].values,
                paired=True,
                null_interval=self.hypotheses_dict['null_intervals'][name],
                tail=self.hypotheses_dict['tails'][name],
                iters_num=self.iters_num)
            for name in self.agg_data.loc[
                self.agg_data['condition'] != 'Baseline', 'condition'].unique()
        }

    def _draw_contrasts(self, dep_var, plot_ax):

        # Include violin plot
        sns.violinplot(x=dep_var, y="condition",
                       data=self.contrasts_data,  # palette=self.figs_palette,
                       ax=plot_ax, inner=None, color='lavender',
                       )
        # Show the conditional means and 95%-CI
        sns.pointplot(x=dep_var, y="condition",
                      data=self.contrasts_data, units='participant',
                      dodge=.532, join=False, color='black',
                      # palette=self.figs_palette,
                      markers="d", scale=.75, ci=None, ax=plot_ax, alpha=0.5,
                      capsize=0.3, errwidth=1.2, linestyles=["--"], label=None,
                      )
        # Show each observation on a scatterplot
        sns.swarmplot(x=dep_var, y="condition",
                      data=self.contrasts_data, dodge=False,
                      zorder=1, ax=plot_ax, palette=self.figs_palette,
                      alpha=0.75)
        # Draw a 95% CI based on the margins acquired from the anova model
        plot_ax.errorbar(
            x=self.results[dep_var]['margins']['emmean'].values,
            y=self.results[dep_var]['margins']['condition'].values,
            xerr=self.results[dep_var]['margins']['SE'].values,
            ecolor='black', color=None, fmt='none', capsize=5,
            elinewidth=2, markeredgewidth=2)

        plot_ax.set_xlabel('{} Change ({})'.format(
            dict(zip(DEP_VARS, TITALIZED_DEP_VARS))[dep_var],
            dict(zip(DEP_VARS, UNITS))[dep_var]))

        plot_ax.set_xlim(self.contrasts_data[dep_var].describe()[
                             ['min',
                              'max']].values * 1.125)
        plot_ax.set_ylabel('Contrast')
        plot_ax.set_title('(A)', loc='left')
        plot_ax.set_yticks([])

        return plot_ax

    def _draw_posterior_distribution(self, dep_var, plot_ax):

        max_val = max([
            np.histogram(
                self.results[dep_var]['t_tests'][cond]['posterior'][
                    'mu'].values,
                density=True, bins=self.iters_num)[0].max()
            for cond in self.contrasts_data['condition'].unique()])

        for k, height, color in zip(
                self.results[dep_var]['t_tests'].keys(),
                max_val * np.linspace(0.15, 0.25, 3),
                self.colors
        ):
            vals = np.sort(
                self.results[dep_var]['t_tests'][k]['posterior']['mu'].values)

            sns.distplot(vals, ax=plot_ax,
                         color=color, label=k)

            # grab the peak of the KDE by extracting the last drawn line
            child = plot_ax.get_lines()[-1]
            _, y = child.get_data()
            plot_ax.plot(
                [np.percentile(vals, 2.5), np.percentile(vals, 97.5)],
                [y.max(), y.max()],
                label=None, color=color)

        plot_ax.set_xlabel(
            'μ {} Change ({})'.format(
                dict(zip(DEP_VARS, TITALIZED_DEP_VARS))[dep_var],
                dict(zip(DEP_VARS, UNITS))[dep_var]))
        plot_ax.set_ylabel('Posterior Density')
        plot_ax.set_title('(B)', loc='left')

        return plot_ax

    def _draw_sequential_bf(self, dep_var, plot_ax):
        n = self.contrasts_data.shape[0] // 3

        # Inconclusiveness zone
        rect = patches.Rectangle((0, 1 / 3), n + 1, (3 - 1 / 3), linewidth=1,
                                 edgecolor='none', facecolor='lightgrey',
                                 alpha=0.35)
        plot_ax.add_patch(rect)

        for k, color in zip(
                self.results[dep_var]['t_tests'].keys(),
                self.colors
        ):
            sns.lineplot(data=self.results[dep_var]['t_tests'][k][
                'sequential_BF'].dropna(),
                         x='N', y='BF', label=False, ax=plot_ax, color=color,
                         alpha=0.5, linewidth=4)

            sns.scatterplot(data=self.results[dep_var]['t_tests'][k][
                'sequential_BF'].dropna(),
                            x='N', y='BF', label=False, ax=plot_ax,
                            color='black', marker='>', alpha=1,
                            edgecolors=['black'], linewidths=8)

        plot_ax.axhline(3, c='black', alpha=0.35, linestyle=(0, (1, 1)))
        plot_ax.axhline(0.33, c='black', alpha=0.35, linestyle=(0, (1, 1)))

        # Generate ticks every 5 participants
        plot_ax.set(
            xticks=[i for i in range(n) if i % 10 == 0],
            xlim=[0, n + 1],
            xlabel='N')
        plot_ax.set_ylabel('Bayes Factor 1:0 (Log Scale)', ha='center')
        plot_ax.legend().remove()
        plot_ax.set_title('(C)', loc='left')
        plot_ax.set_yscale('log')

    def _add_legend(self, legend_ax, source_ax):
        handles, labels = source_ax.get_legend_handles_labels()
        legend_ax.legend(
            handles, labels, title="Contrasts vs. Baseline",
            facecolor=None,
            bbox_to_anchor=(0.5, 1.4), loc='upper center',
            fancybox=True, ncol=3)
        legend_ax.set_axis_off()

    def plot_rt_by_fingers(self):

        with plt.rc_context({'font.size': 20, 'font.weight': 'bold',
                             'font.family': 'Times New Roman'}):
            sns.set_palette(sns.color_palette(
                ['grey', 'dodgerblue', 'darkred', 'orange']
            ))

            fig, axs = plt.subplots(1, 2, figsize=np.array((12, 6)) * 0.7,
                                    sharey=True)

            for (name, group), ax in zip(self.agg_data2.groupby('cuenumber'),
                                         axs.flat):
                sns.violinplot(y='rt', x="condition",
                               data=group,
                               ax=ax, inner=None, color='lavender',
                               )
                sns.pointplot(x='condition', y='rt', color='black',
                              data=group, n_boot=10000, ax=ax, join=False,
                              capsize=.1, markers="_")
                sns.swarmplot(x='condition', y='rt', hue='condition',
                              data=group,
                              label='name',
                              ax=ax, alpha=0.5)
                ax.set_title(name, fontsize=16)

                ax.legend().remove()
                ax.set(ylabel='', xlabel='',
                       xticklabels=[f'{i}\n{m}\n({sd})'
                                    for i, m, sd in
                                    group.groupby('condition')['rt'].agg(
                                        ['mean', 'std']).round(
                                        2).reset_index().values],
                       yticklabels=range(300, 750, 50)
                       )

            axs.flat[0].set_ylabel('Mean Response Time (ms)')

            fig.tight_layout(w_pad=0.005)

            if self.save_figs:
                fig.savefig(
                    f'./output/figs/air/filters applied - {self.filters_applied}/RT By Fingers.png',
                    quality=95, dpi=350)

    def _save_contrasts(self):
        pickle.dump(self.contrasts_data, open(
            './output/pickles/contrasts_filters_applied_'
            f'{self.filters_applied}.p', 'wb'))
