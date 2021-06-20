import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import patches


class airoplot():
    def __init__(self, plot_type, data, dep_var, save_figs=False):

        self.plot_type = plot_type
        self.data = data
        self.dep_var = dep_var
        self.fig = plt.figure(constrained_layout=True,
                              figsize=np.array((12, 12)) * 0.75)

    def plot_findings(self, dep_var, save_figs=None):
        """Generate the figure which holds the subplots for:
            - Empirical Data (Group means, individual means and 95%-CI)
            - Estimation results (Posterior distribution and 95%-HPD)
            - Sequential Bayes Factor

        Parameters
        dep_var: string
            Columns name represeting the dependent variable to plot the
            data for.
        save_figs: bool
            An argument to override the value defined by the AIR object, for
              example if you want to refrain from saving the plots during an
              interactive session. Default is None.

        Returns
            None
        """

        fig = plt.figure(constrained_layout=True,
                         figsize=np.array((12, 12)) * 0.75)

        fig.tight_layout()

        # Configure subplots relative sizes
        grid_spec = fig.add_gridspec(6, 5)  # Y, X
        ax1 = fig.add_subplot(grid_spec[0:2, :])  # All of the 1st row
        ax2 = fig.add_subplot(grid_spec[2:4, :])  # 2/3 of the 2nd row
        ax3 = fig.add_subplot(grid_spec[4:, :])  # Remaining 1/3 of the last row
        # ax4 = fig.add_subplot(grid_spec[4:, 3 :])  # Last row, 2/3
        # ax5 = fig.add_subplot(grid_spec[2:4:, -2:]) # A place for the legend
        self.__gen_estimation_plot__(dep_var, ax2)
        self.__gen_empirical_plot__(dep_var, ax1, ax2)
        self.__gen_squential_plot__(dep_var, ax3)
        # self.__gen_trace_plot__(dep_var, ax4)
        # self.__gen_legend_plot__(ax5, ax2)

        if self.save_figs or save_figs is not None:
            fig.savefig('./figs/composite_{}_filtered_{}.png'.format(
                dep_var, self.filters_applied),
                quality=95, dpi=350)

    def __run_anova__(self, dep_var):
        """Specify (DV) and fit model (Repeated Measures ANOVA)"""
        return self.pyr.anova_within(self.data, dep_var, 'condition',
                                     'participant')

    def __run_t_tests__(self, dep_var):
        """Specify (DV) and run a series of paired-sample t-tests comparing to Baseline"""

        return {
            name: self.pyr.t_test(
                y=self.aggd_data.loc[
                    self.aggd_data['condition'] == 'Baseline', dep_var].values,
                x=self.aggd_data.loc[
                    self.aggd_data['condition'] == name, dep_var].values,
                paired=True,
                null_interval=self.hypotheses_dict['null_intervals'][name],
                tail=self.hypotheses_dict['tails'][name],
                iters_num=self.iters_num)
            for name in self.aggd_data.loc[
                self.aggd_data['condition'] != 'Baseline', 'condition'].unique()
        }

    def __gen_empirical_plot__(self, dep_var, plot_ax, other_plot):

        # Include violinplot
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
                      capsize=.3, errwidth=1.2, linestyles=["--"], label=None
                      )
        # Show each observation on a scatterplot
        sns.swarmplot(x=dep_var, y="condition",
                      data=self.contrasts_data, dodge=False,
                      zorder=1, ax=plot_ax, palette=self.figs_palette,
                      alpha=0.75)

        plot_ax.errorbar(
            x=self.results[dep_var]['margins']['y'].values,
            y=self.results[dep_var]['margins']['condition'].values,
            xerr=self.results[dep_var]['margins']['error'].values,
            ecolor='black', color=None, fmt='none', capsize=5,
            elinewidth=2, markeredgewidth=2)

        plot_ax.set_xlabel('{}'.format({
                                           'rt': 'RT Change (ms)',
                                           'er': 'Error Rate Change (%)'}[
                                           dep_var]))

        plot_ax.set_xlim(self.contrasts_data[dep_var].describe()[
                             ['min', 'max']].values + np.array([-1, 1]))
        plot_ax.set_ylabel('Contrast')
        plot_ax.set_title('(A)')
        plot_ax.set_yticks([])

        plot_ax.legend().remove()  # remove the current labels
        handles, labels = other_plot.get_legend_handles_labels()
        plot_ax.legend(handles, labels, title="Legend (contrast)",
                       title_fontsize='x-large', fontsize='large',
                       facecolor=None,
                       bbox_to_anchor=(0.5, 1.5), loc='upper center',
                       fancybox=True, ncol=3)
        other_plot.legend().remove()

        return plot_ax

    def __gen_estimation_plot__(self, dep_var, plot_ax):

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
                         color=color, label="Baseline vs. {}".format(k))

            # grab the peak of the KDE by extracting the last drawn line
            child = plot_ax.get_lines()[-1]
            _, y = child.get_data()
            plot_ax.plot(
                [np.percentile(vals, 2.5), np.percentile(vals, 97.5)],
                [y.max(), y.max()],
                label=None, color=color)

        plot_ax.set_xlabel('μ RT Change (ms)')
        plot_ax.set_ylabel('Posterior Density')
        plot_ax.set_title('(B)')

        return plot_ax

    def __gen_squential_plot__(self, dep_var, plot_ax):
        n = len(self.contrasts_data) // 3

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
        plot_ax.set_xticks([i for i in range(n) if (i % 5 == 0)])
        plot_ax.set_xlim([0, n + 1])
        # plot_ax.set_ylim(ymin=0)
        plot_ax.set_ylabel('Bayes Factor 1:0 (Log scale)')
        plot_ax.set_xlabel('N')
        plot_ax.legend().remove()
        plot_ax.set_title('(C)')
        plot_ax.set_yscale('log')

    def __gen_trace_plot__(self, dep_var, plot_ax):

        for k, color in zip(self.results[dep_var]['t_tests'].keys(),
                            self.colors):
            vals = self.results[dep_var]['t_tests'][k]['posterior']['mu'].values
            sns.lineplot(y=vals, x=range(len(vals)),
                         ax=plot_ax, alpha=0.6, palette=self.figs_palette,
                         color=color
                         )

        plot_ax.set_xlabel('Iterations')
        plot_ax.set_ylabel('Trace')
        plot_ax.set_title('(D)')
        plot_ax.set_xlim([0, len(vals)])

        return plot_ax

    def __gen_legend_plot__(self, plot_ax, other_plot):

        hand, leg = other_plot.get_legend_handles_labels()
        plot_ax.legend(hand, leg, title="Contrast", title_fontsize='xx-large',
                       fontsize='x-large', loc='center')
        plot_ax.set_axis_off()
        other_plot.legend().remove()

        leg = plot_ax.get_legend()
        [leg.legendHandles[i].set_color(self.colors[i]) for i in
         range(len(self.colors))]
