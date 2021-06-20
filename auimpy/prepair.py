# -*- coding: utf-8 -*-
"""
This module contains the `prepair` class used to prepare (i.e., pre-process)
the data for analysis.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ezdm
from constants import *

COLORS = ['limegreen', 'dodgerblue', 'darkred', 'orange']

sns.set_context('paper')
sns.set_style("white")
sns.set_context(font_scale=5)
plt.rcParams.update({'font.family': 'Times New Roman'})


class PrepAIR:
    """This purpose of this class is to filter is pre-processing of
    of the data for each one of several Automatic Imitation experiments.

        Parameters
        ----------
        data : pd.DataFrame
            A dataframe in a long format with the following columns:
            - 'participant': Some unique ID.
            - 'rt': response time (float).
            - 'er': error rate (float).
            - 'pc': percent correct(float)
            - 'ter': non-decision
            - 'condition': (str), values should be Compatible, Incompatible,
                Irrelevant or Baseline.
        input_method : str
            Either 'mouse' or 'keyboard'. Currently does nothing.
        filters_applied : bool
            Whether to apply maximal filtering of trials and participants.
            Impacts label of report and figures saved. Default is True.
        minimal_accuracy : int
            The threshold for filtering out participants with sub-par accuracy
            rate. Default is 60.
        write_text : bool
            Boolean, whether the textual report of data screening should be
            saved onto a txt file. Default is True.
        save_figs : bool
            Whether the generated figures should be saved.

        Returns
        -------
        PrepAIR object.

        Once created, PrepAIR usage should be used through the following
        functions:
            - Observe accuracy/rt performance, through
            `PrepAIR.plot_descriptives`.
            - Observe performance through `PrepAIR.report_invalid_trials`.
            - Return the summary and processed data used by the AIR object for
                analysis, through `prepair.finalize_data`.
    """

    def __init__(
            self,
            data,
            input_method,
            filters_applied=True,
            minimal_accuracy=60,
            write_text=True,
            save_figs=True,
            **kwargss
    ):
        self.data = data
        self.input_method = input_method
        self.minimal_accuracy = minimal_accuracy
        self.filters_applied = filters_applied
        self.write_text = write_text
        self.save_figs = save_figs
        if not self.filters_applied:
            self.minimal_accuracy = 50

        # Make sure the directory for saving the output exists.
        for i in ['figs/prepair', 'pickles', 'texts']:
            stub = f'./output/{i}'
            if not os.path.exists(stub):
                os.makedirs(stub)

        self._label_valid_trials()
        self._add_accuracy_columns()
        self._clean_data()
        self._run_ez_diffusion()
        self._format_data_for_air()

    def plot_descriptives(self, dep_var, by_condition=False, save_figs=None,
                          pre_filteration=False):
        """Generate a boxplot with individual means. Plot on X-Axis either mean
        RT or Accuracy. Possibly plot results separately for each condition.
        """

        fig, ax = plt.subplots(figsize=(7, 4.8))

        if by_condition:
            group_vars = ['participant', 'condition']
        else:
            group_vars = ['participant']

        if pre_filteration:
            data = self.data.copy()
        else:
            try:
                data = self.clean_data.copy()
            except AttributeError:
                self._clean_data()
                data = self.clean_data.copy()

        data = data.groupby(group_vars, as_index=False).mean()

        d_val = {True: 'condition', False: None}[by_condition]
        sns.boxplot(data=data, x=dep_var, y=d_val, ax=ax,
                    palette=sns.color_palette(COLORS))
        sns.swarmplot(data=data, x=dep_var, y=d_val,
                      size=3, color="orange", linewidth=1, ax=ax,
                      palette=sns.color_palette(COLORS))

        ax.set_xlabel(
            {
                'rt': 'Mean RT (ms)',
                'er': 'Error Rate (%)'}[dep_var])

        ax.set_xlim(
            {
                'rt': [0, 1000],
                'er': [0, 100]}[dep_var])

        if by_condition:
            ax.set_ylabel('Experimental Condition')

        if self.save_figs or save_figs:
            fig.savefig('./output/figs/prepair/{}_pre_filteration_{}_'
                        'by_cond_{}.png'.format(
                dep_var, pre_filteration, by_condition
            ), quality=95, dpi=350)

    def report_invalid_trials(self):
        """Reports the proportion of invalid trials based on several criteria.
        Invalid violate at least one of the following criteria:
        - Response is correct.
        - Response speed is between SLOW_RT and FAST_RT.
        - Keys were pressed down at the beginning of the trial.
        - Premature responses (< 0ms, used in case of minimally applied
            filters).
        - Participants with low response accuracy or a large proportion of
            invalid trials.
        """

        # The following calculates the percentage of trials answering
        # the criteria out of all data.

        total_size = self.data.shape[0]
        # from proportions to percentage

        d = {}

        # Incorrect response (no-response is also incorrect for that matter)
        d['Incorrect_Response'] = self.data[~self.data['correct']]
        d['Slow_Response'] = self.data[self.data['rt'] > SLOW_RT]
        d['Fast_Response'] = self.data[self.data['rt'] <
                                       [0, FAST_RT][self.filters_applied]]
        # Start position (if keys were pressed incorrectly at trial start)
        d['Start_Position'] = self.data[self.data['keys_at_start'] != {
            'mouse': "[1, 0, 1]"}.get(
            self.input_method, None)]
        # Sub-part participants
        d['Poor-performance participants'] =(
            self.data[
                np.where(
                    self.filters_applied,
                    # If we apply maximal filters to the data, we want to filter
                    # participants with less than X% valid trials (regardless of
                    # invalidating criteria).
                    self.data[
                        'percentage_valid_trials'] <= self.minimal_accuracy,
                    # In case we don't filter out based on percentage of validity,
                    # we still have to remove participants performing below chance
                    # on at least one of the conditions as they would cause the
                    # model fitting routine to falter.
                    self.data[
                        'maximal_error_rate_by_participant']
                    >= 50)]
        )

        for k, v in d.items():
            d[k] = np.round(100 * v.shape[0] / total_size, 2)

        # Percentage of filtered data
        d['Percentage_Filtered'] = np.round(
            100 - 100 * len(self.clean_data) / len(self.data), 2)

        # Additionally, we would filter out responses
        # occurring prior to cue presentation, even if we use lenient rather
        # than maximal filters.

        d['N_Poor_Performance_Participants'] = len(self.data.loc[np.where(
            self.filters_applied,
            self.data['percentage_valid_trials'] <= self.minimal_accuracy,
            self.data[
                'maximal_error_rate_by_participant'] >= 50), 'date'].unique()
                                                   )
        # Accuracy threshold
        d['Minimal_Accuracy'] = self.minimal_accuracy

        # Total number of participants - 'N'
        d['N'] = len(self.data['participant'].unique())
        d['Maximal_RT'] = SLOW_RT
        d['Minimal_RT'] = FAST_RT

        if not self.filters_applied:
            s = (
                'Invalid trials included '
                'incorrect or omitted responses ({Incorrect_Response}%) '
                'and trials with premature responses (<0ms). '
                'Finally, all data from '
                '{N_Poor_Performance_Participants:.0f} participants '
                'with below-chance accuracy (<{Minimal_Accuracy}%,'
                ' {Poor-performance participants:.2f}% of {N:.0f}) on at'
                ' least one of the conditions were removed.'
                ' Total filtration amounted '
                'to {Percentage_Filtered}%.'.format(
                    **d))
        else:
            s = (
                'Invalid trials included incorrect or omitted responses '
                '({Incorrect_Response}%), slow responses '
                '(>{Maximal_RT:.0f}ms, {Slow_Response}%) '
                'fast responses (<{Minimal_RT:.0f}ms, {Fast_Response}%), '
                'and trials in which participants did not press the keys '
                'down correctly in the beginning of the trial '
                '({Start_Position}%). '
                'Finally, all data from '
                '{N_Poor_Performance_Participants:.0f} participants '
                '(<{Minimal_Accuracy}% valid trials, '
                '{Poor-performance participants}% of {N:.0f}) were removed. '
                'Total filtration amounted to '
                '{Percentage_Filtered}%.'.format(**d))

        if self.write_text:
            with open(
                    "./output/texts/prepair_report_filtered_data_{}.txt".format(
                        self.filters_applied), "w+") as f:
                f.write('FILTERATION REPORT FOR {} DATA: \n***\n'.format(
                    {False: "UNFILTERED", True: "FILTERED"}[
                        self.filters_applied]))
                f.write(s)
                f.write("\n***\n")

        return s

    def _label_valid_trials(self):
        # Label valid trials on several criteria.

        if not self.filters_applied:
            self.data['valid_trial'] = (
                # Correct response (no-response is incorrect)
                    (self.data['correct']) &
                    # non-negative response time
                    (self.data['rt'] > 0))
        else:
            self.data['valid_trial'] = (
                # Correct response (no-response is incorrect)
                    (self.data['correct'])
                    # Slow responses
                    & (self.data['rt'] <= SLOW_RT)
                    # Fast responses
                    & (self.data['rt'] >= FAST_RT)
                    # Whether keys were pressed correctly when trial began
                    & (self.data['keys_at_start'] == {
                'mouse': "[1, 0, 1]",
                'keyboard': None}
            [self.input_method])
            )

    def _add_accuracy_columns(self):
        # Generate mean accuracy
        self.data['percentage_valid_trials'] = self.data.groupby(
            'participant')['valid_trial'].transform(np.mean).values * 100
        # By participant and condition error rate
        self.data['er'] = 100 - self.data.groupby(['participant', 'condition'])[
            'correct'].transform(np.mean).values * 100
        self.data['maximal_error_rate_by_participant'] = \
            self.data.groupby('participant')['er'].transform(np.max)

    def _clean_data(self):
        # Keep only valid trials and acceptable performance participants
        self.clean_data = self.data.loc[self.data['valid_trial']]
        if self.filters_applied:
            self.clean_data = self.clean_data.loc[
                self.clean_data[
                    'percentage_valid_trials'] > self.minimal_accuracy]
        else:
            self.clean_data = self.clean_data.loc[
                self.clean_data['maximal_error_rate_by_participant'] < 50]

    def _run_ez_diffusion(self):
        """Runs the ez-dm fitting. See ezdm.py for implementation.
        """
        model_input = self.clean_data.copy()

        model_input['proportion_correct'] = \
            model_input.groupby(['participant', 'condition'])[
                'er'].transform(lambda m: (100 - m) / 100).values
        model_input['var_rt'] = \
            model_input.groupby(['participant', 'condition'])[
                'rt'].transform(lambda m: (m / 1000).var()).values
        model_input['mean_rt'] = \
            model_input.groupby(['participant', 'condition'])[
                'rt'].transform(lambda m: m.mean() / 1000).values

        edge_correction_n = self.data.loc[(
                                                  self.data[
                                                      'participant'] == 0) & (
                                                  self.data[
                                                      'condition'] == 'Baseline')].shape[
            0]

        model_input = model_input.groupby(['participant', 'condition'])[
            ['proportion_correct', 'var_rt', 'mean_rt']].mean()
        self.model_output = model_input.join(pd.DataFrame(
            data=pd.DataFrame(data=np.array(ezdm.ez_diffusion(
                *model_input[
                    ['proportion_correct', 'var_rt', 'mean_rt']].values.T,
                edge_correction_n=edge_correction_n)).T,
                              columns=['drift_rate', 'boundary_separation',
                                       'non_decision_time',
                                       'mean_decision_time'],
                              index=model_input.index).round(3)))

    def _format_data_for_air(self):
        # Aggregate by participant and condition
        self.agg_data = self.clean_data.groupby(
            ['participant', 'condition']
        ).mean()[['rt', 'er']].round(2)
        self.agg_data = self.agg_data.join(self.model_output).copy()

        # Subtract the baseline from all other conditions.
        self.contrasts_data = self.agg_data.groupby(level=[0]).apply(
            lambda s: s - s.values[0]).reset_index(level=1, drop=False)

        # Now drop the 'Baseline' rows
        self.contrasts_data = (
            self.contrasts_data.loc[
                self.contrasts_data['condition'] != 'Baseline']
        ).reset_index(drop=False)

        self.agg_data2 = self.clean_data.groupby(
            ['participant', 'condition', 'cuenumber']
        ).mean()[['rt', 'er']].round(2)

    def _gen_summary_table(self):
        # aggregate the data
        summary = self.agg_data.groupby('condition')[DEP_VARS].agg(
            ['mean', 'std'])

        summary['non_decision_time'] *= 1000
        summary[['rt', 'er', 'non_decision_time']] = np.round(
            summary[['rt', 'er', 'non_decision_time']].values, 2)
        summary[['drift_rate', 'boundary_separation']] = np.round(
            summary[['drift_rate', 'boundary_separation']].values, 3)

        # Change column names on both levels of columns' multiindex.
        summary.columns.set_levels([
            DEP_VARS_LABELS, ['M', 'SD']],
            level=[0, 1], inplace=True)

        summary = pd.DataFrame(
            np.array(["{} ({})".format(m, sd) for m, sd in
                      zip(
                          summary.loc[:, (slice(None), 'M')].values.flatten(),
                          summary.loc[:, (slice(None), 'SD')].values.flatten()
                      )
                      ]).reshape((4, 5)),
            columns=summary.columns.levels[0],
            index=summary.index
        )

        summary.rename_axis(
            "Condition (n = {})".format(len(self.contrasts_data) // 3),
            axis='index', inplace=True)
        summary.rename(columns=dict(zip(summary.columns, TITALIZED_DEP_VARS)),
                       inplace=True)

        if self.write_text:
            with open(
                    "./output/texts/prepair_report_filtered_data_{}.txt".format(
                        self.filters_applied),
                    "a") as f:
                f.write(summary.to_string() + "\n***\n")

        return summary

    def get_finalized_data(self):
        """

        Returns
        -------
        pd.DataFrame
            Processed (long) data, with labeling of valid and invalid trials.
        pd.DataFrame
            Aggregated data by participant and experimental condition.
        pd.DataFrame
            Basline condition results subtracted from the results on other
            conditions (all performed on aggregated data on participant and
            condition).
        pd.DataFrame
            Marginal means and standard deviations on each experimental
            condition.
        """
        return (self.agg_data.copy().reset_index(),
                self.contrasts_data.copy().reset_index(),
                self.agg_data2.copy().reset_index(),
                self._gen_summary_table())

    def plot_diffusion(self):
        ezdm.plot_diffusion(
            self.agg_data.copy().reset_index(), self.contrasts_data.copy().reset_index(),
            self.data.copy().reset_index(),
            self.filters_applied, save_figs=self.save_figs)
