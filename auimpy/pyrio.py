"""
pyrio is a module written to set the interface with R (e.g., imports or installs
missing packages). The initialized R interface is further used through the
PyRio object.
"""

import numpy as np
import pandas as pd
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri, numpy2ri

numpy2ri.activate()

# Packages we need to import to the R environment to analyze and tidy the data.
PKGS = ['broom', 'afex', 'base', 'BayesFactor', 'stats', 'psych', 'emmeans',
        'stats', 'broom', 'tibble']


def _convert_df(df):
    # this conversion works on pandas 0.23.4 - but might not work on 0.25
    if isinstance(df, pd.DataFrame):
        return pandas2ri.py2ri(df)
    else:
        return pd.DataFrame(pandas2ri.ri2py(df))


def _install_if_required(pack):
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    utils.install_packages(pack)


class PyRIO:
    """Python-R-Interface-Object"""

    def __init__(self, pkgs=PKGS):

        self.pkgs = pkgs
        self.cur_r_objects = self._get_r_objects()

    def _get_r_objects(self):
        """Imports/installs the required packages.

        Parameters
        ----------
        None.

        Returns
        -------
        dict: Keys are packages names (see PKGS above), values are the R
        packages.
        """
        self._verify_if_installed()
        return dict(zip(self.pkgs, map(
            rpackages.importr, self.pkgs)))

    def _verify_if_installed(self):
        [_install_if_required(pack) for pack in self.pkgs if
         not rpackages.isinstalled(pack)]

    def t_test(self, x, y, paired, null_interval, tail, iters_num):
        """A function to run several frequentist and Bayesian tests or calculate
        measures.
        Parameters
        ----------
        x: np.array
            A 1-d np.array of x (group-1) values on the dependent variable
        y: np.array
            A 1-d np.array of y (group-2) values on the dependent variable
        paired: bool
            Whether to run dependent or independent samples t-tests.
        null_interval: np.array
            A np.array of the values describing the location of H1 on a
            directional test (see https://richarddmorey.github.io/BayesFactor/).
        tail: str
            A string describing the directionality of the frequentist t-test.
            Optional values are 'less' (H1: x < y) 'greater' (H1: x > y) or
            two.sided (H1: x != y)/
        iters_num: int
            Number of iterations to perform for the Bayesian t-test and
            posterior distribution recovery.

        Returns
        -------
        Dict with the following keys:
            'BF': float
                Bayes Factor for H1.
            'posterior': pd.DataFrame
                Posterior distribution of paramter values.
            'freq': pd.DataFrame
                A frequentist t-test. (T-value, df, p-value, etc.).
            'cohen': pd.DataFrame
                The standardized effect size with a 95%-CI.
            'sequential_BF': pd.DataFrame
                A DataFrame with the matching BF (see above) values calculated
                with a sequential addition of a participant to the sample,
                beginning with the 2nd participant.
        """
        return {
            'BF': _convert_df(self.cur_r_objects['base'].data_frame(
                self.cur_r_objects['BayesFactor'].ttestBF(
                    x=x, y=y, paired=paired,
                    nullInterval=null_interval,
                    iterations=iters_num)))[['bf', 'error']].round(
                2).iloc[0].to_dict()['bf'],
            'posterior': _convert_df(
                self.cur_r_objects['BayesFactor'].ttestBF(
                    x=x, y=y, paired=paired, nullInterval=null_interval,
                    posterior=True, iterations=iters_num)).rename(
                # The returned DataFrame has no column names
                columns=dict(zip(range(0, 4),
                                 ["mu", "sig2", "delta", "g"]))),
            'freq': _convert_df(self.cur_r_objects['broom'].tidy_htest(
                (self.cur_r_objects['stats'].t_test(x=x, y=y, paired=paired,
                                                    alternative=tail)))).rename(
                columns={'p.value': 'pvalue'}).replace(
                {'less': 'Lower-tail',
                 'greater': 'Upper-tail',
                 'two.sided': 'Two-tail'}),
            'cohen': dict(zip(['lower', 'cohen', 'upper'],
                              _convert_df(self.cur_r_objects['psych'].d_ci(
                                  (x - y).mean() / (x - y).std(), n1=len(x))).round(3).values.flatten())),
            'sequential_BF':
                pd.Series(dict([
                    (n,
                     self._sequential_bf(x[:n], y[:n], paired, null_interval,
                                         iters_num))
                    for n in range(1, len(x))
                ]), name='BF').reset_index(drop=False).rename(
                    columns={'index': 'N'})
        }

    def anova_within(self, df_py, dep_var, ind_var, subject_id):
        """Performs one-way repeated measures frequentist anova and 95%-CI of
        marginal means based on the model.

        Parameters
        ---------
        df_py: pd.DataFrame
            A pandas dataframe conatining the data to be analyzed - a dependent
            variable, independent variable(s) and a subject identifier.
        dep_var: str
            The dependent variable in the anova. Should be the column name of
            the dv in df_py.
        ind_var: str
            The independent variable in the anova. Should be the column name of
            the iv in df_py.
        subject_id: str
            The subject identifier in the anova. Should be the column name of
            the subject id in df_py.

        Returns
        -------
        anova_frequentist: pd.DataFrame
            The anova results (F-value, Degrees of freedom, effect size,
            p-value, etc).
        margins: pd.DataFrame
            Marginal means and 95%-CI based on the model (see
                https://cran.r-project.org/web/packages/margins)
        """
        # Convert the anova to an R dataframe
        anv_r_df = _convert_df(df_py)

        anv_r_df[anv_r_df.names.index(dep_var)] = self.cur_r_objects[
            'base'].as_numeric(anv_r_df[anv_r_df.names.index(dep_var)])

        # Run the Anova in R
        a1 = self.cur_r_objects['afex'].aov_ez(
            dv=dep_var, id=subject_id, within=ind_var,
            data=anv_r_df)

        m1 = _convert_df(
            self.cur_r_objects['emmeans'].as_data_frame_emmGrid(
                self.cur_r_objects['emmeans'].emmeans(
                    a1,
                    specs=ind_var,
                    type='response')))

        a1 = _convert_df(
            self.cur_r_objects['tibble'].rownames_to_column(
                self.cur_r_objects['base'].data_frame(
                self.cur_r_objects['stats'].anova(a1, es='pes')), 'effect'))

        a1.rename(columns=dict(zip(['Pr..F.', 'num.Df', 'den.Df'],
                     ['pvalue', 'DF1', 'DF2'])), inplace=True)
        return a1, m1

    def _sequential_bf(self, x, y, paired, nullInterval, iterations):
        """Runs a regular Bayesian t-test but handles errors. On some cases
        there are not enough participants (i.e., there is no variance) and the
        t-test function raises an error. Required only for a sequential process
        """
        try:
            return _convert_df(self.cur_r_objects['base'].data_frame(
                self.cur_r_objects['BayesFactor'].ttestBF(
                    x=x, y=y, paired=paired, nullInterval=nullInterval,
                    iterations=iterations)))[['bf', 'error']].round(
                2).iloc[0].to_dict()['bf']
        except:
            return np.nan
