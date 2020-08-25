import pandas as pd
import numpy as np
from os.path import join, expanduser
from glob import glob
from sklearn.linear_model import LogisticRegression

from decim.adjuvant import slurm_submit as slu
from collections import defaultdict
from decim.adjuvant import pointsimulation as pt
from pymeg import parallel as pbs


class Choiceframe(object):

    def __init__(self, df, n):

        self.BehavFrame = df
        self.n_samples = n
        self.parameters = ['LLR', 'PCP', 'psi']
        self.kernels = defaultdict()

    def choice_behavior(self):
        df = self.BehavFrame
        choices = pd.DataFrame({'trial_id': df.loc[df.message == 'decision'].index.values.astype(int),
                                'response': df.loc[df.message == 'decision', 'choice'].values.astype(float)})
        df.belief = df.belief.fillna(method='ffill')
        choices['choice_probability'] =\
            df.loc[df.message == 'decision'].noisy_belief.values.astype(float)
        self.choices = choices

    def kernel_samples(self, parameter, log=False, zs=False):
        '''
        Add last n points before choice onset.
        '''
        df = self.BehavFrame
        points = df.loc[(df.message == 'GL_TRIAL_LOCATION')]
        if log is True:
            points.loc[1:, parameter] = np.log(points.loc[1:, parameter])                   # first surprise is 0 --> inf introduced
        if zs is True:
            points[parameter] = (points[parameter] - points[parameter].mean()) / points[parameter].std()
        p = []
        for i, row in self.choices.iterrows():
            trial_points = points.loc[points.index.astype('float') < row.trial_id]
            if len(trial_points) < self.n_samples:
                trial_points = np.full(self.n_samples, np.nan)
            else:
                trial_points = trial_points[parameter].values[len(trial_points) - self.n_samples:len(trial_points)]
            p.append(trial_points)
        points = pd.DataFrame(p)
        points['trial_id'] = self.choices.trial_id.values
        self.kernels[parameter] = points

    def merge(self):
        '''
        Merge everything into one pd.MultiIndex pd.DataFrame.
        '''
        self.choices.columns =\
            pd.MultiIndex.from_product([['behavior'], ['parameters'],
                                        self.choices.columns],
                                       names=['source', 'type', 'name'])
        for p in self.parameters:
            self.kernels[p].columns =\
                pd.MultiIndex.from_product([['behavior'], [p],
                                            list(range(self.kernels[p].shape[1] - 1)) + ['trial_id']],
                                           names=['source', 'type', 'name'])

        master = pd.concat([self.kernels[p] for p in self.parameters] + [self.choices], axis=1)
        self.master = master.set_index([master.behavior.parameters.trial_id])


def simulate_regression(trials, model_H, model_V, regression_C, n, out_dir, sub='optimal_observer', regression_iter=1000):

    df = pt.complete(pt.fast_sim(trials), H=model_H, V=model_V, method='inverse')
    c = Choiceframe(df, n=n)
    c.choice_behavior()
    c.kernel_samples('LLR')
    c.kernel_samples('PCP', zs=True, log=True)
    c.kernel_samples('psi', zs=True)
    c.merge()
    coefs = []
    for i in range(regression_iter):
        epochs = c.master
        llr_cpp = epochs.behavior.PCP.drop('trial_id', axis=1).multiply(epochs.behavior.LLR.drop('trial_id', axis=1))
        llr_cpp = llr_cpp.rename(columns={i: 'cpp{0}'.format(i) for i in llr_cpp.columns})
        llr_psi = -epochs.behavior.psi.drop('trial_id', axis=1).abs().multiply(epochs.behavior.LLR.drop('trial_id', axis=1))
        llr_psi = llr_psi.rename(columns={i: 'psi{0}'.format(i) for i in llr_psi.columns})
        data = pd.concat([epochs.behavior.LLR.drop('trial_id', axis=1), llr_cpp, llr_psi, epochs.behavior.parameters.choice_probability], axis=1)
        data = data.dropna(axis=0)
        x = data.drop('choice_probability', axis=1)
        x = (x - x.mean()) / x.std()
        logreg = LogisticRegression(C=regression_C)
        logreg.fit(x.values, np.random.binomial(n=1, p=data.choice_probability))
        coefs.append(logreg.coef_[0])
    coefs = pd.DataFrame(coefs, columns=x.columns)
    coefs['n'] = n
    coefs['trials'] = len(df.loc[df.message == 'decision'].belief.values)
    coefs['H'] = model_H
    coefs['V'] = model_V
    coefs['C'] = regression_C
    coefs['subject'] = sub
    coefs.mean().to_hdf(join(out_dir, 'simulated_regression_{0}_{1}_{2}'.format(n, model_V, sub)), key=regression_C)


def submit():
    fits = pd.read_csv('/home/khagena/FLEXRULE/behavior/summary_stan_fits.csv')
    subjects = fits.loc[fits.vmode < 2.5].subject.unique()
    out_dir = join('/home/khagena/FLEXRULE/behavior/kernel_simulation')
    slu.mkdir_p(out_dir)
    for subject in subjects:
        V = fits.loc[fits.subject == subject].vmode.values
        H = fits.loc[fits.subject == subject].hmode.values
        for C in [1, 1e8]:
            for n in [8, 12]:
                pbs.pmap(simulate_regression, [(100000, H, V, C, n, out_dir, subject)],
                         walltime='1:00:00', memory=15, nodes=1, tasks=1,
                         name='kernels')
    for V in [1, 1.5, 2., 2.5]:
        for C in [1, 1e8]:
            for n in [8, 12]:
                pbs.pmap(simulate_regression, [(100000, 1 / 70, V, C, n, out_dir)],
                         walltime='1:00:00', memory=15, nodes=1, tasks=1,
                         name='kernels')


def single():
    fits = pd.read_csv('/home/khagena/FLEXRULE/behavior/summary_stan_fits.csv')
    subjects = fits.loc[fits.vmode < 2.5].subject.unique()
    out_dir = join('/home/khagena/FLEXRULE/behavior/kernel_simulation')
    slu.mkdir_p(out_dir)
    for subject in subjects[0]:
        print(subject)
        V = fits.loc[fits.subject == subject].vmode.mean()
        H = fits.loc[fits.subject == subject].hmode.mean()
        for C in [1]:
            for n in [8]:
                pbs.pmap(simulate_regression, [(100000, H, V, C, n, out_dir, subject)],
                         walltime='1:00:00', memory=15, nodes=1, tasks=1,
                         name='kernels')
