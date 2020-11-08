import pandas as pd
import numpy as np
from os.path import join, expanduser
from glob import glob
from sklearn.linear_model import LogisticRegression
import datetime


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

    def prev_psi(self):
        '''
        Add last n points before choice onset.
        '''
        df = self.BehavFrame
        points = df.loc[(df.message == 'GL_TRIAL_LOCATION')]
        points['psi'] = (points['psi'] - points['psi'].mean()) / points['psi'].std()
        p = []
        for i, row in self.choices.iterrows():
            trial_points = points.loc[points.index.astype('float') < row.trial_id]
            if len(trial_points) < self.n_samples:
                trial_points = np.full(self.n_samples, np.nan)
            else:
                trial_points = trial_points['psi'].values[len(trial_points) - self.n_samples]
            p.append(trial_points)
        pp = pd.DataFrame(columns=['prev_psi', 'trial_id'])
        pp['prev_psi'] = pd.Series(p)
        pp['trial_id'] = self.choices.trial_id.values
        self.kernels['prev_psi'] = pp

    def merge(self):
        '''
        Merge everything into one pd.MultiIndex pd.DataFrame.
        '''
        self.choices.columns =\
            pd.MultiIndex.from_product([['behavior'], ['parameters'],
                                        self.choices.columns],
                                       names=['source', 'type', 'name'])
        self.kernels['prev_psi'].columns =\
            pd.MultiIndex.from_product([['behavior'], ['prev_psi'],
                                        self.kernels['prev_psi'].columns],
                                       names=['source', 'type', 'name'])
        for p in self.parameters:
            self.kernels[p].columns =\
                pd.MultiIndex.from_product([['behavior'], [p],
                                            list(range(self.kernels[p].shape[1] - 1)) + ['trial_id']],
                                           names=['source', 'type', 'name'])

        master = pd.concat([self.kernels[p] for p in self.parameters] + [self.choices] + [self.kernels['prev_psi']], axis=1)
        self.master = master.set_index([master.behavior.parameters.trial_id])


def simulate_regression(trials, model_H, model_V, regression_C, n, out_dir, gen_var, psi=True, sub='optimal_observer', regression_iter=1000):

    for sub in range(22):

        df = pt.complete(pt.fast_sim(trials, gen_var=gen_var, tH=model_H), H=model_H, V=model_V, method='inverse', gen_var=gen_var)
        c = Choiceframe(df, n=n)
        c.choice_behavior()
        c.kernel_samples('LLR')
        c.kernel_samples('PCP', zs=True)
        c.kernel_samples('psi', zs=True)
        c.prev_psi()
        c.merge()
        coefs = []
        for i in range(regression_iter):
            epochs = c.master
            llr_cpp = epochs.behavior.PCP.drop('trial_id', axis=1).multiply(epochs.behavior.LLR.drop('trial_id', axis=1))
            llr_cpp = llr_cpp.rename(columns={i: 'cpp{0}'.format(i) for i in llr_cpp.columns})
            llr_psi = -epochs.behavior.psi.drop('trial_id', axis=1).abs().multiply(epochs.behavior.LLR.drop('trial_id', axis=1))
            llr_psi = llr_psi.rename(columns={i: 'psi{0}'.format(i) for i in llr_psi.columns})
            if psi is True:
                data = pd.concat([epochs.behavior.LLR.drop('trial_id', axis=1), epochs.behavior.prev_psi.prev_psi, llr_cpp, llr_psi, epochs.behavior.parameters.choice_probability], axis=1)
            elif psi is False:
                data = pd.concat([epochs.behavior.LLR.drop('trial_id', axis=1), llr_cpp, llr_psi, epochs.behavior.parameters.choice_probability], axis=1)
            data = data.dropna(axis=0)
            x = data.drop('choice_probability', axis=1)
            x = (x - x.mean()) / x.std()
            logreg = LogisticRegression(C=regression_C)
            logreg.fit(x.values, np.random.binomial(n=1, p=data.choice_probability))
            coefs.append(logreg.coef_[0])
        coefs = pd.DataFrame(coefs, columns=x.columns).mean()
        coefs['n'] = n
        coefs['trials'] = len(df.loc[df.message == 'decision'].belief.values)
        coefs['H'] = model_H
        coefs['V'] = model_V
        coefs['C'] = regression_C
        coefs['subject'] = sub
        coefs['gen_var'] = gen_var
        coefs['previous_psi'] = psi
        coefs.to_hdf(join(out_dir, 'simulated_regression_{0}_{1}_{2}_{3}_psi-{4}.hdf'.format(gen_var, model_H, sub, trials, psi)), key=str(regression_C))


def submit():
    fits = pd.read_csv('/home/khagena/FLEXRULE/behavior/summary_stan_fits.csv')
    subjects = fits.loc[fits.vmode < 2.5].subject.unique()
    out_dir = join('/home/khagena/FLEXRULE/behavior/kernel_simulation/KernelSimulation_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d")))
    slu.mkdir_p(out_dir)
    '''
    for subject in subjects:
        V = fits.loc[fits.subject == subject].vmode.mean()
        H = fits.loc[fits.subject == subject].hmode.mean()
        for C in [1, 1e8]:
            for n in [8, 12]:
                pbs.pmap(simulate_regression, [(100000, H, V, C, n, out_dir, subject)],
                         walltime='1:00:00', memory=15, nodes=1, tasks=1,
                         name='kernels')\
    '''
    for H in [1 / 70, 0.08]:
        for gen_sigma in [0.75, 1, 1.25]:
            for n in [12]:
                for psi in [False, True]:
                    pbs.pmap(simulate_regression, [(6000, H, 1, 1, n, out_dir, gen_sigma, psi)],
                             walltime='4:00:00', memory=15, nodes=1, tasks=1,
                             name='kernels')


def single():
    fits = pd.read_csv('/home/khagena/FLEXRULE/behavior/summary_stan_fits.csv')
    subjects = fits.loc[fits.vmode < 2.5].subject.unique()
    out_dir = join('/home/khagena/FLEXRULE/behavior/kernel_simulation/KernelSimulation_{}-2'.format(datetime.datetime.now().strftime("%Y-%m-%d")))
    slu.mkdir_p(out_dir)
    for subject in subjects[0:2]:
        V = fits.loc[fits.subject == subject].vmode.mean()
        H = fits.loc[fits.subject == subject].hmode.mean()
        gen_sigma = 0.75
        for C in [1]:
            for n in [8]:
                print(subject, n, C, V, H, out_dir)
                pbs.pmap(simulate_regression, [(100000, H, V, C, n, out_dir, gen_sigma)],
                         walltime='1:00:00', memory=15, nodes=1, tasks=1,
                         name='kernels')
