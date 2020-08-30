import pandas as pd
import numpy as np
from os.path import join
from sklearn.linear_model import LogisticRegression
from decim.adjuvant import glaze_model as gm
from decim.adjuvant import slurm_submit as slu
from pymeg import parallel as pbs
from scipy.special import expit
from scipy.stats import norm


def make_trial(H, V, var, dist_mean, samples=12):

    change_points = np.append([0], [np.random.binomial(n=1, p=H) for i in range(11)])
    l = []
    for i, cp in enumerate(change_points):
        if i == 0:
            distribution = 2 * (np.random.binomial(p=.5, n=1) - 0.5) * dist_mean       # random first generative distribution
            psi = 0
            pcp = 0
        elif i > 0:
            if cp == 1:
                distribution = distribution * -1
            if cp == 0:
                distribution = distribution
            psi = gm.prior(Ln, H)[0]
            pcp = gm.pcp(loc, Ln, H, e_right=dist_mean,
                         e_left=-dist_mean, sigma=var)[0]
        loc = norm.rvs(distribution, var, 1)
        LLR = gm.LLR(loc, dist_mean, -dist_mean, var)

        Ln = psi + LLR

        sample = {'cp': cp,
                  'distribution': distribution,
                  'loc': loc[0],
                  'LLR': LLR[0],
                  'psi': psi,
                  'Ln': Ln[0],
                  'pcp': np.log(pcp),
                  'choice_prob': expit(Ln / V)[0]
                  }
        l.append(sample)
    trial = pd.DataFrame(l)
    return trial


def multi_trials(trials, H, V, var, dist_mean):
    trial_array = []
    for i in range(trials):
        t = make_trial(H, V, var, dist_mean)
        trial_array.append(t.stack())
    simulated_trials = pd.concat(trial_array, axis=1).T
    return simulated_trials


def simulate_regression(trials, H, V, regression_C, n,
                        out_dir, var, dist_mean,
                        sub='optimal_observer', regression_iter=1000):
    simulated_trials = multi_trials(trials, H, V, var, dist_mean)
    coefs = []
    for i in range(regression_iter):
        llr_cpp = simulated_trials.xs('pcp', axis=1, level=1, drop_level=True)
        llr_cpp = (llr_cpp - llr_cpp.values.mean()) / llr_cpp.values.std()
        llr_cpp = llr_cpp.multiply(simulated_trials.xs('LLR', axis=1, level=1, drop_level=True))

        llr_psi = -simulated_trials.xs('psi', axis=1, level=1, drop_level=True).abs()
        llr_psi = (llr_psi - llr_psi.values.mean()) / llr_psi.values.std()
        llr_psi = llr_psi.multiply(simulated_trials.xs('LLR', axis=1, level=1, drop_level=True))

        data = pd.concat([simulated_trials.xs('LLR', axis=1, level=1, drop_level=True), llr_cpp, llr_psi, simulated_trials.loc[:, 11].behavior.parameters.choice_probability], axis=1)

        x = data.drop('choice_prob', axis=1)
        x = (x - x.mean()) / x.std()
        logreg = LogisticRegression(C=1)
        logreg.fit(x.drop(0, axis=1).values, np.random.binomial(n=1, p=data.choice_prob))
        coefs.append(logreg.coef_[0])
    coefs = pd.DataFrame(coefs, columns=x.columns).mean()
    coefs['n'] = n
    coefs['trials'] = trials
    coefs['H'] = H
    coefs['V'] = V
    coefs['C'] = regression_C
    coefs['subject'] = sub
    coefs['gen_var'] = var
    coefs['dist_mean'] = dist_mean
    coefs.to_hdf(join(out_dir, 'sim_reg_{0}_{1}_{2}_{3}.hdf'.format(var, H, sub, trials)), key=str(regression_C))


def submit():
    out_dir = join('/home/khagena/FLEXRULE/behavior/kernel_simulation/simulate_murphy_task')
    slu.mkdir_p(out_dir)

    for H in [0.001, 0.01, 1 / 70, 0.08, 0.2, 0.3]:
        for gen_sigma in [1, 0.5, 0.75, 1.25]:
            for n in [12]:
                for C in [1, 1e8]:
                    pbs.pmap(simulate_regression, [(2500, H, 1, C, n, out_dir, gen_sigma)],
                             walltime='1:00:00', memory=15, nodes=1, tasks=1,
                             name='kernels')


def single():
    fits = pd.read_csv('/home/khagena/FLEXRULE/behavior/summary_stan_fits.csv')
    subjects = fits.loc[fits.vmode < 2.5].subject.unique()
    out_dir = join('/home/khagena/FLEXRULE/behavior/kernel_simulation')
    slu.mkdir_p(out_dir)
    for subject in subjects[0:2]:
        print(subject)
        V = fits.loc[fits.subject == subject].vmode.mean()
        H = fits.loc[fits.subject == subject].hmode.mean()
        for C in [1]:
            for n in [8]:
                pbs.pmap(simulate_regression, [(100000, H, V, C, n, out_dir, subject)],
                         walltime='1:00:00', memory=15, nodes=1, tasks=1,
                         name='kernels')
