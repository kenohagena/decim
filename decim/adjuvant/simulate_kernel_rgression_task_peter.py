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

    change_points = np.append([0], [np.random.binomial(n=1, p=H) for i in range(n - 1)])
    l = []
    for i, cp in enumerate(change_points):
        if i == 0:
            distribution = 2 * (np.random.binomial(p=.5, n=1) - 0.5) * dist_mean       # random first generative distribution
            psi = 0
            pcp = 0
            loc = norm.rvs(distribution, var, 1)
            LLR = gm.LLR(loc, dist_mean, -dist_mean, var)
        elif i > 0:
            if cp == 1:
                distribution = distribution * -1
            if cp == 0:
                distribution = distribution
            psi = gm.prior(Ln, H)[0]
            loc = norm.rvs(distribution, var, 1)
            LLR = gm.LLR(loc, dist_mean, -dist_mean, var)
            pcp = gm.pcp(loc, Ln, H, e_right=dist_mean,
                         e_left=-dist_mean, sigma=var)[0]

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
                        out_dir, var, dist_mean=.5,
                        sub='optimal_observer', regression_iter=1000, subs=20):
    su = []
    for s in range(subs):
        simulated_trials = multi_trials(trials, H, V, var, dist_mean)
        coefs = []
        for i in range(regression_iter):
            llr_cpp = simulated_trials.xs('pcp', axis=1, level=1, drop_level=True).drop(0, axis=1)
            llr_cpp = ((llr_cpp.T - llr_cpp.mean(axis=1)) / llr_cpp.std(axis=1)).T
            llr_cpp = llr_cpp.multiply(simulated_trials.xs('LLR', axis=1, level=1, drop_level=True).drop(0, axis=1))

            llr_psi = -simulated_trials.xs('psi', axis=1, level=1, drop_level=True).abs().drop(0, axis=1)
            llr_psi = ((llr_psi.T - llr_psi.mean(axis=1)) / llr_psi.std(axis=1)).T
            llr_psi = llr_psi.multiply(simulated_trials.xs('LLR', axis=1, level=1, drop_level=True).drop(0, axis=1))
            data = pd.concat([simulated_trials.xs('LLR', axis=1, level=1, drop_level=True), llr_cpp, llr_psi, simulated_trials.loc[:, 11].choice_prob], axis=1)

            x = data.drop('choice_prob', axis=1)
            x = (x - x.mean()) / x.std()

            logreg = LogisticRegression(C=regression_C)
            logreg.fit(x.values, np.random.binomial(n=1, p=data.choice_prob))
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
        su.append(coefs)
    su = pd.concat(su, axis=1)
    su.to_hdf(join(out_dir, 'sim_reg_{0}_{1}_{2}_{3}.hdf'.format(var, H, sub, trials)), key=str(regression_C))


def submit():
    out_dir = join('/home/khagena/FLEXRULE/behavior/kernel_simulation/simulate_murphy_task')
    slu.mkdir_p(out_dir)

    for H in [0.001, 0.01, 1 / 70, 0.08, 0.2, 0.3]:
        for gen_sigma in [1, 0.5, 0.75, 1.25, 1.5]:
            for n in [12]:
                for C in [1, 1e8]:
                    pbs.pmap(simulate_regression, [(5000, H, 1, C, n, out_dir, gen_sigma)],
                             walltime='4:00:00', memory=15, nodes=1, tasks=1,
                             name='kernels')


def single():
    out_dir = join('/home/khagena/FLEXRULE/behavior/kernel_simulation/simulate_murphy_task')
    slu.mkdir_p(out_dir)

    for H in [1 / 70]:
        for gen_sigma in [0.75]:
            for n in [12]:
                for C in [1]:
                    pbs.pmap(simulate_regression, [(50, H, 1, C, n, out_dir, gen_sigma)],
                             walltime='1:00:00', memory=15, nodes=1, tasks=1,
                             name='kernels')
