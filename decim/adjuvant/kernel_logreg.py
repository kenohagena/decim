import numpy as np
import pandas as pd
from decim.adjuvant import slurm_submit as slu
from os.path import join
from sklearn.linear_model import LogisticRegression
from pymeg import parallel as pbs
from scipy.special import expit
import datetime


samples = {8: '2020-08-26--6.0',
           12: '2020-08-26--5.0',
           16: '2020-08-26--4.0',
           20: '2020-08-26--3.0',
           24: '2020-08-26--2.0',
           'psi': '2020-11-03'}

normative_fits = pd.read_csv('/home/khagena/FLEXRULE/behavior/summary_stan_fits.csv')
leaky_fits = pd.read_csv('/home/khagena/FLEXRULE/behavior/Stan_Fits_Leaky_2020-08-22/new/summary_stan_fits.csv')
f = {'leak': leaky_fits,
     'normative': normative_fits}
y = {'leak': 'accumulated_leaky_belief',
     'normative': 'accumulated_belief'}


def regress(n, krun, C, out_dir, mode):
    fits = f[mode]
    bel = y[mode]
    coefs = []
    for sub in range(1, 23):
        if sub == 11:
            continue

        elif 'sub-{}'.format(sub) in fits.loc[fits.vmode > 2.5].subject.unique():
            print('discard', sub)
            continue

        else:
            coef_mean = []
            for i in range(1000):
                e = []
                for ses in [2, 3]:
                    for run in [4, 5, 6]:
                        try:
                            epochs = pd.read_hdf('/home/khagena/FLEXRULE/Workflow/Sublevel_KernelEpochs_Climag_{2}/sub-{0}/KernelEpochs_sub-{0}_ses-{1}.hdf'.format(sub, ses, krun),
                                                 key='inference_run-{}'.format(run))
                            epochs['choice_probabilities'] = expit(epochs.behavior.parameters[bel].values /
                                                                   fits.loc[(fits.subject == 'sub-{}'.format(sub)) & (fits.session == 'ses-{}'.format(ses))].vmode.values)
                            e.append(epochs)
                        except FileNotFoundError:
                            print('no file', sub, ses, run)
                        except KeyError:
                            print('key', sub, ses, run)
                epochs = pd.concat(e, ignore_index=True, axis=0)
                llr_cpp = epochs.behavior.surprise.drop('trial_id', axis=1).multiply(epochs.behavior.LLR.drop('trial_id', axis=1))
                llr_cpp = llr_cpp.rename(columns={i: 'cpp{0}'.format(i) for i in llr_cpp.columns})
                llr_psi = -epochs.behavior.psi.drop('trial_id', axis=1).abs().multiply(epochs.behavior.LLR.drop('trial_id', axis=1))
                llr_psi = llr_psi.rename(columns={i: 'psi{0}'.format(i) for i in llr_psi.columns})
                data = pd.concat([epochs.behavior.prev_psi.prev_psi, epochs.behavior.LLR.drop('trial_id', axis=1), llr_cpp, llr_psi, epochs.choice_probabilities], axis=1)
                data = data.dropna(axis=0)
                x = data.drop('choice_probabilities', axis=1)
                x = (x - x.mean()) / x.std()
                l = LogisticRegression(C=C)
                l.fit(x.values, np.random.binomial(n=1, p=data.choice_probabilities))
                coef_mean.append(l.coef_[0])
            coefs.append(pd.DataFrame(coef_mean).mean())
    pd.DataFrame(coefs).to_hdf(join(out_dir, '{0}_model_kernels_C={1}.hdf'.format(mode, C)), key=str(n))


def submit():
    out_dir = join('/home/khagena/FLEXRULE/behavior/kernels_psi-{}'.format(datetime.datetime.now().strftime("%Y-%m-%d")))
    slu.mkdir_p(out_dir)
    C = 1
    n = 12
    run = samples['psi']

    for mode in ['leak', 'normative']:
        pbs.pmap(regress, [(n, run, C, out_dir, mode)],
                 walltime='1:00:00', memory=15, nodes=1, tasks=1,
                 name='kernels_{0}'.format(n))
