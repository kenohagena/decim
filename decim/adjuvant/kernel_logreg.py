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
           'psi': '2020-11-12_genH',
           'psi2': '2020-11-12-3',
           'psi4': '2020-11-12-4'}

normative_fits = pd.read_csv('/home/khagena/FLEXRULE/behavior/summary_stan_fits.csv')
leaky_fits = pd.read_csv('/home/khagena/FLEXRULE/behavior/Stan_Fits_Leaky_2020-08-22/new/summary_stan_fits.csv')
f = {'leak': leaky_fits,
     'normative': normative_fits}
y = {'leak': 'accumulated_leaky_belief',
     'normative': 'accumulated_belief'}


def regress(n, krun, C, out_dir, mode, sub, psi):
    fits = f[mode]
    bel = y[mode]
    coef_mean = []
    for i in range(1000):
        e = []
        for ses in [2, 3]:
            V = fits.loc[(fits.subject == 'sub-{}'.format(sub)) & (fits.session == 'ses-{}'.format(ses))].vmode.values
            #V = 1
            for run in [4, 5, 6]:
                try:
                    epochs = pd.read_hdf('/home/khagena/FLEXRULE/Workflow/Sublevel_KernelEpochs_Climag_{2}/sub-{0}/KernelEpochs_sub-{0}_ses-{1}.hdf'.format(sub, ses, krun),
                                         key='inference_run-{}'.format(run))
                    epochs['choice_probabilities'] = expit(epochs.behavior.parameters[bel].values / V)
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
        if psi is True:
            data = pd.concat([epochs.behavior.prev_psi.prev_psi, epochs.behavior.LLR.drop('trial_id', axis=1), llr_cpp, llr_psi, epochs.choice_probabilities], axis=1)
        elif psi is False:
            data = pd.concat([epochs.behavior.LLR.drop('trial_id', axis=1), llr_cpp, llr_psi, epochs.choice_probabilities], axis=1)
        #data.to_hdf('/home/khagena/FLEXRULE/behavior/delete-11-12-2.hdf', key=str(sub))
        data = data.dropna(axis=0)
        x = data.drop('choice_probabilities', axis=1)
        x = (x - x.mean()) / x.std()
        logreg = LogisticRegression(C=C)
        logreg.fit(x.values, np.random.binomial(n=1, p=data.choice_probabilities))
        coef_mean.append(logreg.coef_[0])
    pd.DataFrame(coef_mean).mean().to_hdf(join(out_dir, '{0}_model_kernels_psi={1}.hdf'.format(mode, psi)), key=str(sub))


def submit():
    out_dir = join('/home/khagena/FLEXRULE/behavior/kernels_psi-{}-3'.format(datetime.datetime.now().strftime("%Y-%m-%d")))
    slu.mkdir_p(out_dir)
    C = 1
    n = 12
    run = samples['psi4']
    for sub in range(1, 23):
        for psi in [False, True]:
            for mode in ['normative']:
                pbs.pmap(regress, [(n, run, C, out_dir, mode, sub, psi)],
                         walltime='4:00:00', memory=15, nodes=1, tasks=1,
                         name='kernels_{0}'.format(n))
