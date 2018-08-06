import pandas as pd
import numpy as np
import json
from scipy.interpolate import interp1d
from decim import glaze_control as glc
from os.path import join
from glob import glob
import decim.slurm_submit as slu
import sys
from multiprocessing import Pool


runs = ['inference_run-4', 'inference_run-5', 'inference_run-6', 'instructed_run-7', 'instructed_run-8']
data_dir = '/Volumes/flxrl/fmri/bids_mr'
out_dir = '/home/khagena/FLEXRULE/fmri/behav_fmri_aligned'
hummel_out = '/work/faty014/FLEXRULE/behavior/behav_fmri_aligned'


slu.mkdir_p(hummel_out)


def hrf(t):
    '''
    A hemodynamic response function
    '''
    h = t ** 8.6 * np.exp(-t / 0.547)
    h = np.concatenate((h * 0, h))
    return h / h.sum()


def make_bold(evidence, dt=0.25):
    '''
    Convolve with haemodynamic response function.
    '''
    t = np.arange(0, 20, dt)
    return np.convolve(evidence, hrf(t), 'same')


def interp(x, y, target):
    '''
    Interpolate
    '''
    f = interp1d(x.values.astype(int), y)
    target = target[target.values.astype(int) > min(x.values.astype(int))]
    return pd.DataFrame({y.name: f(target.values.astype(int))}, index=target)


def regular(df, target='16ms'):
    '''
    Set datetime index and resample to target frequency.
    '''
    dt = pd.to_timedelta(df.index.values, unit='ms')
    df = df.set_index(dt)
    target = df.resample(target).mean().index
    return pd.concat([interp(dt, df[c], target) for c in df.columns], axis=1)


def origin_stamp(subject, session, run_index, data_dir):
    '''
    Load starting point of first EPI.
    '''
    log = glc.load_logs_bids(subject, session, data_dir)[1][run_index]
    return log.value[0]


def rep_time(subject, session, run_index, data_dir):
    with open(glob(join(data_dir, 'sub-%s' % subject, 'ses-%s' % session, 'func', '*inference*json'))[run_index]) as json_data:
        d = json.load(json_data)
    repTime = d['RepetitionTime']
    return repTime


def execute(sub, ses, run_index):
    '''
    Output: pd.DataFrame with
                - parameters as columns
                - timedelta as index
                - convolved with hrf
                - downsampled to EPI-f
    '''

    b = pd.read_hdf('/work/faty014/FLEXRULE/behavior/behav_dataframes/behav_sub-{0}_ses-{1}.hdf'.
                    format(sub, ses), key='{}'.format([4, 5, 6, 7, 8][run_index]))
    b.onset = b.onset.astype(float)
    b = b.sort_values(by='onset')
    b = b.loc[:, ['onset', 'belief', 'murphy_surprise', 'switch', 'point', 'response', 'response_left',
                  'response_right', 'stimulus_horiz', 'stimulus_vert', 'stimulus',
                  'rresp_left', 'rresp_right', 'LLR']]
    b = b.set_index((b.onset.values * 1000).astype(int)).drop('onset', axis=1)
    b = b.reindex(pd.Index(np.arange(0, b.index[-1] + 15000, 1)))
    b.loc[0] = 0
    b.belief = b.belief.fillna(method='ffill')
    b.murphy_surprise = b.murphy_surprise.fillna(method='ffill')
    b['abs_belief'] = b.belief.abs()
    b['belief_left'] = -b.belief
    b['LLR_right'] = b.LLR
    b['LLR_left'] = -b.LLR
    b['abs_LLR'] = b.LLR.abs()
    b = b.fillna(False).astype(float)
    for column in b.columns:
        b[column] = make_bold(b[column].values, dt=.001)
    b = regular(b, target='1900ms')
    b.loc[pd.Timedelta(0)] = 0
    b = b.sort_index()
    b.to_hdf(join(hummel_out, 'beh_regressors_sub-{0}_ses-{1}.hdf'.format(sub, ses)), key=runs[run_index])
    return b


def execute_delay(sub, ses, run_index, delay, behav_dir='/work/faty014'):
    '''
    Output: pd.DataFrame with
                - delays as columns
                - timedelta as index
                - no convolution, just shifted
                - downsampled to EPI-f
    '''
    b = pd.read_csv(join(behav_dir, 'behav_dataframes', 'sub-{0}/behav_sub-{0}_ses-{1}_run-{2}.csv'.
                         format(sub, ses, [4, 5, 6][run_index])),
                    index_col=0)
    b.onset = b.onset.astype(float)
    b = b.sort_values(by='onset')
    b = b.loc[:, ['onset', 'belief', 'murphy_surprise', 'switch', 'point', 'response', 'response_left',
                  'response_right', 'stimulus_horiz', 'stimulus_vert', 'stimulus',
                  'rresp_left', 'rresp_right', 'LLR']]
    b = b.set_index((b.onset.values * 1000).astype(int)).drop('onset', axis=1)
    b = b.reindex(pd.Index(np.arange(0, b.index[-1] + 15000, 1)))
    b.loc[0] = 0
    b.belief = b.belief.fillna(method='ffill')
    b.murphy_surprise = b.murphy_surprise.fillna(method='ffill')
    b['abs_belief'] = b.belief.abs()
    b['belief_left'] = -b.belief
    b = b.fillna(False).astype(float)
    b = b.shift(delay)
    b.iloc[0:delay] = 0
    b = regular(b, target='1900ms')
    b.loc[pd.Timedelta(0)] = 0
    b = b.sort_index()
    b.to_csv(join(hummel_out, 'beh_regressors_sub-{0}_ses-{1}_{2}_delay={3}'.format(sub, ses, runs[run_index], delay)))
    return b


def keys():
    keys = []
    for sub in range(1, 23):
        for ses in [2, 3]:
            for run in [0,1,2,3,4]:
                keys.append((sub, ses, run))
    return keys


def par_execute(keys):
    with Pool(16) as p:
        p.starmap(execute, keys)


def submit():
    slu.pmap(par_execute, keys(), walltime='2:55:00',
             memory=30, nodes=1, tasks=16, name='fmri_align')
