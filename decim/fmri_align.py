import pandas as pd
import numpy as np
import json
from scipy.interpolate import interp1d
from decim import glaze_control as glc
from os.path import join
from glob import glob


data_dir = '/Volumes/flxrl/fmri/bids_mr'


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


def execute(subject, session, run_index):
    '''
    Output: pd.DataFrame with
                - parameters as columns
                - timedelta as index
                - convolved with hrf
                - downsampled to EPI-f
    '''
    b = pd.read_csv('/Users/kenohagena/Flexrule/fmri/analyses/behav_dataframes_310518/sub-{0}/behav_sub-{0}_ses-{1}_run-{2}.csv'.
                    format(subject, session, [4, 5, 6][run_index]),
                    index_col=0).sort_values(by='onset')
    b = b.loc[:, ['onset', 'belief', 'murphy_surprise', 'switch']].dropna(how='any')
    b['abs_belief'] = b.belief.abs()
    b = b.set_index((b.onset.values * 1000).astype(int)).drop('onset', axis=1)
    b = b.reindex(pd.Index(np.arange(0, b.index[-1] + 15000, 1)))  # +10.000 because of 10s EPIs after task is finished
    b.loc[0] = 0
    b = b.fillna(method='ffill').astype(float)
    for column in b.columns:
        b[column] = make_bold(b[column].values, dt=.001)
    b = regular(b, target='1900ms')
    b.loc[pd.Timedelta(0)] = 0
    b = b.sort_index()
    return b


### TO EXECUTE UNCOMMENT AND INSERT SUBJECT ARRAY ###
'''
subjects = [1, 2, 3, 4, 5, 6]
runs = ['inference_run-4', 'inference_run-5', 'inference_run-6']

for sub in range(1, 23):
    for ses in [2, 3]:
        for run in [0, 1, 2]:
            try:
                regressor = execute(sub, ses, run)
                regressor.to_csv('/Users/kenohagena/Flexrule/fmri/analyses/behav_fmri_aligned_060618/beh_regressors_sub-{0}_ses-{1}_{2}'.format(sub, ses, runs[run]))
            except FileNotFoundError:
                print('file not found for {0} {1} {2}'.format(sub, ses, run))

'''
