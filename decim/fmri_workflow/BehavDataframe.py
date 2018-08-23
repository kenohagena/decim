import pandas as pd
import numpy as np
from decim import glaze_model as gm
from os.path import join, expanduser
from scipy.interpolate import interp1d
from decim import slurm_submit as slu
from joblib import Memory
cachedir = expanduser('~/joblib_cache')
slu.mkdir_p(cachedir)
memory = Memory(cachedir=cachedir, verbose=0)
'''
INPUT: Behavioral data from .tsv files in BIDS-Format
OUTPUT: Pandas data frame with the following columns
    - event: point / choice onset / response / stimulus onset
    - onset: time of event
    - value: either point location or choice
    - belief: Glaze Belief
    - gen_side: active distribution
    - obj belief
    - stim_id
    - rule response

To execute, make sure to set:
    - bids_mr: where is the raw data? up-to-date version?
    - outpath: where to store the output DFs?
    - summary: summary-file of stan-fits
    - subject-loop, session-loop
'''


class BehavDataframe(object):

    def __init__(self, subject, session, run, flex_dir):
        self.subject = subject
        self.session = session
        self.run = run
        self.bids_path = join(flex_dir, 'raw', 'bids_mr_v1.2')

    def inference(self, summary):
        logs = gm.load_logs_bids(self.subject, self.session, self.bids_path)
        df = logs[self.run]
        H = summary.loc[(summary.subject == self.subject) & (summary.session == self.session)].hmode.values[0]
        df['belief'], df['psi'], df['LLR'], df['surprise'] = gm.belief(df, H=H, ident='event')
        df.belief = df.belief.fillna(method='ffill')
        df = df.loc[df.event.isin(['GL_TRIAL_LOCATION', 'GL_TRIAL_GENSIDE',
                                   'GL_TRIAL_STIM_ID', 'CHOICE_TRIAL_ONSET',
                                   'CHOICE_TRIAL_STIMOFF', 'CHOICE_TRIAL_RESP',
                                   'CHOICE_TRIAL_RULE_RESP', 'GL_TRIAL_START',
                                   'GL_TRIAL_REWARD', 'CHOICE_TRIAL_RT'])]
        df = df.reset_index()
        df = df.replace('n/a', np.nan)
        df['trial_id'] = df.loc[df.event == 'GL_TRIAL_START'].value.astype(int)
        df.trial_id = df.trial_id.fillna(method='ffill')
        df['gen_side'] = df.loc[df.event == 'GL_TRIAL_GENSIDE'].value.astype('float')
        df.gen_side = df.gen_side.fillna(method='ffill')
        df['stim_id'] = df.loc[df.event == 'GL_TRIAL_STIM_ID'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_ONSET'].index).value.astype('float')
        df['rule_resp'] = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).value.astype('float')
        df['reward'] = df.loc[df.event == 'GL_TRIAL_REWARD'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).value.astype('float')
        df['rt'] = df.loc[df.event == 'CHOICE_TRIAL_RT'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).value.astype('float')
        df = df.loc[df.event.isin(['GL_TRIAL_LOCATION',
                                   'CHOICE_TRIAL_ONSET',
                                   'CHOICE_TRIAL_STIMOFF',
                                   'CHOICE_TRIAL_RESP'])]
        df = df.loc[:, ['onset', 'event', 'value',
                        'belief', 'psi', 'LLR', 'gen_side',
                        'stim_id', 'rule_resp', 'trial_id', 'reward', 'rt']]
        df = df.reset_index(drop=True)
        asign = np.sign(df.belief.values)
        signchange = (np.roll(asign, 1) - asign)
        signchange[0] = 0
        df['switch'] = signchange != 0
        df['switch_right'] = -signchange / 2
        df['switch_left'] = signchange / 2
        df['point'] = df.event == 'GL_TRIAL_LOCATION'
        df['response'] = df.event == 'CHOICE_TRIAL_RESP'
        df['response_left'] = ((df.event == 'CHOICE_TRIAL_RESP') & (df.value == '0'))
        df['response_right'] = (df.event == 'CHOICE_TRIAL_RESP') & (df.value == '1')
        df['stimulus_horiz'] = (df.event == 'CHOICE_TRIAL_ONSET') & (df.stim_id == 0)
        df['stimulus_vert'] = (df.event == 'CHOICE_TRIAL_ONSET') & (df.stim_id == 1)
        df['stimulus'] = df.event == 'CHOICE_TRIAL_ONSET'
        df['rresp_left'] = (df.event == 'CHOICE_TRIAL_RESP') & (df.rule_resp == 0)
        df['rresp_right'] = (df.event == 'CHOICE_TRIAL_RESP') & (df.rule_resp == 1)
        df.onset = df.onset.astype(float)
        df = df.sort_values(by='onset')
        self.BehavDataframe = df

    def instructed(self):
        logs = gm.load_logs_bids(self.subject, self.session, self.bids_path, run='instructed')
        df = logs[self.run]
        df = df.loc[df.event.isin(['REWARDED_RULE_STIM', 'IR_STIM', 'IR_TRIAL_START',
                                   'CHOICE_TRIAL_ONSET', 'CHOICE_TRIAL_STIMOFF',
                                   'CHOICE_TRIAL_RESP', 'CHOICE_TRIAL_RT', 'CHOICE_TRIAL_RULE_RESP',
                                   'IR_TRIAL_REWARD'])]
        df.value = df.value.replace('n/a', np.nan)
        df.onset = df.onset.astype(float)
        df = df.sort_values(by='onset').reset_index(drop=True)
        if df.loc[df.event == 'CHOICE_TRIAL_ONSET'].value.values.astype(float).mean() > 1:  # In some subjects grating ID is decoded with 0 / 1 in others with 1 /2
            df['stim_id'] = df.loc[df.event == 'CHOICE_TRIAL_ONSET'].value.astype('float') - 1
        elif df.loc[df.event == 'CHOICE_TRIAL_ONSET'].value.values.astype(float).mean() < 1:
            df['stim_id'] = df.loc[df.event == 'CHOICE_TRIAL_ONSET'].value.astype('float')
        df['rule_resp'] = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).value.astype('float')
        df['rt'] = df.loc[df.event == 'CHOICE_TRIAL_RT'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).value.astype('float')
        df['reward'] = df.loc[df.event == 'IR_TRIAL_REWARD'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).value.astype('float')

        df = df.loc[df.event.isin(['REWARDED_RULE_STIM',
                                   'CHOICE_TRIAL_ONSET',
                                   'CHOICE_TRIAL_STIMOFF',
                                   'CHOICE_TRIAL_RESP'])]

        df = df.loc[:, ['onset', 'event', 'value', 'rt', 'rewarded_rule',
                        'stim_id', 'rule_resp', 'reward']].reset_index(drop=True)
        df.rewarded_rule = df.rewarded_rule.ffill()
        df.value = df.value.astype(float)

        df['switch'] = df.index.isin(np.where(np.diff(df.rewarded_rule.values) != 0)[0] + 1)
        df['switch_right'] = df.index.isin(np.where(np.diff(df.rewarded_rule.values) == 1)[0] + 1)
        df['switch_left'] = df.index.isin(np.where(np.diff(df.rewarded_rule.values) == -1)[0] + 1)
        df['response'] = df.event == 'CHOICE_TRIAL_RESP'
        df['response_left'] = (df.event == 'CHOICE_TRIAL_RESP') & (df.value == 0)
        df['response_right'] = (df.event == 'CHOICE_TRIAL_RESP') & (df.value == 1)
        df['stimulus_horiz'] = (df.event == 'CHOICE_TRIAL_ONSET') & (df.stim_id == 0)
        df['stimulus_vert'] = (df.event == 'CHOICE_TRIAL_ONSET') & (df.stim_id == 1)
        df['stimulus'] = df.event == 'CHOICE_TRIAL_ONSET'
        df['rresp_left'] = (df.event == 'CHOICE_TRIAL_RESP') & (df.rule_resp == 0)
        df['rresp_right'] = (df.event == 'CHOICE_TRIAL_RESP') & (df.rule_resp == 1)
        self.BehavDataframe = df


@memory.cache
def execute(subject, session, run, type, flex_dir, summary):
    summary = summary
    bd = BehavDataframe(subject, session, run, flex_dir)
    if type == 'inference':
        bd.inference(summary=summary)
    elif type == 'instructed':
        bd.instructed()
    return bd.BehavDataframe


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


@memory.cache
def fmri_align(BehavDf, task):
    '''
    Output: pd.DataFrame with
                - parameters as columns
                - timedelta as index
                - convolved with hrf
                - downsampled to EPI-f
    '''

    b = BehavDf
    b.onset = b.onset.astype(float)
    b = b.sort_values(by='onset')
    if task == 'inference':
        b = b.loc[:, ['onset', 'switch', 'switch_left', 'switch_right',
                      'belief', 'LLR', 'surprise',
                      'point', 'response', 'response_left', 'response_right',
                      'stimulus_horiz', 'stimulus_vert', 'stimulus',
                      'rresp_left', 'rresp_right']]
    elif task == 'instructed':
        b = b.loc[:, ['onset', 'switch_left', 'switch_right', 'switch',
                      'response', 'response_left', 'response_right',
                      'stimulus_horiz', 'stimulus_vert', 'stimulus',
                      'rresp_left', 'rresp_right']]
    b = b.set_index((b.onset.values * 1000).astype(int)).drop('onset', axis=1)
    b = b.reindex(pd.Index(np.arange(0, b.index[-1] + 15000, 1)))
    b.loc[0] = 0
    if task == 'inference':
        b.belief = b.belief.fillna(method='ffill')
        b['belief_right'] = b.belief
        b['belief_left'] = -b.belief
        b['belief'] = b.belief_right.abs()
        b['LLR_right'] = b.LLR
        b['LLR_left'] = -b.LLR
        b['LLR'] = b.LLR.abs()
    b = b.fillna(False).astype(float)
    for column in b.columns:
        b[column] = make_bold(b[column].values, dt=.001)
    b = regular(b, target='1900ms')
    b.loc[pd.Timedelta(0)] = 0
    b = b.sort_index()
    return b
