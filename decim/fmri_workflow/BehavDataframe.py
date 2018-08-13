import pandas as pd
import numpy as np
from decim import glaze_control as gc
from decim import glaze2 as gl
from os.path import join
from scipy.interpolate import interp1d
'''
from joblib import Memory
cachedir = '/Users/kenohagena/Flexrule/cachedir'
memory = Memory(cachedir=cachedir, verbose=0)
'''
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

    def inference(self, summary=summary):

        logs = gc.load_logs_bids(self.subject, self.session, self.bids_path)
        df = logs[self.run]
        H = summary.loc[(summary.subject == self.subject) & (summary.session == self.session)].hmode.values[0]
        df['belief'] = gl.belief(df, H=H, ident='event')
        df.belief = df.belief.fillna(method='ffill')
        df['obj_belief'] = gl.belief(df, H=1 / 70, ident='event')
        df = df.loc[df.event.isin(['GL_TRIAL_LOCATION', 'GL_TRIAL_GENSIDE',
                                   'GL_TRIAL_STIM_ID', 'CHOICE_TRIAL_ONSET',
                                   'CHOICE_TRIAL_STIMOFF', 'CHOICE_TRIAL_RESP',
                                   'CHOICE_TRIAL_RULE_RESP', 'GL_TRIAL_START',
                                   'GL_TRIAL_REWARD', 'CHOICE_TRIAL_RT'])]
        df = df.reset_index()
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
                        'belief', 'obj_belief', 'gen_side',
                        'stim_id', 'rule_resp', 'trial_id', 'reward', 'rt']]
        df = df.reset_index(drop=True)
        df['LLR'] = np.nan
        df.loc[df.event == 'GL_TRIAL_LOCATION', 'LLR'] =\
            gl.LLR(df.loc[df.event == 'GL_TRIAL_LOCATION'].value.astype(float).values)
        df['test'] = np.convolve(df.belief.values, [1, 1], 'same')
        df['test2'] = np.convolve(df.belief.abs().values, [1, 1], 'same')
        df['switch'] = df.test.abs() < df.test2.abs()
        df = df.drop(['test', 'test2'], axis=1)
        df['prior_belief'] = np.nan
        df.loc[df.event == 'GL_TRIAL_LOCATION', 'prior_belief'] =\
            np.roll(df.loc[df.event == 'GL_TRIAL_LOCATION', 'belief'].values, 1)
        df.loc[0, 'prior_belief'] = 0
        df['murphy_surprise'] = np.nan
        df.loc[df.event == 'GL_TRIAL_LOCATION', 'murphy_surprise'] =\
            gl.murphy_surprise(df.loc[df.event == 'GL_TRIAL_LOCATION'].prior_belief.values, df.loc[df.event == 'GL_TRIAL_LOCATION'].belief.values)
        df['point'] = df.event == 'GL_TRIAL_LOCATION'
        df['response'] = df.event == 'CHOICE_TRIAL_RESP'
        df['response_left'] = ((df.event == 'CHOICE_TRIAL_RESP') & (df.value == '0'))
        df['response_right'] = (df.event == 'CHOICE_TRIAL_RESP') & (df.value == '1')
        df['stimulus_horiz'] = (df.event == 'CHOICE_TRIAL_ONSET') & (df.stim_id == 0)
        df['stimulus_vert'] = (df.event == 'CHOICE_TRIAL_ONSET') & (df.stim_id == 1)
        df['stimulus'] = df.event == 'CHOICE_TRIAL_ONSET'
        df['rresp_left'] = (df.event == 'CHOICE_TRIAL_RESP') & (df.rule_resp == 0)
        df['rresp_right'] = (df.event == 'CHOICE_TRIAL_RESP') & (df.rule_resp == 1)
        df['belief_left'] = - df.belief
        df.onset = df.onset.astype(float)
        df = df.sort_values(by='onset')
        self.BehavDataframe = df

    def instructed(self):
        logs = gc.load_logs_bids(self.subject, self.session, self.bids_path, run='instructed')
        df = logs[self.run]
        df = df.loc[df.event.isin(['REWARDED_RULE_STIM', 'IR_STIM', 'IR_TRIAL_START',
                                   'CHOICE_TRIAL_ONSET', 'CHOICE_TRIAL_STIMOFF',
                                   'CHOICE_TRIAL_RESP', 'CHOICE_TRIAL_RT', 'CHOICE_TRIAL_RULE_RESP',
                                   'IR_TRIAL_REWARD'])]
        df.value = df.value.replace('n/a', np.nan)
        df.onset = df.onset.astype(float)
        df = df.sort_values(by='onset').reset_index(drop=True)
        df['stim_id'] = df.loc[df.event == 'CHOICE_TRIAL_ONSET'].value.astype('float') - 1
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


#@memory.cache
def execute(subject, session, run, type, flex_dir, summary):
    summary = summary
    bd = BehavDataframe(subject, session, run, flex_dir)
    if type == 'inference':
        bd.inference()
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


#@memory.cache
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
        b = b.loc[:, ['onset', 'belief', 'murphy_surprise', 'switch', 'point', 'response', 'response_left',
                      'response_right', 'stimulus_horiz', 'stimulus_vert', 'stimulus',
                      'rresp_left', 'rresp_right', 'LLR']]
    elif task == 'instructed':
        b = b.loc[:, ['onset', 'switch_left', 'switch_right', 'switch', 'response', 'response_left',
                      'response_right', 'stimulus_horiz', 'stimulus_vert', 'stimulus',
                      'rresp_left', 'rresp_right']]

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
    return b
