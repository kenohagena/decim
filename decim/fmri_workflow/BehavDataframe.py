import pandas as pd
import numpy as np
from decim import glaze_model as gm
from os.path import join, expanduser
from scipy.interpolate import interp1d
from decim import slurm_submit as slu
import matplotlib.pyplot as plt
from joblib import Memory
if expanduser('~') == '/home/faty014':
    cachedir = expanduser('/work/faty014/joblib_cache')
else:
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


codings:
    - stimulus: -1 --> vertical, 1 --> horizontal
    - rule: -1 --> vertical-left (A), 1 --> vertical-right (B)
    - response: -1 --> left, 1 --> right
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
        H = summary.loc[(summary.subject == self.subject) &
                        (summary.session == self.session)].hmode.values[0]
        df['belief'], psi, df['LLR'], df['surprise'] =\
            gm.belief(df, H=H, ident='event')
        df = df.loc[df.event.isin(['GL_TRIAL_LOCATION', 'GL_TRIAL_GENSIDE',
                                   'GL_TRIAL_STIM_ID', 'CHOICE_TRIAL_ONSET',
                                   'CHOICE_TRIAL_STIMOFF', 'CHOICE_TRIAL_RESP',
                                   'CHOICE_TRIAL_RULE_RESP', 'GL_TRIAL_START',
                                   'GL_TRIAL_REWARD', 'CHOICE_TRIAL_RT'])]
        df = df.reset_index()
        df = df.replace('n/a', np.nan)
        df['trial_id'] = df.loc[df.event == 'GL_TRIAL_START'].value.astype(int)
        df.trial_id = df.trial_id.fillna(method='ffill')
        df['gen_side'] = df.loc[df.event == 'GL_TRIAL_GENSIDE'].\
            value.astype('float')
        df.gen_side = df.gen_side.fillna(method='ffill')
        df['stimulus'] = df.loc[df.event == 'GL_TRIAL_STIM_ID'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_ONSET'].index).\
            value.astype('float') * (-2) + 1                                    # horiz -> -1 | vert -> +1
        df['stimulus_off'] = df.event == 'CHOICE_TRIAL_STIMOFF'
        df['rule_resp'] = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).\
            value.astype('float') * 2 - 1                                       # vertical-left (A) --> -1 | --> vertical-right (B) --> 1
        df['reward'] = df.loc[df.event == 'GL_TRIAL_REWARD'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).\
            value.astype('float')
        df['rt'] = df.loc[df.event == 'CHOICE_TRIAL_RT'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).\
            value.astype('float')
        df = df.loc[df.event.isin(['GL_TRIAL_LOCATION',
                                   'CHOICE_TRIAL_ONSET',
                                   'CHOICE_TRIAL_STIMOFF',
                                   'CHOICE_TRIAL_RESP'])]
        df = df.loc[:, ['onset', 'event', 'value',
                        'belief', 'LLR', 'gen_side',
                        'stimulus', 'stimulus_off', 'rule_resp',
                        'trial_id', 'reward', 'rt', 'surprise']]
        df = df.reset_index(drop=True)
        asign = np.sign(df.belief.values)
        signchange = (np.roll(asign, 1) - asign)
        signchange[0] = 0
        df['switch'] = signchange / 2
        df['point'] = df.event == 'GL_TRIAL_LOCATION'
        df['response'] = df.event == 'CHOICE_TRIAL_RESP'
        df.loc[df.response == True, 'response'] =\
            df.loc[df.response == True, 'value'].\
            astype(float) * 2 - 1                                               # left -> -1 | right -> +1
        df.onset = df.onset.astype(float)
        df = df.sort_values(by='onset')
        self.BehavDataframe = df

    def instructed(self):
        logs = gm.load_logs_bids(self.subject, self.session,
                                 self.bids_path, run='instructed')
        df = logs[self.run]
        df['rt'] = df.loc[df.event == 'CHOICE_TRIAL_RT'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).\
            value.astype('float')
        df = df.loc[df.event.isin(['REWARDED_RULE_STIM',
                                   'CHOICE_TRIAL_ONSET',
                                   'CHOICE_TRIAL_STIMOFF',
                                   'CHOICE_TRIAL_RESP'])]
        df.value = df.value.replace('n/a', np.nan)
        df.onset = df.onset.astype(float)
        df = df.sort_values(by='onset').reset_index(drop=True)

        df.loc[df.event != 'REWARDED_RULE_STIM', 'rewarded_rule'] = np.nan      # Error in matlab code for first 6 subjects
        df.rewarded_rule = df.rewarded_rule.fillna(method='ffill') * 2 - 1

        if df.loc[df.event == 'CHOICE_TRIAL_ONSET'].\
                value.values.astype(float).mean() > 1:                          # In some subjects grating ID is decoded with 0 / 1 in others with 2 /1
            df['stimulus'] = df.loc[df.event == 'CHOICE_TRIAL_ONSET'].\
                value.astype('float') * (2) - 3
        elif df.loc[df.event == 'CHOICE_TRIAL_ONSET'].\
                value.values.astype(float).mean() < 1:
            df['stimulus'] = df.loc[df.event == 'CHOICE_TRIAL_ONSET'].\
                value.astype('float') * (-2) + 1
        df.stimulus = df.stimulus.fillna(method='ffill', limit=4)
        df['response'] = df.event == 'CHOICE_TRIAL_RESP'
        df.loc[df.response == True, 'response'] =\
            df.loc[df.response == True, 'value'].astype(float) * 2 - 1          # left -> -1 | right -> +1
        df.loc[(df.event == 'CHOICE_TRIAL_RESP'), 'rule_resp'] =\
            -np.abs(df.loc[(df.event == 'CHOICE_TRIAL_RESP')].stimulus +
                    df.loc[(df.event == 'CHOICE_TRIAL_RESP')].response) + 1
        df.loc[(df.event == 'CHOICE_TRIAL_RESP'), 'reward'] =\
            df.loc[(df.event == 'CHOICE_TRIAL_RESP')].rule_resp ==\
            df.loc[(df.event == 'CHOICE_TRIAL_RESP')].rewarded_rule

        df = df.loc[:, ['onset', 'event', 'rt', 'rewarded_rule', 'response',
                        'stimulus', 'stimulus_off', 'rule_resp', 'reward']].\
            reset_index(drop=True)

        df['switch'] = np.append([0], np.diff(df.rewarded_rule.values)) / 2

        try:                                                                    # Sanity check I
            assert all(df.loc[df.switch != 0, 'event'] == 'REWARDED_RULE_STIM')
        except AssertionError:
            print('''AssertionError:
                Some switches do NOT coincide with rewarded rule stimulus''')
        try:                                                                    # Sanity check II
            assert df.loc[df.event == 'CHOICE_TRIAL_RESP'].reward.mean() > .8
        except AssertionError:
            print('AssertionError: Performance in instructed run at {}'.
                  format(df.loc[df.event == 'CHOICE_TRIAL_RESP'].reward.mean()))

        self.BehavDataframe = df


#@memory.cache
def execute(subject, session, run, type, flex_dir, summary):
    summary = summary
    bd = BehavDataframe(subject, session, run, flex_dir)
    if type == 'inference':
        bd.inference(summary=summary)
    elif type == 'instructed':
        bd.instructed()
    return bd.BehavDataframe


'''
b = BehavDataframe('sub-19', 'ses-3', 'inference_run-4', '/Volumes/flxrl/FLEXRULE')
b.inference(pd.read_csv('/Volumes/flxrl/FLEXRULE/behavior/bids_stan_fits/summary_stan_fits.csv'))
# b.instructed()
f = fmri_align(b.BehavDataframe, 'inference', fast=True)
print(f.stimulus_rr_horiz.mean())
'''


print(execute('sub-3', 'ses-3', 'instructed_run-7', 'instructed', '/Volumes/flxrl/FLEXRULE', 'akdsjlekwjf'))
