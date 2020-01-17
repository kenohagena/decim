import pandas as pd
import numpy as np
from decim.fmri_workflow import LinregVoxel
from decim.adjuvant import glaze_model as gm
from os.path import join, expanduser
from decim.adjuvant import slurm_submit as slu
from joblib import Memory
if expanduser('~') == '/home/faty014':
    cachedir = expanduser('/work/faty014/joblib_cache')
else:
    cachedir = expanduser('~/joblib_cache')
slu.mkdir_p(cachedir)
memory = Memory(cachedir=cachedir, verbose=0)


'''
This script loads and computes behavioral data per subject, session and run:

1. Load data from raw BIDS directory (using script glaze_model)
2. Retrieve fitted H from summary file
3. Compute Glaze belief, LLR, psi, surprise (using script glaze_model)
4. Restructure in the following format:
    - rows: timepoints
    - columns:
        a) event (point / choice onset / response / stimulus onset)
        b) onset (time of event)
        c) value of event (point location or choice)
        d) Glaze belief, LLR, surprise
        e) gen_side (currently active distribution)
        g) stimulus (-1 --> vertical, 1 --> horizontal, 0 --> no stimulus)
        h) stimulus_off (dummy-coded when grating stimulus disappears)
        i) response (-1 --> left, 1 --> right, 0 --> no response)
        j) rule_response (-1 --> vertical-left (A), 1 --> vertical-right (B))
        k) switch (belief right-to-left --> 1, belief left-to-right --> -1)
        l) reward


EXAMPLE:

behav_df = execute(subject='sub-17', session='ses-2',
run='inference_run-4', task='inference', flex_dir='/Volumes/flxrl/FLEXRULE/',
summary=pd.read_csv('/Volumes/flxrl/FLEXRULE/behavior/bids_stan_fits/summary_stan_fits.csv'))

'''


class BehavDataframe(object):
    '''
    Inititialize

    - Arguments:
        a) subject (e.g. 'sub-17')
        b) session (e.g. 'ses-2')
        c) run (e.g. 'inference_run-4')
        d) Flexrule directory
    '''

    def __init__(self, subject, session, run, flex_dir):
        self.subject = subject
        self.session = session
        self.run = run
        self.bids_path = join(flex_dir, 'raw', 'bids_mr_v1.2')
        self.flex_dir = flex_dir

    def inference(self, summary, Hs=[]):
        logs = gm.load_logs_bids(self.subject, self.session, self.bids_path)    # Load data from raw directory
        df = logs[self.run]
        H = summary.loc[(summary.subject == self.subject) &                     # Retrieve fitted H
                        (summary.session == self.session)].hmode.values[0]
        df['belief'], psi, df['LLR'], df['surprise'] =\
            gm.belief(df, H=H, ident='event')                                   # Compute belief, LLR, surprise
        for different_H in Hs:
            df['belief_{}'.format(different_H)] = gm.belief(df, H=different_H, ident='event')[0]
        df = df.loc[df.event.isin(['GL_TRIAL_LOCATION', 'GL_TRIAL_GENSIDE',
                                   'GL_TRIAL_STIM_ID', 'CHOICE_TRIAL_ONSET',
                                   'CHOICE_TRIAL_STIMOFF', 'CHOICE_TRIAL_RESP',
                                   'CHOICE_TRIAL_RULE_RESP', 'GL_TRIAL_START',
                                   'GL_TRIAL_REWARD', 'CHOICE_TRIAL_RT'])]
        if 0 in Hs:
            df.loc[df.event == 'GL_TRIAL_LOCATION', 'first_sample'] = 0
            df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP', 'first_sample'] = 1
            df.loc[df.index[0], 'first_sample'] = 1
            df.first_sample = np.roll(df.first_sample.fillna(method='ffill').values, 1)
            firsts = df.loc[(df.event == 'GL_TRIAL_LOCATION') & (df.first_sample == 1)].index.values
            df.loc[:, 'belief_reset'] = gm.belief(df, H=0, ident='event', reset_firsts=firsts)[0]

        df = df.reset_index()
        df = df.replace('n/a', np.nan)
        df['trial_id'] = df.loc[df.event == 'GL_TRIAL_START'].value.astype(int)
        df.trial_id = df.trial_id.fillna(method='ffill')
        df['gen_side'] = df.loc[df.event == 'GL_TRIAL_GENSIDE'].\
            value.astype('float')
        df.gen_side = df.gen_side.fillna(method='ffill')
        df['stimulus'] = df.loc[df.event == 'GL_TRIAL_STIM_ID'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_ONSET'].index).\
            value.astype('float') * (-2) + 1                                    # stimulus coding: horiz -> -1 | vert -> +1
        df['stimulus_off'] = df.event == 'CHOICE_TRIAL_STIMOFF'
        df['rule_resp'] = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].\
            set_index(df.loc[df.event == 'CHOICE_TRIAL_RESP'].index).\
            value.astype('float') * 2 - 1                                       # rule_reponse coding: vertical-left (A) --> -1 | vertical-right (B) --> 1
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
        cols = ['onset', 'event', 'value',
                'belief', 'LLR', 'gen_side',
                'stimulus', 'stimulus_off', 'rule_resp',
                'trial_id', 'reward', 'rt', 'surprise'] +\
            ['belief_{}'.format(different_H) for different_H in Hs + ['reset']]
        df = df.loc[:, cols]
        df = df.reset_index(drop=True)
        asign = np.sign(df.belief.values)
        signchange = (np.roll(asign, 1) - asign)                                # switch direction coding: from left to right --> -1 | from right to left --> +1
        signchange[0] = 0
        df['switch'] = signchange / 2
        df['point'] = df.event == 'GL_TRIAL_LOCATION'
        df['response'] = df.event == 'CHOICE_TRIAL_RESP'
        df.loc[df.response == True, 'response'] =\
            df.loc[df.response == True, 'value'].\
            astype(float) * 2 - 1                                               # response coding: left -> -1 | right -> +1
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
                value.values.astype(float).mean() > 1:                          # In some subjects grating ID is encoded with 0 / 1 in others with 2 /1
            df['stimulus'] = df.loc[df.event == 'CHOICE_TRIAL_ONSET'].\
                value.astype('float') * (2) - 3
        elif df.loc[df.event == 'CHOICE_TRIAL_ONSET'].\
                value.values.astype(float).mean() < 1:
            df['stimulus'] = df.loc[df.event == 'CHOICE_TRIAL_ONSET'].\
                value.astype('float') * (-2) + 1
        df['response'] = df.event == 'CHOICE_TRIAL_RESP'
        df.loc[df.response == True, 'response'] =\
            df.loc[df.response == True, 'value'].astype(float) * 2 - 1          # response coding: left -> -1 | right -> +1
        df.loc[(df.event == 'CHOICE_TRIAL_RESP'), 'rule_resp'] =\
            -np.abs(df.loc[(df.event == 'CHOICE_TRIAL_ONSET')].stimulus.values +
                    df.loc[(df.event == 'CHOICE_TRIAL_RESP')].response.values) + 1
        df.loc[(df.event == 'CHOICE_TRIAL_RESP'), 'reward'] =\
            df.loc[(df.event == 'CHOICE_TRIAL_RESP')].rule_resp ==\
            df.loc[(df.event == 'CHOICE_TRIAL_RESP')].rewarded_rule

        df = df.loc[:, ['onset', 'event', 'rt', 'rewarded_rule', 'response',
                        'stimulus', 'stimulus_off', 'rule_resp', 'reward', 'value']].\
            reset_index(drop=True)

        df['switch'] = np.append([0], np.diff(df.rewarded_rule.values)) / 2

        try:                                                                    # Sanity check I (in instructed rule runs, all switches should occur when rewarded rule stimulus is shown)
            assert all(df.loc[df.switch != 0, 'event'] == 'REWARDED_RULE_STIM')
        except AssertionError:
            print('''AssertionError:
                Some switches do NOT coincide with rewarded rule stimulus''')
        try:                                                                    # Sanity check II (subject should have performance rates near 100%)
            assert df.loc[df.event == 'CHOICE_TRIAL_RESP'].reward.mean() > .8
        except AssertionError:
            print('AssertionError: Performance in instructed run at {}'.
                  format(df.loc[df.event == 'CHOICE_TRIAL_RESP'].reward.mean()))

        self.BehavDataframe = df


def realign_to_TR(behav, convolve_hrf=False, task='instructed'):
    if task == 'inference':
        dm = behav.loc[:, ['belief', 'onset']]
    elif task == 'instructed':
        dm = behav.loc[:, ['rewarded_rule', 'onset']]
    dm = dm.set_index((dm.onset.values * 1000).
                      astype(int)).drop('onset', axis=1)
    dm = dm.reindex(pd.Index(np.arange(0, dm.index[-1] + 15000, 1)))
    dm = dm.fillna(method='ffill', limit=99)

    dm = dm.loc[np.arange(dm.index[0],
                          dm.index[-1], 100)]
    dm.loc[0] = 0
    dm = dm.fillna(method='ffill')
    if convolve_hrf is True:
        for column in dm.columns:
            print('Align ', column)
            dm[column] = LinregVoxel.make_bold(dm[column].values, dt=.1)

    dm = LinregVoxel.regular(dm, target='1900ms')
    dm.loc[pd.Timedelta(0)] = 0
    dm = dm.sort_index()
    return dm


#@memory.cache
def execute(subject, session, run, task, flex_dir, summary, belief_TR=False):
    '''
    Execute this script.

    - Arguments;
        a) subject (e.g. 'sub-17')
        b) session (e.g. 'ses-2')
        c) run (e.g. 'inference_run-4')
        d) task ('inference' or "instructed")
        e) Flexrule directory
        f) summary file with fitted hazard rates
            (only important when running for task == 'inference')

    - Output: pd.DataFrame with behavior
    '''
    summary = summary
    bd = BehavDataframe(subject, session, run, flex_dir)
    if task == 'inference':
        bd.inference(summary=summary, Hs=[0, .5, .014])
    elif task == 'instructed':
        bd.instructed()
    if belief_TR is True:
        bd.BehavDataframe = realign_to_TR(bd.BehavDataframe, task=task)
    return bd.BehavDataframe


'''
flex_dir = '/Volumes/flxrl/FLEXRULE'
subject = 'sub-17'
session = 'ses-3'
run = 'instructed_run-7'
task = 'instructed'
summary = pd.read_csv('/Users/kenohagena/Flexrule/fmri/analyses/bids_stan_fits/summary_stan_fits.csv')
print(execute(subject, session, run, task, flex_dir, summary, belief_TR=False).head())
'''
