import pandas as pd
import numpy as np
from decim import glaze_control as gc
from decim import glaze2 as gl
from decim import slurm_submit as slurm
from os.path import expanduser, join
from collections import defaultdict

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


# SET OPTIONS
runs = ['inference_run-4', 'inference_run-5', 'inference_run-6', 'instructed_run-7', 'instructed_run-8']
bids_mr = '/Volumes/flxrl/FLEXRULE/raw/bids_mr_v1.1/'
outpath = '/Volumes/flxrl/FLEXRULE/behavior/behav_dataframes/'
summary = pd.read_csv('/Users/kenohagena/Flexrule/fmri/analyses/bids_stan_fits/summary_stan_fits.csv')
subjects = range(1, 23)
sessions = [1, 2, 3]


class BehavDataframe(object):

    def __init__(self, sub, ses, bids_path, out_path):
        self.sub = sub
        self.ses = ses
        self.subject = 'sub-{}'.format(sub)
        self.session = 'ses-{}'.format(ses)
        self.out_path = out_path
        self.bids_path = bids_path
        slurm.mkdir_p(self.out_path)
        self.BehavDataframe = defaultdict(dict)

    def inference_run(self, summary=summary):

        logs = gc.load_logs_bids(self.sub, self.ses, self.bids_path)
        logs = logs[1]
        for df in logs:
            run = str(df.iloc[25].block)[0]  # '25' is quite arbitrary
            H = summary.loc[(summary.subject == self.subject) & (summary.session == self.session)].hmode.values[0]
            belief = gl.belief(df, H=H, ident='event')
            df['belief'] = belief
            df['obj_belief'] = gl.belief(df, H=1 / 70, ident='event')
            df = df.loc[df.event.isin(['GL_TRIAL_LOCATION', 'GL_TRIAL_GENSIDE',
                                       'GL_TRIAL_STIM_ID', 'CHOICE_TRIAL_ONSET',
                                       'CHOICE_TRIAL_STIMOFF', 'CHOICE_TRIAL_RESP',
                                       'CHOICE_TRIAL_RULE_RESP'])]
            df = df.reset_index()
            df.value = df.value.replace('n/a', np.nan).astype(float)
            df['gen_side'] = df.loc[df.event == 'GL_TRIAL_GENSIDE'].\
                set_index(df.loc[df.event == 'GL_TRIAL_GENSIDE'].index + 1).value.astype('float')
            df['stim_id'] = df.loc[df.event == 'GL_TRIAL_STIM_ID'].\
                set_index(df.loc[df.event == 'GL_TRIAL_STIM_ID'].index + 2).value.astype('float')
            df['rule_resp'] = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].\
                set_index(df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].index - 1).value.astype('float')
            df = df.loc[df.event.isin(['GL_TRIAL_LOCATION',
                                       'CHOICE_TRIAL_ONSET',
                                       'CHOICE_TRIAL_STIMOFF',
                                       'CHOICE_TRIAL_RESP'])]
            df = df.loc[:, ['onset', 'event', 'value',
                            'belief', 'obj_belief', 'gen_side',
                            'stim_id', 'rule_resp']]
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
            df['event'] = df['event'].map({'GL_TRIAL_LOCATION': 'point', 'CHOICE_TRIAL_ONSET': 'choice_onset',
                                           'CHOICE_TRIAL_RESP': 'response', 'CHOICE_TRIAL_STIMOFF': 'stimulus_offset'})
            df.loc[df.event == 'point', 'murphy_surprise'] =\
                gl.murphy_surprise(df.loc[df.event == 'point'].prior_belief.values, df.loc[df.event == 'point'].belief.values)
            df['point'] = df.event == 'point'
            df['response'] = df.event == 'response'
            df['response_left'] = ((df.event == 'response') & (df.value == 0))
            df['response_right'] = (df.event == 'response') & (df.value == 1)
            df['stimulus_horiz'] = (df.event == 'choice_onset') & (df.stim_id == 0)
            df['stimulus_vert'] = (df.event == 'choice_onset') & (df.stim_id == 1)
            df['stimulus'] = df.event == 'choice_onset'
            df['rresp_left'] = (df.event == 'response') & (df.rule_resp == 0)
            df['rresp_right'] = (df.event == 'response') & (df.rule_resp == 1)
            df['belief_left'] = - df.belief
            df = df.sort_index(by='onset')
            self.BehavDataframe['inference'][run] = df
            df.to_hdf(join(self.out_path, 'behav_{0}_{1}.hdf'.format(self.subject, self.session)), key=runs[int(run) - 4])

    def instructed(self):
        logs = gc.load_logs_bids(self.sub, self.ses, self.bids_path, run='instructed')
        logs = logs[1]
        for df in logs:
            run = str(df.iloc[25].block)[0]
            df = df.loc[df.event.isin(['START_IR', 'SUBJECT', 'IR_STIM',
                                       'IR_REWARDED_RULE', 'IR_TRIAL_START', 'IR_TRIAL_FIXON',
                                       'CHOICE_TRIAL_ONSET', 'CHOICE_TRIAL_STIMOFF', 'BUTTON_PRESS',
                                       'CHOICE_TRIAL_RESP', 'CHOICE_TRIAL_RT', 'CHOICE_TRIAL_RULE_RESP',
                                       'IR_TRIAL_REWARD'])]
            df = df.reset_index()
            df.value = df.value.replace('n/a', np.nan)
            df['stim_id'] = df.loc[df.event == 'CHOICE_TRIAL_ONSET'].\
                set_index(df.loc[df.event == 'CHOICE_TRIAL_ONSET'].index).value.astype('float')
            df['rule_resp'] = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].\
                set_index(df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].index - 1).value.astype('float')
            df['rule_resp'] = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].\
                set_index(df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].index - 2).value.astype('float')
            df['rt'] = df.loc[df.event == 'CHOICE_TRIAL_RT'].\
                set_index(df.loc[df.event == 'CHOICE_TRIAL_RT'].index - 1).value.astype('float')
            df = df.loc[df.event.isin(['IR_REWARDED_RULE',
                                       'CHOICE_TRIAL_ONSET',
                                       'CHOICE_TRIAL_STIMOFF',
                                       'CHOICE_TRIAL_RESP'])]
            df['rewarded_rule'] = df.loc[df.event == 'IR_REWARDED_RULE'].\
                set_index(df.loc[df.event == 'IR_REWARDED_RULE'].index).value.astype('float')
            df = df.loc[:, ['onset', 'event', 'value', 'rt', 'rewarded_rule',
                            'stim_id', 'rule_resp']].reset_index(drop=True)
            df.rewarded_rule = df.rewarded_rule.ffill()
            df['switch'] = df.index.isin(np.where(np.diff(df.rewarded_rule.values) != 0)[0][1:] + 1)
            df['switch_right'] = df.index.isin(np.where(np.diff(df.rewarded_rule.values) == 1)[0] + 1)
            df['switch_left'] = df.index.isin(np.where(np.diff(df.rewarded_rule.values) == -1)[0] + 1)
            df['event'] = df['event'].map({'IR_REWARDED_RULE': 'rewarded_rule', 'IR_TRIAL_FIXON': 'fixation_on', 'CHOICE_TRIAL_ONSET': 'choice_onset',
                                           'CHOICE_TRIAL_RESP': 'response', 'CHOICE_TRIAL_STIMOFF': 'stimulus_offset'})
            df['response'] = df.event == 'response'
            df['response_left'] = ((df.event == 'response') & (df.value == 0))
            df['response_right'] = (df.event == 'response') & (df.value == 1)
            df['stimulus_horiz'] = (df.event == 'choice_onset') & (df.stim_id == 0)
            df['stimulus_vert'] = (df.event == 'choice_onset') & (df.stim_id == 1)
            df['stimulus'] = df.event == 'choice_onset'
            df['rresp_left'] = (df.event == 'response') & (df.rule_resp == 0)
            df['rresp_right'] = (df.event == 'response') & (df.rule_resp == 1)
            df = df.sort_index(by='onset')
            self.BehavDataframe['instructed'][run] = df
            df.to_hdf(join(self.out_path, 'behav_{0}_{1}.hdf'.format(self.subject, self.session)), key=runs[int(run) - 4])


if __name__ == '__main__':
    for sub in range(1, 23):
        print(sub)
        for ses in [2, 3]:
            try:
                cl = Behav_Dataframes(sub, ses, bids_mr, outpath)
                cl.inference_run(summary=summary)
            except RuntimeError:
                print('Runtime')
                continue
