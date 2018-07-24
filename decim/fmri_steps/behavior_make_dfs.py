import pandas as pd
import numpy as np
from decim import glaze_control as gc
from decim import glaze2 as gl
from decim import slurm_submit as slurm
from os.path import expanduser, join

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
bids_mr = '/Volumes/flxrl/fmri/bids_mr_v1.1/'
outpath = expanduser('~/Flexrule/fmri/analyses/behav_dataframes')
summary = pd.read_csv('/Users/kenohagena/Flexrule/fmri/analyses/bids_stan_fits/summary_stan_fits.csv')
subjects = range(1, 23)
sessions = [1, 2, 3]

for sub in subjects:
    subject = 'sub-{}'.format(sub)
    savepath = join(outpath, '1406', subject)
    slurm.mkdir_p(savepath)
    for ses in sessions:  # range(1, 4):
        session = 'ses-{}'.format(ses)
        try:
            logs = gc.load_logs_bids(sub, ses, bids_mr)
            logs = logs[1]
            for df in logs:
                run = str(df.iloc[25].block)[0]  # '25' is quite arbitrary
                belief = gl.belief(df, H=summary.loc[(summary.subject ==
                                                      'sub-1') & (summary.session == 'ses-1')].hmode.values[0], ident='event')
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
                df = df.sort_index(by='onset').to_csv(join(savepath, 'behav_{0}_{1}_run-{2}.csv'.format(subject, session, run)))
        except RuntimeError:
            pass
