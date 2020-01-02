from itertools import product
import pandas as pd
import numpy as np
from scipy.io import loadmat


'''
1. Load .tsv file from bids_dir
2. Load sequences.mat file
3. Add entries for showing of rewarded rule grating
4. Save as .csv

'''


def repair_tsv(sub, ses, run):
    '''
    subjects 1 & 2 --> connectivity_sequences.mat
    subjects 3 - 6 --> new_connectivity_sequences-S1-S6.mat
    subjects > 7  -- > new_connectivity_sequences-S7andfollowing.mat

    For subjects 1 & 2 the sequences.mat has just the ISIs (not the absolute onset times)>
    When the rewarded stimulus changes and the block change stimulus is shown, intervals are drawn as follows:
    STIMOFF(n) --> ISI2 (n+1) --> block change stim (~2s) --> ISI1 (n+1) --> CHOICE_ONSET (n+1)

    sub-1, ses-2, run-7 took sequences from 1-2-8.
    '''
    if sub > 6:
        seq = loadmat('/Users/kenohagena/Flexrule/matlab/new_connectivity_sequences-S7andfollowing.mat')['sequences']
    elif sub < 3:
        seq = loadmat('/Users/kenohagena/Flexrule/matlab/connectivity_sequences.mat')['sequences']
    else:
        seq = loadmat('/Users/kenohagena/Flexrule/matlab/new_connectivity_sequences-S1-S6.mat')['sequences']
    df = pd.read_table('/Volumes/flxrl/FLEXRULE/raw/bids_mr_v1.2/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_task-instructed_run-{2}_events.tsv'.format(sub, ses, run))

    if sub < 3:
        blockchange = seq[0][sub - 1][0][ses - 1][0][run - 1][0][0][3][0].astype(bool)
        isi = seq[0][sub - 1][0][ses - 1][0][run - 1][0][0][5][0]
        isi2 = seq[0][sub - 1][0][ses - 1][0][run - 1][0][0][6][0]
        rew_rule = seq[0][sub - 1][0][ses - 1][0][run - 1][0][0][4][0]

        real_isi = df.loc[df.event == 'CHOICE_TRIAL_ONSET'].onset.values.astype(float) - df.loc[df.event == 'IR_TRIAL_START'].onset.values.astype(float)
        try:
            assert (real_isi - isi[0:len(real_isi)]).min() > -.05
            assert (real_isi - isi[0:len(real_isi)]).max() < .05
        except AssertionError:
            print(sub, ses, run, (real_isi - isi).min(), (real_isi - isi).max())

        first_stimulus = float(df.loc[df.event == '0'].iloc[4].onset) + isi2[0]

        following_stimuli = df.loc[df.event == 'CHOICE_TRIAL_STIMOFF'].onset.values.astype(float)[0:len(isi) - 1] + isi2[1:]

        block_change_onsets = np.append(first_stimulus, following_stimuli)
        block_change_onsets = block_change_onsets[blockchange]
        rewarded_rule_stimuli = rew_rule[blockchange]
        rewarded_rule_choice_trial = rew_rule

    else:
        if sub > 20:
            typ = seq[0][sub - 21][0][ses - 1][0][run - 1][0][0][2][0]
            typ = typ.astype(bool)
            onset = seq[0][sub - 21][0][ses - 1][0][run - 1][0][0][1][0]
            rewarded_rule = seq[0][sub - 21][0][ses - 1][0][run - 1][0][0][4][0]
        else:
            typ = seq[0][sub - 1][0][ses - 1][0][run - 1][0][0][2][0]
            typ = typ.astype(bool)
            onset = seq[0][sub - 1][0][ses - 1][0][run - 1][0][0][1][0]
            rewarded_rule = seq[0][sub - 1][0][ses - 1][0][run - 1][0][0][4][0]
        choice_onsets = onset[typ]
        IR_stim = onset[~typ]
        table_onsets = df.loc[df.event == 'CHOICE_TRIAL_ONSET'].onset.values
        mean = (table_onsets.astype(float) - choice_onsets[0:len(table_onsets)]).mean()
        std = (table_onsets.astype(float) - choice_onsets[0:len(table_onsets)]).std()
        try:
            assert std < 0.005
        except AssertionError:
            print(sub, ses, run, 'AssertionError', std)
        block_change_onsets = IR_stim + mean
        rewarded_rule_stimuli = rewarded_rule[0:len(typ)][~typ]
        rewarded_rule_choice_trial = rewarded_rule[0:len(typ)][typ]
    stims = pd.DataFrame({'onset': block_change_onsets})
    stims['event'] = 'REWARDED_RULE_STIM'
    df = pd.concat([stims, df], ignore_index=True)
    df['rewarded_rule'] = np.nan
    df.loc[df.event == 'REWARDED_RULE_STIM', 'rewarded_rule'] = rewarded_rule_stimuli[0:len(df.loc[df.event == 'REWARDED_RULE_STIM', 'rewarded_rule'])]
    df.loc[df.event == 'CHOICE_TRIAL_ONSET', 'rewarded_rule'] = rewarded_rule_choice_trial[0:len(df.loc[df.event == 'CHOICE_TRIAL_ONSET', 'rewarded_rule'])]
    return df


for sub, ses, run in product(range(2, 23), [2, 3], [7, 8]):
    print(sub, ses, run)
    try:
        df = repair_tsv(sub, ses, run)
        df.to_csv('/Volumes/flxrl/FLEXRULE/raw/bids_mr_v1.2/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_task-instructed_run-{2}_events.csv'.format(sub, ses, run))
    except FileNotFoundError:
        print('file not found', sub, ses, run)
