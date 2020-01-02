import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from decim import glaze_model as gl
from os.path import join
from glob import glob
import seaborn as sns


def data_from_raw():
    '''
    Make dataframe with mean rewards and reaction time per block, session & subject
    '''
    path = '/Volumes/flxrl/FLEXRULE/raw/bids_mr_v1.2/sub-*/ses-*/*/sub-*_ses-*_task-inference_run-*_events.tsv'
    files = glob(join(path))
    run_sum = []
    for file in files:
        sub = file[file.find('sub-'): file.find('/ses-')]
        ses = file[file.find('ses'): file.find('ses') + 5]
        run = file[file.find('run'): file.find('_events')]
        df = pd.read_table(file)

        if (ses == 'ses-1') & ((sub == 'sub-1') | (sub == 'sub-2')):
            gensides = df.iloc[df.loc[df.event ==
                                      'CHOICE_TRIAL_RULE_RESP'].index - 7].value.values.astype(float)
            rule_resp = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP'].\
                value.values.astype(float)
            rews = ((-rule_resp + .5 - gensides) == 0).mean()

        else:
            reward = df.loc[df.event == 'GL_TRIAL_REWARD']
            rews = reward.value.replace('n/a', 0).astype(float).mean()

        rts = df.loc[df.event == 'CHOICE_TRIAL_RT']
        rts = rts.value.replace('n/a', 0).astype(float)
        run_sum.append({'subject': sub,
                        'session': ses,
                        'run': run,
                        'rt': rts.mean(),
                        'rewards': rews})

    return pd.DataFrame(run_sum)
