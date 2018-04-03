import numpy as np
import pandas as pd
from scipy import signal
import math
import pupil_pp_cluster as ppp
import choice_frame_cluster as cfp


def sessions():
    for sub in [1, 2, 3, 4, 6, 7, 9]:
        for ses in [1, 2, 3]:
            yield(sub, ses)


df = pd.read_csv('sessionframe.csv')

for sub, ses in sessions():
    mode = df.loc[(df.subject == sub) & (df.session == ses) & (df.condition == 'vaccine'), 'mode'].values
    if (sub == 3) & (ses == 2):
        c = cfp.Choiceframe(sub, ses, 'immuno/data/vaccine/', blocks=[1, 2, 3, 4, 5, 6])
    else:
        c = cfp.Choiceframe(sub, ses, 'immuno/data/vaccine/')
    c.choicetrials()
    c.points()
    c.glaze_belief(mode)
    c.choice_pupil()
    c.merge()
    c.choices = c.choices.set_index(['block', 'trial_id'])
    c.choices.to_csv('immuno/choiceframes200218/cpf_{0}{1}.csv'.format(sub, ses), index=True)
