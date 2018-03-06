from glob import glob
from os.path import join
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

subjects = [1, 2, 3, 4, 6, 7, 9]

# INPUT DATA
df = pd.read_csv('/Users/kenohagena/Documents/immuno/da/decim/g_pupilframes/cpf12_all.csv',
                 header=[0, 1, 2],
                 index_col=[0, 1, 2, 3],
                 dtype=np.float64)
# TRANSFORM DATA
params = df.pupil.parameter

# PLOT
colors = {1: '#fdb147', 2: '#8fb67b', 3: '#3778bf'}
N = len(subjects)
ind = np.arange(N)
width = 0.25
f, ax = plt.subplots(figsize=(16, 9))
for session in [1, 2, 3]:
    alltrials = params.loc(axis=0)[:, session].groupby(level=[0]).count().trial_id
    artifacts = params.loc[(params.blink > 0) | (params.all_artifacts > .2)].loc(axis=0)[:, session].groupby(level=[0]).count().trial_id.values
    blinks = params.loc[(params.blink > 0)].loc(axis=0)[:, session].groupby(level=[0]).count().trial_id.values
    aritfacts = artifacts - blinks
    p2 = plt.bar(ind + width * (session - 1), alltrials, width, color=colors[session], alpha=.9, label=session)
    p3 = plt.bar(ind + width * (session - 1), artifacts, width, color='white', alpha=.5)
    p1 = plt.bar(ind + width * (session - 1), blinks, width, color='white', alpha=.7)


plt.ylabel('Trials')
plt.xlabel('Subject')
plt.title('Trials contaminated with blinks and minor artifacts')
plt.xticks(ind + width, [1, 2, 3, 4, 6, 7, 9])
plt.legend()
sns.despine()

f.savefig('loss_of_trials.png', dpi=160)
