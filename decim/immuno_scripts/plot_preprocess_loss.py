from glob import glob
from os.path import join
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import product

subjects = {'VPIM01': 1, 'VPIM02': 2, 'VPIM03': 3, 'VPIM04': 4, 'VPIM06': 6, 'VPIM07': 7, 'VPIM09': 9}
sessions = {'A': 1, 'B': 2, 'C': 3}
blocks = [1, 2, 3, 4, 5, 6, 7]

# INPUT FILES: 'session-pupilframes'
pa = '/Users/kenohagena/Documents/immuno/da/decim/pupilframes/session_pupilframes_230218/'

# TRANSFORM INPUT
dfs = []
for subject, session in product(subjects, sessions):
    file = glob(join(pa, '*{0}{1}.csv'.format(subjects[subject], sessions[session])))
    df = pd.read_csv(file[0])
    df = df.loc[:, ['blink', 'all_artifacts']]
    df = df.mean()
    df['subject'] = subject
    df['session'] = session
    dfs.append(df)
df = pd.concat(dfs, axis=1).T


# PLOT
colors = {'A': '#fdb147', 'B': '#8fb67b', 'C': '#3778bf'}
pos = {'A': 0, 'B': 1, 'C': 2}
N = len(subjects)
ind = np.arange(N)
width = 0.25
f, ax = plt.subplots(figsize=(16, 9))
for session in sessions:
    blinks = df.loc[df.session == session].blink.values
    artifacts = df.loc[df.session == session].all_artifacts.values
    artifacts = artifacts - blinks
    p1 = plt.bar(ind + width * pos[session], blinks, width, color=colors[session], label=session, alpha=.9)
    p2 = plt.bar(ind + width * pos[session], artifacts, width,
                 bottom=blinks, color=colors[session], alpha=.4)


plt.ylabel('Fraction of Data')
plt.xlabel('Subject')
plt.title('Pupil data polluted by blinks and minor artifacts')
plt.xticks(ind + width, subjects)
#plt.yticks(np.arange(0, 81, 10))
plt.legend()
sns.despine()

f.savefig('artifacts_preprocessing.png', dpi=160)
