import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# DATA
df = pd.read_csv('pupilframes/choiceframes200218/cpf_all.csv',
                 header=[0, 1, 2],
                 index_col=[0, 1, 2, 3],
                 dtype=np.float64)

# TRANSFORM DATA
data = df.pupil.triallock.groupby(level=[0]).mean()
data = data.T
data = pd.DataFrame(data.stack()).reset_index(level=['name', 'subject'])
data.columns = ['name', 'subject', 'value']
data.subject = data.subject.astype(int)
data.name = data.name.astype(float)

# PLOT

f, ax = plt.subplots(figsize=(9, 6))
sns.set_context('notebook', font_scale=1.5)
sns.tsplot(data=data, time='name', unit='subject', value='value', ci=[0, 100])
#ax.axvline(1000, color='black', alpha=.3) #grating shown
#ax.axvline(3000, color='black', alpha=.3) #end of choice trial
#ax.axvline(3500, color='black', alpha=.3) #next point shown
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Pupilsize (normalized)')
ax.set_title('Pupilsize grating locked')
sns.despine()
f.savefig('pupil_triallock_mean.png', dpi=160)
