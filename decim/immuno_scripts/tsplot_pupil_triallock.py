import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# DATA
df = pd.read_csv('/Users/kenohagena/Documents/immuno/da/decim/g_pupilframes/cpf12_all.csv',
                 header=[0, 1, 2],
                 index_col=[0, 1, 2, 3],
                 dtype=np.float64)

# TRANSFORM DATA
clean = df.loc[(df.pupil.parameter.blink == 0) & (df.pupil.parameter.all_artifacts < .2)]
data = clean.pupil.triallock.groupby(level=[0]).mean()
data = data.T
data = pd.DataFrame(data.stack()).reset_index(level=['name', 'subject'])
data.columns = ['name', 'subject', 'value']
data.subject = data.subject.astype(int)
data.name = data.name.astype(float)

# PLOT

f, ax = plt.subplots(figsize=(9, 6))
sns.set_context('notebook', font_scale=1.5)
sns.tsplot(data=data, time='name', unit='subject', value='value', ci=[0, 100], err_style='unit_traces')
ax.axvline(1000, color='black', alpha=.3)  # grating shown
ax.axvline(3000, color='black', alpha=.3)  # end of choice trial
ax.axvline(3500, color='black', alpha=.3)  # next point shown
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Pupilsize (normalized)')
ax.set_title('Pupilsize grating locked')
sns.despine()
f.savefig('gl_p_triallock_12_c.png', dpi=160)
