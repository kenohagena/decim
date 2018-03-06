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
data = clean.pupil.choicelock.groupby(level=[0]).mean()
data = data.T
data = pd.DataFrame(data.stack()).reset_index(level=['name', 'subject'])
data.columns = ['name', 'subject', 'value']
data.subject = data.subject.astype(int)
data.name = data.name.astype(float)
print(clean.loc[6])
# PLOT

f, ax = plt.subplots(figsize=(9, 6))
sns.set_context('notebook', font_scale=1.5)
sns.tsplot(data=data, time='name', unit='subject', value='value', ci=[0, 100], err_style='unit_traces')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Pupilsize (normalized)')
ax.set_title('Pupilsize choicelocked')
sns.despine()
f.savefig('pupil_choicelock_mean_c.png', dpi=160)
