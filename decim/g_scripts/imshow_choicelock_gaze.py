from glob import glob
from os.path import join
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# INPUT
df = pd.read_csv('/Users/kenohagena/Documents/immuno/da/decim/g_pupilframes/choiceframes260218/cpf12_all.csv', header=[0, 1, 2], index_col=[0, 1, 2, 3], dtype=np.float64)

# TRANSFORM INPUT
clean = df.loc[(df.pupil.parameter.blink == 0) & (df.pupil.parameter.all_artifacts < .2)]
rt = clean.pupil.parameter.onset_gaze
t = clean.pupil.choicelock
t = t.set_index(rt)
t = t.sort_index(ascending=False).dropna()

# PLOTf,
f, ax = plt.subplots(figsize=(10, 10))
ax.imshow(t, aspect='auto')
ax.set_yticks([0, len(t)])
ax.set_yticklabels(['high Gazeangle', 'low Gazeangle'])
ax.set_xticks(np.arange(0, 2500, 500))
ax.set_xticklabels([0, .5, 1, .5, 2, .5])
ax.set(title='Choicelocked trials sorted by gaze angle at onset', xlabel='Time (s)', ylabel='Trial')
ax.axvline(1000, color='0.1')
f.savefig('gl_imshow_gazesort_cl.png', dpi=160)
