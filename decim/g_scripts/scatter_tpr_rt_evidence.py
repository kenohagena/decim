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
dftpr = df.loc[~df.pupil.parameter.tpr.isnull()]

# PLOT
f, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].scatter(dftpr.pupil.parameter.tpr.values,
              dftpr.behavior.parameter.reaction_time,
              color=sns.color_palette("GnBu_d")[1])
ax[1].scatter(dftpr.pupil.parameter.tpr.values,
              dftpr.behavior.parameter.subjective_evidence.abs(),
              color=sns.color_palette("GnBu_d")[2])
ax[2].scatter(dftpr.pupil.parameter.tpr.values,
              dftpr.behavior.parameter.true_evidence.abs(),
              color=sns.color_palette("GnBu_d")[3])


ax[0].set_xlabel('TPR')
ax[0].set_ylabel('RT')
ax[0].set_title('RT x TPR')

ax[1].set_xlabel('TPR')
ax[1].set_ylabel('Subjective Evidence')
ax[1].set_title('Subjective Evidence x TPR')


ax[2].set_xlabel('TPR')
ax[2].set_ylabel('True Evidence')
ax[2].set_title('Real Evidence x TPR')

sns.despine(trim=True)
f.savefig('tpr_rt_evidence.png', dpi=160)
