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
dftpr = df.loc[~df.pupil.parameter.tpr.isnull()]
bins = dftpr.pupil.parameter.tpr.describe()[3:8].values
names = [1, 2, 3, 4]
dftpr.loc[:, ('pupil', 'parameter', 'tprbin')] = pd.cut(dftpr.pupil.parameter.tpr, bins=bins, labels=names).values
df = pd.concat([dftpr.behavior.parameter.reaction_time,
                dftpr.behavior.parameter.true_evidence,
                dftpr.behavior.parameter.subjective_evidence,
                dftpr.pupil.parameter.tprbin], axis=1).reset_index(drop=True)


# PLOT
f, ax = plt.subplots(1, 3, figsize=(16, 4))
sns.boxplot(y=df.reaction_time, x=df.tprbin, data=df, ax=ax[0], palette=sns.color_palette("GnBu_d"))
sns.boxplot(y=df.subjective_evidence.abs(), x=df.tprbin, data=df, ax=ax[1], palette=sns.color_palette("GnBu_d"))
sns.boxplot(y=df.true_evidence.abs(), x=df.tprbin, data=df, ax=ax[2], palette=sns.color_palette("GnBu_d"))

ax[0].set_xlabel('TPR')
ax[0].set_xticklabels(['1. Quantile', '2. Quantile', '3. Quantile', '4. Quantile'])
ax[0].set_ylabel('RT')
ax[0].set_title('RT x TPR bins')

ax[1].set_xlabel('TPR')
ax[1].set_xticklabels(['1. Quantile', '2. Quantile', '3. Quantile', '4. Quantile'])
ax[1].set_ylabel('Subjective Evidence')
ax[1].set_title('Subjective Evidence x TPR Bins')


ax[2].set_xlabel('TPR')
ax[2].set_xticklabels(['1. Quantile', '2. Quantile', '3. Quantile', '4. Quantile'])
ax[2].set_ylabel('True Evidence')
ax[2].set_title('Real Evidence x TPR Bins')
sns.despine()

f.savefig('tpr_binned_rt_evidence.png', dpi=160)
