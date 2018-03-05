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
df = pd.concat([dftpr.behavior.parameter.reward,
                dftpr.behavior.parameter.stimulus,
                dftpr.behavior.parameter.response,
                dftpr.behavior.parameter.rule_response,
                dftpr.pupil.parameter.tpr], axis=1).reset_index(drop=True)

# PLOT
sns.set_style("ticks")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

f, ax = plt.subplots(1, 4, figsize=(16, 3))
sns.violinplot(x='rule_response', y='tpr', data=df, ax=ax[0], palette=sns.color_palette("GnBu_d"))
sns.violinplot(x='reward', y='tpr', data=df, ax=ax[1], palette=sns.color_palette("GnBu_d"))
sns.violinplot(x='stimulus', y='tpr', data=df, ax=ax[2], palette=sns.color_palette("GnBu_d"))
sns.violinplot(x='response', y='tpr', data=df, ax=ax[3], palette=sns.color_palette("GnBu_d"))

ax[0].set_title('Rule Response')
ax[0].set_xticklabels(['Left', 'Right'])
ax[0].set_ylabel('TPR')

ax[1].set_title('Reward')
ax[1].set_xticklabels(['True', 'False'])
ax[1].set_ylabel('TPR')

ax[2].set_title('Stimulus')
ax[2].set_xticklabels(['Vertical', 'Horizontal'])
ax[2].set_ylabel('TPR')

ax[3].set_title('Response')
ax[3].set_xticklabels(['X', 'M'])
ax[3].set_ylabel('TPR')
sns.despine(bottom=True)

f.savefig('tpr_violinplot.png', dpi=160)
