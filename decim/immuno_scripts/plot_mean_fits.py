import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

'''
PLot mean across subjects of modes of fit and 2* SEM
'''

# DATA
summary = []
for sub in [1, 2, 3, 4, 6, 7, 9]:
     for ses in ['A', 'B', 'C']:
          s = pd.read_csv('/Users/kenohagena/Documents/immuno/da/decim/g_behav/gl_in_fits090318/summary_VPIM0{0}{1}'.format(sub, ses))
          s['subject'] = sub
          s['session'] = ses
          H = s.iloc[0:1]
          H['parameter'] = 'H'
          V = s.iloc[1:2]
          V['parameter'] = 'V'
          gen_var = s.iloc[2:3]
          gen_var['parameter'] = 'gen_var'
          s = pd.concat([H, V, gen_var])
          summary.append(s)
summary = pd.concat(summary, ignore_index=True)


# PLOT
x = np.array([1, 2, 3])
colors = sns.cubehelix_palette(8, start=.5, rot=-.75)
f, ax = plt.subplots(1, 3, figsize=(24, 6))

ax[0].errorbar(x, summary.loc[summary.parameter == 'H'].groupby('session')['50%'].mean().values,
               yerr=2 * summary.loc[summary.parameter == 'H'].groupby('session')['50%'].sem().values,
               marker='.',
               elinewidth=2,
               markersize=12,
               color=colors[3])
ax[0].set(xticks=x,
          xticklabels=['A', 'B', 'C'],
          title='Hazard Rate',
          xlabel='Session',
          ylabel='H',
          ylim=[0, 0.05],
          yticks=[0, .025, .05])


ax[1].errorbar(x, summary.loc[summary.parameter == 'V'].groupby('session')['50%'].mean().values,
               yerr=2 * summary.loc[summary.parameter == 'V'].groupby('session')['50%'].sem().values,
               marker='.',
               elinewidth=2,
               markersize=12,
               color=colors[3])
ax[1].set(xticks=x,
          xticklabels=['A', 'B', 'C'],
          title='Internal Noise V',
          xlabel='Session',
          ylabel='V',
          ylim=[0, 1],
          yticks=[0, .5, 1])

ax[2].errorbar(x, summary.loc[summary.parameter == 'gen_var'].groupby('session')['50%'].mean().values,
               yerr=2 * summary.loc[summary.parameter == 'gen_var'].groupby('session')['50%'].sem().values,
               marker='.',
               elinewidth=2,
               markersize=12,
               color=colors[3])
ax[2].set(xticks=x,
          xticklabels=['A', 'B', 'C'],
          title='Estimate of Generative Variance',
          xlabel='Session',
          ylabel='gen_var',
          ylim=[0, 1],
          yticks=[0, .5, 1])

sns.despine(offset=0)

f.savefig('H_int_noise_vaccine.png', dpi=160)
