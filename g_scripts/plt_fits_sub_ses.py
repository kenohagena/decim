import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
import statmisc


'''
Plot means & 95-HDI per subject (color coded) as a function of session for fitted parameters (H, V, gen_var)
'''
colors = sns.cubehelix_palette(8, start=.5, rot=-.75)

f, ax = plt.subplots(1, 3, figsize=(24, 6))
for p, pos in zip(['H', 'V', 'gen_var'], [0, 1, 2]):
    for subject, i in zip([1, 2, 3, 4, 6, 7, 9], range(7)):
        # DATA INPUT
        dfs = []
        summary = []
        for ses in ['A', 'B', 'C']:
            df = pd.read_csv('/Users/kenohagena/Documents/immuno/da/decim/g_behav/gl_in_fits090318/samples_VPIM0{0}{1}'.format(subject, ses))
            df['session'] = ses
            dfs.append(df)
            s = pd.read_csv('/Users/kenohagena/Documents/immuno/da/decim/g_behav/gl_in_fits090318/summary_VPIM0{0}{1}'.format(subject, ses))
            s['session'] = ses
            summary.append(s)
        df = pd.concat(dfs, ignore_index=True)
        means = df.groupby('session')[p].mean().values
        hdi = np.array([np.array(statmisc.hdi(df.loc[df.session == s, p].values)) for s in ['A', 'B', 'C']]).T
        ax[pos].plot([1 + i / 30, 2 + i / 30, 3 + i / 30], means, color=colors[i], lw=3, label=subject)
        ax[pos].vlines(1 + i / 30, ymin=hdi.T[0, 0], ymax=hdi.T[0, 1], color=colors[i])
        ax[pos].vlines(2 + i / 30, ymin=hdi.T[1, 0], ymax=hdi.T[1, 1], color=colors[i])
        ax[pos].vlines(3 + i / 30, ymin=hdi.T[2, 0], ymax=hdi.T[2, 1], color=colors[i])
        ax[pos].set(xticks=[1.1, 2.1, 3.1], xticklabels=['A', 'B', 'C'],
                    title=p, xlabel='Session')
        ax[pos].legend()
ax[0].set(ylim=[0, .1], yticks=[0, .05, .1])
ax[1].set(ylim=[0, 1], yticks=[0, .5, 1])
ax[2].set(ylim=[0, 1], yticks=[0, .5, 1])
sns.despine()
f.savefig('H_noise_means_sub_session.png', dpi=160)
