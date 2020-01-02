import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os.path import expanduser, join
from decim import statmisc

sns.set(style='ticks', font_scale=1, rc={
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 10,
    'axes.linewidth': 2.5,
    'xtick.major.width': 2.5,
    'ytick.major.width': 2.5,
    'ytick.major.pad': 2.0,
    'ytick.minor.pad': 2.0,
    'xtick.major.pad': 2.0,
    'xtick.minor.pad': 2.0,
    'axes.labelpad': 4.0,
})

'''
PLot mean across subjects of modes of fit and 2* SEM
'''

# DATA
summary = []
for sub in range(1, 23):
    for ses in range(1, 4):
        try:
            s = pd.read_csv(expanduser('/Volumes/flxrl/FLEXRULE/behavior/bids_stan_fits/sub-{0}_ses-{1}_stanfit.csv'.format(sub, ses)))
            dr = {'vmode': statmisc.mode(s.V.values, 50, decimals=False), 'vupper': statmisc.hdi(s.V.values)[1],
                  'vlower': statmisc.hdi(s.V.values)[0], 'hmode': statmisc.mode(s.H.values, 50, decimals=False),
                  'hupper': statmisc.hdi(s.H.values)[1], 'hlower': statmisc.hdi(s.H.values)[0],
                  'subject': 'sub-{}'.format(sub), 'session': 'ses-{}'.format(ses)}
            summary.append(dr)
        except FileNotFoundError:
            print("No fit for subject {0} session {1}".format(sub, ses))
            pass
summary = pd.DataFrame(summary)
summary.to_csv('/Users/kenohagena/Flexrule/fmri/analyses/bids_stan_fits/summary_stan_fits.csv')


# PLOT
x = np.array([1, 2, 3])
colors = sns.cubehelix_palette(8, start=.5, rot=-.75)
f, ax = plt.subplots(1, 2, figsize=(24, 6))

ax[0].errorbar(x, summary.groupby('session')['hmode'].mean().values,
               yerr=2 * summary.groupby('session')['hmode'].sem().values,
               marker='.',
               elinewidth=4,
               markersize=0,
               color=colors[3])
ax[0].set(xticks=x,
          xticklabels=['A', 'B', 'C'],
          title='Hazard Rate H',
          xlabel='Session',
          ylabel='H',
          ylim=[0, 0.05],
          yticks=[0, .025, .05])


ax[1].errorbar(x, summary.groupby('session')['vmode'].mean().values,
               yerr=2 * summary.groupby('session')['vmode'].sem().values,
               marker='.',
               elinewidth=4,
               markersize=0,
               color=colors[3])
ax[1].set(xticks=x,
          xticklabels=['A', 'B', 'C'],
          title='Internal Noise V',
          xlabel='Session',
          ylabel='V',
          ylim=[0, 3],
          yticks=[0, 1.5, 3])


sns.despine(offset=5)

f.savefig('/Users/kenohagena/Desktop/HV_flexrule_stan_fits.png', dpi=160)
