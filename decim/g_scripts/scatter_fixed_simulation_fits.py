import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from os.path import join
import decim.statmisc as ds
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font_scale=1, rc={
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 5,
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'ytick.major.pad': 2.0,
    'ytick.minor.pad': 2.0,
    'xtick.major.pad': 2.0,
    'xtick.minor.pad': 2.0,
    'axes.labelpad': 4.0,
})
sns.set_palette((sns.color_palette("GnBu_d")))
palette = (sns.color_palette("GnBu_d"))

# Data & Transform
files = glob(join('/Users/kenohagena/Documents/immuno/da/analyses/g_behav/lisa_validation_1fixed120418/', '*.csv'))
fileframe = []
for file in files:
    cond = file[file.find('samples_'):file.find('fix_')][8:]
    df = pd.read_csv(file)
    H = ds.mode(df.H.values, 100)
    if cond == 'gv':
        V = ds.mode(df.V.values, 100)
        gen_var = np.nan
    else:
        V = np.nan
        gen_var = ds.mode(df.gen_var.values, 100)
    true_H = float(file[file.find('H='):file.find('gv=')][2:])
    true_V = float(file[file.find('V='):file.find('H=')][2:])
    true_gv = float(file[file.find('gv='):file.find('.csv')][3:])
    fileframe.append({'H': H, 'V': V, 'gen_var': gen_var,
                      'true_H': true_H, 'true_V': true_V,
                      'true_gen_var': true_gv,
                      'fixed': cond})
df = pd.DataFrame(fileframe)


# PLOT
f, ax = plt.subplots(2, 2, figsize=(9, 9))
f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                  wspace=.25, hspace=.25)
vf = df.loc[df.fixed == 'gv']
ax[0, 0].scatter(vf.loc[vf.true_V == 1, 'true_H'],
                 vf.loc[vf.true_V == 1, 'H'])
ax[0, 1].scatter(vf.loc[vf.true_V == 1, 'true_H'],
                 vf.loc[vf.true_V == 1, 'V'])
ax[1, 0].scatter(vf.loc[vf.true_H == 0.015, 'true_V'],
                 vf.loc[vf.true_H == 0.015, 'H'])
ax[1, 1].scatter(vf.loc[vf.true_H == 0.015, 'true_V'],
                 vf.loc[vf.true_H == 0.015, 'V'])
ax[0, 0].plot([0, .4], [0, .4], color=palette[1], lw=5, alpha=.3)
ax[0, 1].plot([0, .4], [1, 1], color=palette[1], lw=5, alpha=.3)
ax[1, 0].plot([1, 5], [0.015, 0.015], color=palette[1], lw=5, alpha=.3)
ax[1, 1].plot([1, 5], [1, 5], color=palette[1], lw=5, alpha=.3)
ax[0, 0].set(ylim=(-.05, .5), title='fitted H',
             ylabel='H varied', yticks=[0, .2, .4])
ax[0, 1].set(ylim=(.8, 5), title='fitted V', yticks=[1, 3, 5])
ax[1, 0].set(ylim=(-.05, .5), ylabel='V varied',
             yticks=[0, .2, .4], xticks=[1, 3, 5])
ax[1, 1].set(ylim=(.8, 5), yticks=[1, 3, 5], xticks=[1, 3, 5])
sns.despine(trim=True)
f.savefig('fixed_gv_simulation_fits.png', dpi=160)


f, ax = plt.subplots(2, 2, figsize=(9, 9))
f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                  wspace=.25, hspace=.25)
gvf = df.loc[df.fixed == 'v']

ax[0, 0].scatter(gvf.loc[gvf.true_gen_var == 1, 'true_H'],
                 gvf.loc[gvf.true_gen_var == 1, 'H'])
ax[0, 1].scatter(gvf.loc[gvf.true_gen_var == 1, 'true_H'],
                 gvf.loc[gvf.true_gen_var == 1, 'gen_var'])
ax[1, 0].scatter(gvf.loc[gvf.true_H == 0.015, 'true_gen_var'],
                 gvf.loc[gvf.true_H == 0.015, 'H'])
ax[1, 1].scatter(gvf.loc[gvf.true_H == 0.015, 'true_gen_var'],
                 gvf.loc[gvf.true_H == 0.015, 'gen_var'])
ax[0, 0].plot([0, .5], [0, .5], color=palette[1], lw=5, alpha=.3)
ax[0, 1].plot([0, .5], [1, 1], color=palette[1], lw=5, alpha=.3)
ax[1, 0].plot([1, 4], [0.015, 0.015], color=palette[1], lw=5, alpha=.3)
ax[1, 1].plot([1, 4], [1, 4], color=palette[1], lw=5, alpha=.3)
ax[0, 0].set(ylim=(-.05, .5), title='fitted H', ylabel='H varied',
             yticks=[0, .2, .4])
ax[0, 1].set(ylim=(.5, 4), title='fitted gen_var', yticks=[1, 2, 3, 4])
ax[1, 0].set(ylim=(-.05, .5), ylabel='gen_var varied', yticks=[0, .2, .4])
ax[1, 1].set(ylim=(.5, 4), yticks=[1, 2, 3, 4])
sns.despine(trim=True)
f.savefig('fixed_V_simulation_fits.png', dpi=160)
