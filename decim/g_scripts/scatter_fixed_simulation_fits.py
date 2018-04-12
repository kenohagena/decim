import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from os.path import join
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font_scale=1, rc={
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 5,
    'axes.linewidth': 0.25,
    'xtick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.pad': 2.0,
    'ytick.minor.pad': 2.0,
    'xtick.major.pad': 2.0,
    'xtick.minor.pad': 2.0,
    'axes.labelpad': 4.0,
})

#Data & Transform

files = glob(join('/Users/kenohagena/Documents/immuno/da/decim/g_behav/H-simu120318/', '*.csv'))

fileframe = []
for file in files:
    df = pd.read_csv(file, index_col=[0])
    df = df.iloc[0:3]
    df['H'] = float(file[file.find('H='):file.find('V=')][2:])
    df['V'] = float(file[file.find('V='):file.find('gv=')][2:])
    df['gen_var'] = float(file[file.find('gv='):file.find('iter')][3:])
    df['condition'] = file[-5]
    fileframe.append(df)

fileframe = pd.concat(fileframe, ignore_index=False)


files = glob(join('/Users/kenohagena/Documents/immuno/da/decim/g_behav/gv_simu120318/', '*.csv'))
gvframe = []
for file in files:
    df = pd.read_csv(file, index_col=[0])
    df = df.iloc[0:3]
    df['H'] = float(file[file.find('H='):file.find('V=')][2:])
    df['V'] = float(file[file.find('V='):file.find('gv=')][2:])
    df['gen_var'] = float(file[file.find('gv='):file.find('iter')][3:])
    df['condition'] = file[-5]
    gvframe.append(df)

gvframe = pd.concat(gvframe, ignore_index=False)

files = glob(join('/Users/kenohagena/Documents/immuno/da/decim/g_behav/V_simu120318/', '*.csv'))
vframe = []
for file in files:
    df = pd.read_csv(file, index_col=[0])
    df = df.iloc[0:3]
    df['H'] = float(file[file.find('H='):file.find('V=')][2:])
    df['V'] = float(file[file.find('V='):file.find('gv=')][2:])
    df['gen_var'] = float(file[file.find('gv='):file.find('iter')][3:])
    df['condition'] = file[-5]
    vframe.append(df)

vframe = pd.concat(vframe, ignore_index=False)


# PLOT


f, ax = plt.subplots(3, 3, figsize=(16, 12))

f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                  wspace=.25, hspace=.25)
ax[0, 0].plot([0, 1], [0, 1], color='black')
ax[0, 0].scatter(fileframe.loc['H']['H'], fileframe.loc['H']['50%'])
ax[0, 1].scatter(fileframe.loc['V']['H'], fileframe.loc['V']['50%'])
ax[0, 1].hlines(1, 0, 1)
ax[0, 2].scatter(fileframe.loc['gen_var']['H'], fileframe.loc['gen_var']['50%'])
ax[0, 2].hlines(1, 0, 1)
ax[0, 0].set(title='H', xlabel='Generating H', ylabel='fitted H')
ax[0, 1].set(title='V', xlabel='Generating H', ylabel='fitted V')
ax[0, 2].set(title='gen_var', xlabel='Generating H', ylabel='fitted gen_var')


ax[1, 0].scatter(vframe.loc['H']['V'], vframe.loc['H']['50%'])
ax[1, 0].hlines(1 / 70, 0, 10)
ax[1, 1].scatter(vframe.loc['V']['V'], vframe.loc['V']['50%'])
ax[1, 1].plot([0.25, 0.5, 0.75, 1, 2, 5, 10], [0.25, 0.5, 0.75, 1, 2, 5, 10], color='black')
ax[1, 2].scatter(vframe.loc['gen_var']['V'], vframe.loc['gen_var']['50%'])
ax[1, 2].hlines(1, 0, 10)
ax[1, 0].set(xlabel='True V', ylabel='fitted H')
ax[1, 1].set(xlabel='True V', ylabel='fitted V')
ax[1, 2].set(xlabel='True H', ylabel='fitted gen_var')

ax[2, 0].scatter(gvframe.loc['H']['gen_var'], gvframe.loc['H']['50%'])
ax[2, 0].hlines(1 / 70, .5, 2)
ax[2, 1].scatter(gvframe.loc['V']['gen_var'], gvframe.loc['V']['50%'])
ax[2, 1].hlines(1, .5, 2)
ax[2, 2].scatter(gvframe.loc['gen_var']['gen_var'], gvframe.loc['gen_var']['50%'])
ax[2, 2].plot(np.linspace(0.5, 2, 15), np.linspace(0.5, 2, 15), color='black')
ax[2, 0].set(xlabel='True gen_var', ylabel='fitted H')
ax[2, 1].set(xlabel='True gen_var', ylabel='fitted V')
ax[2, 2].set(xlabel='True gen_var', ylabel='fitted gen_var')


f.savefig('HVgv_simulation.png', dpi=160)
