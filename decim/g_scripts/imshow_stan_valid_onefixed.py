import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from os.path import join
import decim.statmisc as ds
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

files = glob(join('/Users/kenohagena/Documents/immuno/da/analyses/g_behav/lisa_die_dritte120418/', '*.csv'))
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

vf = df.loc[df.fixed == 'gv']
vim = vf.loc[:, ['V', 'true_H', 'true_V']]
vimz = vim
vimz.true_V = (vim.true_V - vim.V.mean()) / vim.V.std()
vimz.V = (vim.V - vim.V.mean()) / vim.V.std()
vimz['value'] = vimz.true_V - vimz.V
vimz.value = vimz.value.abs()
vimz = vimz.drop('V', axis=1).groupby(['true_H', 'true_V']).mean().\
            reset_index(level=['true_H', 'true_V'])
vim = vimz.pivot(index='true_H', columns='true_V', values='value')

hvim = vf.loc[:, ['H', 'true_H', 'true_V']]
hvimz = hvim
hvimz.true_H = (hvim.true_H - hvim.H.mean()) / hvim.H.std()
hvimz.H = (hvim.H - hvim.H.mean()) / hvim.H.std()
hvimz['value'] = hvimz.true_H - hvimz.H
hvimz.value = hvimz.value.abs()
hvimz = hvimz.drop('H', axis=1).groupby(['true_H', 'true_V']).mean().\
                reset_index(level=['true_H', 'true_V'])
hvim = hvimz.pivot(index='true_H', columns='true_V', values='value')

gvf = df.loc[df.fixed == 'v']
gvim = gvf.loc[:, ['gen_var', 'true_H', 'true_gen_var']]
gvimz = gvim
gvimz.true_gen_var = (gvim.true_gen_var - gvim.gen_var.mean()) /\
                        gvim.gen_var.std()
gvimz.gen_var = (gvim.gen_var - gvim.gen_var.mean()) / gvim.gen_var.std()
gvimz['value'] = gvimz.true_gen_var - gvimz.gen_var
gvimz.value = gvimz.value.abs()
gvimz = gvimz.drop('gen_var', axis=1).groupby(['true_H', 'true_gen_var']).\
            mean().reset_index(level=['true_H', 'true_gen_var'])
gvim = gvimz.pivot(index='true_H', columns='true_gen_var', values='value')

hgvim = gvf.loc[:, ['H', 'true_H', 'true_gen_var']]
hgvimz = hgvim
hgvimz.true_H = (hgvim.true_H - hgvim.H.mean()) / hgvim.H.std()
hgvimz.H = (hgvim.H - hgvim.H.mean()) / hgvim.H.std()
hgvimz['value'] = hgvimz.true_H - hgvimz.H
hgvimz.value = hgvimz.value.abs()
hgvimz = hgvimz.drop('H', axis=1).groupby(['true_H', 'true_gen_var']).mean().\
            reset_index(level=['true_H', 'true_gen_var'])
hgvim = hgvimz.pivot(index='true_H', columns='true_gen_var', values='value')


f, ax = plt.subplots(2, 2, figsize=(12, 12))
f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                  wspace=0, hspace=.3)
im1 = ax[0, 0].imshow(hvim.sort_index(ascending=False),
                      cmap='GnBu', vmin=0, vmax=2)
im2 = ax[1, 0].imshow(vim.sort_index(ascending=False),
                      cmap='GnBu', vmin=0, vmax=2)
im3 = ax[0, 1].imshow(hgvim.sort_index(ascending=False),
                      cmap='GnBu', aspect=.6, vmin=0, vmax=2)
im4 = ax[1, 1].imshow(gvim.sort_index(ascending=False),
                      cmap='GnBu', aspect=.6, vmin=0, vmax=2)

ax[0, 0].set(yticks=[0, 5, 10], yticklabels=[0.45, .2, .01],
             xticks=[0, 4, 8], xticklabels=[1, 3, 5],
             ylabel='H', title='mean error of fitted H')
ax[1, 0].set(yticks=[0, 5, 10], yticklabels=[0.45, .2, .01],
             xticks=[0, 4, 8], xticklabels=[1, 3, 5],
             ylabel='H', xlabel='V', title='mean error of fitted V')
ax[0, 1].set(yticks=[0, 5, 10], yticklabels=[0.45, .2, .01],
             xticks=[0, 2, 4], xticklabels=[1, 2, 3],
             title='mean error of fitted H')
ax[1, 1].set(yticks=[0, 5, 10], yticklabels=[0.45, .2, .01],
             xticks=[0, 2, 4], xticklabels=[1, 2, 3],
             xlabel='gen-var', title='mean error of fitted gen_var')


axins = inset_axes(ax[1, 1],
                   width="10%",  # width = 10% of parent_bbox width
                   height="230%",  # height : 50%
                   loc=3,
                   bbox_to_anchor=(1.2, 0., 1, 1),
                   bbox_transform=ax[1, 1].transAxes,
                   borderpad=0
                   )

plt.colorbar(im1, cax=axins, ticks=[0, 1, 2])
sns.despine(bottom=False)
f.savefig('imshow_validation_fits.png', dpi=160)
