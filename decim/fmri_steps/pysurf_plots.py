import numpy as np
import pandas as pd
import nibabel as nib
from surfer import Brain
import matplotlib.pyplot as plt
from decim import slurm_submit as slu
from os.path import join
import seaborn as sns


def get_data():
    aparc_file = '/Volumes/flxrl/FLEXRULE/fmri/completed_preprocessed/sub-10/freesurfer/fsaverage/label/lh.HCPMMP1.annot'
    labels, ctab, names = nib.freesurfer.read_annot(aparc_file)
    str_names = [str(i) for i in names]
    str_names = [i[2:-1] if i == "b'???'" else i[4:-1] for i in str_names]
    t_data = pd.read_csv('/Volumes/flxrl/FLEXRULE/fmri/surface_textures/surface_textures_average.csv')
    p_data = pd.read_csv('/Volumes/flxrl/FLEXRULE/fmri/surface_textures/surface_textures_average_p.csv')

    def category_sort(x_data):
        x_data.names = x_data.names.astype('category')
        x_data.names.cat.set_categories(str_names, inplace=True)
        x_data = x_data.sort_values(by='names')
        return x_data
    t_data = category_sort(t_data)
    p_data = category_sort(p_data)
    t_data.iloc[0, 1:] = 0
    return t_data, p_data, str_names, labels


def fdr_filter(t_data, p_data, parameter):
    data = t_data[parameter].values
    filte = benjamini_hochberg(p_data[parameter], 0.05).values
    data[filte != True] = 0
    return data


def benjamini_hochberg(pvals, alpha):
    p_values = pd.DataFrame({'p': pvals, 'index': np.arange(0, len(pvals))})
    p_values = p_values.sort_values(by='p')
    p_values['rank_'] = np.arange(0, len(p_values)) + 1
    p_values['q'] = (p_values.rank_ / len(p_values)) * alpha
    p_values['reject'] = p_values.p < p_values.q
    thresh = p_values.loc[p_values.reject == True].rank_.max()
    p_values['reject'] = p_values.rank_ <= thresh
    return p_values.sort_values(by='index').reject


def montage_plot(parameter, fdr_correct=True):
    fsaverage = "fsaverage"
    hemi = "lh"
    surf = "inflated"
    t_data, p_data, str_names, labels = get_data()
    if fdr_correct is True:
        data = fdr_filter(t_data, p_data, parameter)
    else:
        data = t_data[parameter].values
    data = data[labels]
    brain = Brain(fsaverage, hemi, surf,
                  background="white")
    brain.add_data(data, -10, 10, thresh=None, colormap="RdBu_r", alpha=.8)
    montage = brain.save_montage(None, [['lateral', 'parietal'], ['medial', 'frontal']],
                                 border_size=0, colorbar=None)
    fig, a = plt.subplots(figsize=(24, 24))
    im = plt.imshow(montage, cmap='RdBu_r')
    a.set(xticks=[], yticks=[])
    sns.despine(bottom=True, left=True)
    cbar = fig.colorbar(im, ticks=[montage.min(), (montage.min() + 255) / 2, 255], orientation='horizontal',
                        drawedges=False)
    cbar.ax.set_xticklabels(['-10', '0', '10'])
    plt.rcParams['pdf.fonttype'] = 3
    plt.rcParams['ps.fonttype'] = 3
    sns.set(style='ticks', font_scale=1, rc={
        'axes.labelsize': 6,
        'axes.titlesize': 40,
        'xtick.labelsize': 40,
        'ytick.labelsize': 5,
        'legend.fontsize': 250,
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
    a.set_title(parameter)
    return fig


def plot_single_roi(roi):
    fsaverage = "fsaverage"
    hemi = "lh"
    surf = "inflated"
    t, p, str_names, labels = get_data()
    df = pd.DataFrame({'labs': str_names,
                       't': 0})
    df.loc[df.labs == roi, 't'] = -4
    data = df.t.values
    data = data[labels]
    brain = Brain(fsaverage, hemi, surf,
                  background="white", views=['lateral', 'ventral', 'medial', 'frontal'])
    brain.add_data(data, -10, 11, thresh=None, colormap="RdBu_r", alpha=1)
    f = brain.save_montage(None, [['lateral', 'parietal'], ['medial', 'frontal']],
                           border_size=0, colorbar=None)
    fig, a = plt.subplots()
    im = plt.imshow(f, cmap='RdBu_r')
    a.set(xticks=[], yticks=[])
    sns.despine(bottom=True, left=True)
    cbar = fig.colorbar(im, ticks=[f.min(), (f.min() + 255) / 2, 255], orientation='horizontal',
                        drawedges=False)
    cbar.ax.set_xticklabels(['-10', '0', '10'])
    a.set_title(roi)
    return f, data


def roi_names_param(parameter, correlation):
    t_data, p_data, str_names, labels = get_data()
    data = fdr_filter(t_data, p_data, parameter)
    df = pd.DataFrame({'labs': str_names,
                       't': data})
    if correlation[0:3] == 'pos':
        df = df.loc[df.t > 0]
    elif correlation[0:3] == 'neg':
        df = df.loc[df.t < 0]
    return df


for parameter in p_data.columns.drop('names'):
    p_data, t_data = get_data()
    fig = implot(parameter, fdr_correct=False)
    fig.savefig(join('/Users/kenohagena/Flexrule/fmri/plots/surface_t-stats', '{}.png'.format(parameter)), dpi=160)
