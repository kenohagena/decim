import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.image import resample_img
from os.path import join
from glob import glob
import decim.slurm_submit as slu
from pymeg import parallel as pbs
from multiprocessing import Pool
from matplotlib import pyplot as plt
import seaborn as sns


atlases = [
    'AAN_DR',
    'basal_forebrain_123_Zaborszky',
    'basal_forebrain_4_Zaborszky',
    'CIT168_MNI',
    'LC_Keren_2std',
    'LC_standard_1'
]

'''
FIRST: Extract brainstem Rois and weight them
'''


def extract_brainstem(sub, flex_dir, task):
    cit = pd.read_table(join(flex_dir, 'fmri', 'atlases', 'CIT168_RL_Subcortical_Nuclei/CIT168_Reinf_Learn_v1/labels.txt'), sep='  ', header=None)
    subject = 'sub-{}'.format(sub)
    files = glob(join(flex_dir, 'SubjectLevel2', subject, 'VoxelReg*{}*'.format(task)))
    l_coef_ = []
    for file in files:
        nifti = nib.load(file)
        session = file[file.find('_ses-') + 1:file.find('_ses-') + 6]
        parameter = file[file.find(session) + 5:file.find(task)]
        for a in atlases:
            atlas = nib.load(join(flex_dir, 'fmri', 'atlases', '{0}/{1}_T1w_{0}.nii.gz'.format(subject, a)))
            atlas = resample_img(atlas, nifti.affine, target_shape=nifti.shape[0:3])
            if a == 'CIT168_MNI':
                for i in range(16):
                    atlasdata = atlas.get_data()[:, :, :, i] / atlas.get_data()[:, :, :, i].sum()
                    coef_ = np.multiply(nifti.get_data()[:, :, :, 0], atlasdata[:, :, :]).sum()
                    l_coef_.append({'subject': subject,
                                    'session': session,
                                    'atlas': cit.iloc[i, 1].replace(' ', ''),
                                    'parameter': parameter,
                                    'task': task,
                                    'coef_': coef_})
            else:
                atlasdata = atlas.get_data() / atlas.get_data().sum()
                coef_ = np.multiply(nifti.get_data()[:, :, :, 0], atlasdata[:, :, :, 0]).sum()
                l_coef_.append({'subject': subject,
                                'session': session,
                                'atlas': a,
                                'parameter': parameter,
                                'task': task,
                                'coef_': coef_})
    out_dir = join(flex_dir, 'fmri', 'brainstem_regression')
    slu.mkdir_p(out_dir)
    pd.DataFrame(l_coef_).to_hdf(join(out_dir, 'brainstem_coefs_{}.hdf'.format(task)), key=subject)


def hummel_submit():
    flex_dir = '/work/faty014/FLEXRULE'

    def keys():
        keys = []
        for sub in range(1, 23):
            for task in ['inference', 'instructed']:
                keys.append((sub, flex_dir, task))
        return keys

    def par_execute(keys):
        with Pool(16) as p:
            p.starmap(extract_brainstem, keys)

    slu.pmap(par_execute, keys(), walltime='2:55:00',
             memory=60, nodes=1, tasks=16, name='brainstem_coefs')


def climag_submit():
    flex_dir = '/home/khagena/FLEXRULE'
    for sub in [10]:
        for task in ['inference']:
            pbs.pmap(extract_brainstem, [(sub, flex_dir, task)], walltime='1:00:00',
                     memory=15, nodes=1, tasks=1, name='bs_coefs_{}'.format(sub))


'''
SECOND: Concat per subject & plot
'''

parameters = {'abs_belief': 'Glaze belief (magnitude)',
              'murphy_surprise': 'Surprise',
              'response': 'Response',
              'stimulus': 'Stimulus',
              'switch': 'Switch',
              'abs_LLR': 'LLR (magnitude)'}


def concat_all():
    b = []
    for sub in range(1, 23):
        subject = 'sub-{}'.format(sub)
        for task in ['inference', 'instructed']:
            try:
                brainstem = pd.read_hdf('/Volumes/flxrl/FLEXRULE/fmri/brainstem_regression/brainstem_coefs_{}.hdf'.format(task), key=subject)
                b.append(brainstem)
            except KeyError:
                print(sub, task)
                continue
    brainstem = pd.concat(b, ignore_index=True)
    data = brainstem.groupby(['subject', 'parameter', 'atlas', 'task']).mean().reset_index()
    data = data.loc[data.atlas.isin(['AAN_DR', 'LC_standard_1', 'VTA', 'SNc', 'basal_forebrain_4_Zaborszky', 'basal_forebrain_123_Zaborszky', 'NAC'])]
    return brainstem


def overview_plot(data):

    sns.set(style="ticks")
    d = data.loc[data.parameter.isin(parameters.keys())]

    f, ax = plt.subplots(3, 2, figsize=(18, 16))
    plt.subplots_adjust(wspace=.2, hspace=.3)

    for param, a in zip(parameters.keys(), ax.flatten()):
        d = data.loc[data.parameter == param]

        sns.boxplot(x="coef_", y="atlas", data=d,
                    whis="range", palette="vlag", ax=a)

        sns.swarmplot(x="coef_", y="atlas", data=d,
                      size=4, color=".25", linewidth=0, dodge=True, ax=a)

        a.xaxis.grid(True)
        a.set(xlabel="Regression coefficients", ylabel='Region of interest', yticklabels=['Dorsal raphe', 'Locus coeruleus', 'Ncl. Acc'
                                                                                          'Substantia nigra', 'Ventral tegmental area',
                                                                                          'Basalforebrain 1-3', 'Basal forebrain 4'], xticks=np.arange(-.04, .06, .02),
              title=parameters[param])
    for axlist in ax:
        axlist[1].set(ylabel='', yticks=[])
    sns.despine(trim=True, left=True)
    f.savefig('/Users/kenohagena/Flexrule/fmri/plots/brainstem_regressions/all.png', dpi=160)


def single_plots(data):
    plt.rcParams['pdf.fonttype'] = 3
    plt.rcParams['ps.fonttype'] = 3
    sns.set(style='ticks', font_scale=1, rc={
        'axes.labelsize': 15,
        'axes.titlesize': 17,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 10,
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

    for param, title in parameters.items():
        f, a = plt.subplots(figsize=(9, 5))
        d = data.loc[data.parameter == param]

        sns.boxplot(x="coef_", y="atlas", data=d,
                    whis="range", palette="vlag", ax=a)

        sns.swarmplot(x="coef_", y="atlas", data=d,
                      size=4, color=".25", linewidth=0, dodge=True, ax=a)

        a.xaxis.grid(True)
        a.set(xlabel="Regression coefficients", ylabel='Region of interest', yticklabels=['DR', 'LC', 'NAC',
                                                                                          'SNc', 'VTA',
                                                                                          'BF 1-3', 'BF 4'], xticks=np.arange(-.04, .06, .02),
              title=parameters[param])

        sns.despine(trim=True, left=True)
        f.savefig('/Users/kenohagena/Flexrule/fmri/plots/brainstem_regressions/{}.png'.format(param), dpi=160)


'''
data = concat_all()
print(data.loc[data.subject == 'sub-10'])
'''
