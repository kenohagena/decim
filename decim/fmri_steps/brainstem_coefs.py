import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.image import resample_img
from os.path import join

vox = '/work/faty014/FLEXRULE/fmri/voxel_denoise_debug2/voxel_regressions'
atlas_dir = '/work/faty014/FLEXRULE/fmri/atlases'
parameters = ['belief', 'murphy_surprise', 'switch', 'point', 'response',
              'response_left', 'response_right', 'stimulus_horiz', 'stimulus_vert',
              'stimulus', 'rresp_left', 'rresp_right', 'abs_belief']
atlases = [
    'AAN_DR',
    'basal_forebrain_123_Zaborszky',
    'basal_forebrain_4_Zaborszky',
    'CIT168_MNI',
    'LC_Keren_2std',
    'LC_standard_1'
]

cit = pd.read_table(join(atlas_dir, 'CIT168_RL_Subcortical_Nuclei/CIT168_Reinf_Learn_v1/labels.txt'), sep='  ', header=None)


def weighted_coef(subject, session, parameter):
    nifti = nib.load(join(vox, '{0}_{1}_{2}.nii.gz'.format(subject, session, parameter)))
    for a in atlases:
        atlas = nib.load(join(atlas_dir, '{0}/{1}_T1w_{0}.nii.gz'.format(subject, a)))
        atlas = resample_img(atlas, nifti.affine, target_shape=nifti.shape[0:3])
        if a == 'CIT168_MNI':
            for i in range(16):
                atlasdata = atlas.get_data()[:, :, :, i] / atlas.get_data()[:, :, :, i].sum()
                coef_ = np.multiply(nifti.get_data()[:, :, :, 0], atlasdata[:, :, :]).sum()
                l_coef_.append({'subject': subject,
                                'session': session,
                                'atlas': cit.iloc[i, 1].replace(' ', ''),
                                'parameter': parameter,
                                'coef_': coef_})
        else:
            atlasdata = atlas.get_data() / atlas.get_data().sum()
            coef_ = np.multiply(nifti.get_data()[:, :, :, 0], atlasdata[:, :, :, 0]).sum()
            l_coef_.append({'subject': subject,
                            'session': session,
                            'atlas': a,
                            'parameter': parameter,
                            'coef_': coef_})


def execute(sub):
    subject = 'sub-{}'.format(sub)
    l_coef_ = []
    for session in ['ses-2', 'ses-3']:
        for parameter in parameters:
            try:
                weighted_coef(subject, session, parameter)
                print(subject, session, parameter, 'done')
            except FileNotFoundError:
                print(subject, session, parameter)
    pd.DataFrame(l_coef_).to_hdf(join(vox, 'brainstem_coefs.hdf'), key=subject)


def keys():
    keys = [s for s in range(1, 23)]
    return keys


def par_execute(keys):
    with Pool(16) as p:
        p.starmap(execute, keys)


def submit():
    slu.pmap(par_execute, keys(), walltime='2:55:00',
             memory=30, nodes=1, tasks=16, name='brainstem_coefs')
