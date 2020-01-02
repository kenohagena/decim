import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import pandas as pd
from os.path import join
import nibabel as nib
import itertools
from decim import slurm_submit as slu
import sys

epi_dir = '/home/khagena/FLEXRULE/fmri/completed_preprocessed'
fmri_dir = '/home/khagena/FLEXRULE/fmri/CompCor_denoising'
runs = ['inference_run-4', 'inference_run-5', 'inference_run-6']


def denoise(sub, session, run, epi_dir, fmri_dir):
    '''
    Regress voxeldata to CompCor nuisance regressors & Subtract predicted noise.

    INPUT: confound.tsv & nifti file
    OUTPUT: _denoise.nii nifti file in preprocessed directory & pandas.csv.
    '''
    subject = 'sub-{}'.format(sub)
    confounds = pd.read_table(join(epi_dir, subject, 'fmriprep', subject, session, 'func',
                                   '{0}_{1}_task-{2}_bold_confounds.tsv'.format(subject, session, run)))
    confounds = confounds[['tCompCor00', 'tCompCor01', 'tCompCor02',
                           'tCompCor03', 'tCompCor04', 'tCompCor05']]
    nifti = nib.load(join(epi_dir, subject, 'fmriprep', subject, session, 'func',
                          '{0}_{1}_task-{2}_bold_space-T1w_preproc.nii.gz'.format(subject, session, run)))
    shape = nifti.get_data().shape
    data = nifti.get_data()
    d2 = np.stack([data[:, :, :, i].ravel() for i in range(data.shape[-1])])
    linreg = LinearRegression(n_jobs=1, normalize=False)
    linreg.fit(confounds, d2)
    predict = linreg.predict(confounds)
    df = pd.DataFrame(linreg.coef_, columns=confounds.columns)
    df['r2_score'] = r2_score(d2, predict, multioutput='raw_values')
    df['mean_squared_error'] = mean_squared_error(d2, predict, multioutput='raw_values')
    df['intercept'] = linreg.intercept_
    noise = predict - linreg.intercept_
    denoise = d2 - noise
    new_shape = np.stack([denoise[i, :].reshape(shape[0:3]) for i in range(denoise.shape[0])], -1)
    new_image = nib.Nifti1Image(new_shape, affine=nifti.affine)
    new_image.to_filename(join(epi_dir, subject, 'fmriprep', subject, session, 'func',
                               '{0}_{1}_task-{2}_bold_space-T1w_preproc_denoise.nii.gz'.format(subject, session, run)))
    df.to_csv(join(fmri_dir, '{0}_{1}_{2}_denoising.csv'.format(subject, session, run)))


def execute(sub):
    for ses, run in itertools.product(['ses-2', 'ses-3'], runs):
        try:
            denoise(sub, ses, run, epi_dir, fmri_dir)
            print('{0} {1} {2} succesful'.format(sub, ses, run))
        except FileNotFoundError:
            print('{0} {1} {2} file not found'.format(sub, ses, run))


if __name__ == '__main__':
    slu.mkdir_p(fmri_dir)
    execute(sys.argv[1])
