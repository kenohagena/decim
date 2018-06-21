import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from os.path import join
import nibabel as nib
from sklearn.linear_model import LinearRegression
import itertools
from decim import slurm_submit as slu
import sys

runs = ['inference_run-4', 'inference_run-5', 'inference_run-6']
epi_dir = '/home/khagena/FLEXRULE/fmri/completed_preprocessed'
behav_dir = '/home/khagena/FLEXRULE/behavior/behav_fmri_aligned'
out_dir = '/home/khagena/FLEXRULE/fmri/voxel_regressions'


def linreg_voxel(sub, session, epi_dir, behav_dir, out_dir):
    '''
    Concatenate runwise BOLD- and behavioral timeseries per subject-session.
    Regress each voxel on each behavioral parameter.

    Return one Nifti per session, subject & parameter with four frames:
        coef_, intercept_, r2_score, mean_squared_error
    '''
    subject = 'sub-{}'.format(sub)
    session_nifti = []
    session_behav = []
    for run in runs:
        nifti = nib.load(join(epi_dir, subject, 'fmriprep', subject, session, 'func',
                              '{0}_{1}_task-{2}_bold_space-T1w_preproc_denoise.nii.gz'.format(subject, session, run)))
        behav = pd.read_csv(join(behav_dir, 'beh_regressors_{0}_{1}_{2}'.format(subject, session, run)),
                            index_col=0)
        shape = nifti.get_data().shape
        data = nifti.get_data()
        d2 = np.stack([data[:, :, :, i].ravel() for i in range(data.shape[-1])])
        if len(d2) > len(behav):
            d2 = d2[0: len(behav)]
        elif len(d2) < len(behav):
            behav = behav.iloc[0:len(d2)]
        session_behav.append(behav)
        session_nifti.append(pd.DataFrame(d2))
    session_nifti = pd.concat(session_nifti, ignore_index=True)
    session_behav = pd.concat(session_behav, ignore_index=True)
    assert session_behav.shape[0] == session_nifti.shape[0]

    for param in behav.columns:
        linreg = LinearRegression()
        linreg.fit(session_behav[param].values.reshape(-1, 1),
                   session_nifti)
        predict = linreg.predict(session_behav[param].values.reshape(-1, 1))
        reg_result = np.concatenate(([linreg.coef_.flatten()], [linreg.intercept_],
                        [r2_score(session_nifti, predict, multioutput='raw_values')],
                       [mean_squared_error(session_nifti, predict, multioutput='raw_values')]), axis=0)
        new_shape = np.stack([reg_result[i, :].reshape(shape[0:3]) for i in range(reg_result.shape[0])], -1)
        new_image = nib.Nifti1Image(new_shape, affine=nifti.affine)
        new_image.to_filename(join(out_dir,
                                  '{0}_{1}_{2}.nii.gz'.format(subject, session, param)))


if __name__ == '__main__':
    slu.mkdir_p(out_dir)
    for ses in ['ses-2', 'ses-3']:
        linreg_voxel(sys.argv[0], ses, epi_dir, behav_dir, out_dir)
