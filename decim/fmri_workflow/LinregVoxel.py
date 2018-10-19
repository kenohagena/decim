import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from os.path import join
import nibabel as nib
from sklearn.linear_model import LinearRegression
from nilearn.surface import vol_to_surf
from collections import defaultdict


'''
Use script in two steps:

FIRST: Voxel Regressions & Vol2 Surf
    --> per subject & session

SECOND: Concatenate and average magnitude and lateralization
    --> for all
'''


class VoxelSubject(object):
    def __init__(self, subject, session, runs, flex_dir, BehavAligned, task):
        self.subject = subject
        self.session = session
        self.runs = runs
        self.flex_dir = flex_dir
        self.BehavAligned = BehavAligned
        self.voxel_regressions = {}
        self.surface_textures = defaultdict(dict)
        self.task = task

    def linreg_data(self, denoise=True):
        '''
        Concatenate runwise BOLD- and behavioral timeseries per subject-session.
        Regress each voxel on each behavioral parameter.

        Return one Nifti per session, subject & parameter with four frames:
            coef_, intercept_, r2_score, mean_squared_error

        Z-score voxel (not behavior anymore...)
        '''
        print('load glm data...')
        session_nifti = []
        session_behav = []
        for run in self.runs:
            behav = self.BehavAligned[run]
            if denoise is True:
                nifti = nib.load(join(self.flex_dir, 'fmri', 'completed_preprocessed', self.subject, 'fmriprep', self.subject, self.session, 'func',
                                      '{0}_{1}_task-{2}_bold_space-T1w_preproc_denoise.nii.gz'.format(self.subject, self.session, run)))
            else:
                nifti = nib.load(join(self.flex_dir, 'fmri', 'completed_preprocessed', self.subject, 'fmriprep', self.subject, self.session, 'func',
                                      '{0}_{1}_task-{2}_bold_space-T1w_preproc.nii.gz'.format(self.subject, self.session, run)))

            self.nifti_shape = nifti.get_data().shape
            self.nifti_affine = nifti.affine
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
        session_nifti = (session_nifti - session_nifti.mean()) / session_nifti.std()
        session_nifti = session_nifti.fillna(0)  # because if voxels have std == 0 --> NaNs introduced
        # session_behav = (session_behav - session_behav.mean()) / session_behav.std()
        assert session_behav.shape[0] == session_nifti.shape[0]
        self.session_nifti = session_nifti
        self.session_behav = session_behav

    def single_linreg(self):
        # Z-Score behavior and voxels
        voxels = self.session_nifti
        behav = self.session_behav
        for param in behav.columns:
            print(param)
            linreg = LinearRegression()
            linreg.fit(behav[param].values.reshape(-1, 1),
                       voxels)
            predict = linreg.predict(behav[param].values.reshape(-1, 1))
            reg_result = np.concatenate(([linreg.coef_.flatten()], [linreg.intercept_],
                                         [r2_score(voxels, predict, multioutput='raw_values')],
                                         [mean_squared_error(voxels, predict, multioutput='raw_values')]), axis=0)
            new_shape = np.stack([reg_result[i, :].reshape(self.nifti_shape[0:3]) for i in range(reg_result.shape[0])], -1)
            new_image = nib.Nifti1Image(new_shape, affine=self.nifti_affine)
            self.voxel_regressions[param] = new_image

    def glm(self):
        voxels = self.session_nifti
        behav = self.session_behav
        if self.task == 'instructed':
            behav = behav.loc[:, ['stimulus_vert', 'stimulus_horiz',
                                  'response_left', 'response_right',
                                  'switch', 'abs_switch']]
        elif self.task == 'inference':
            behav = behav.loc[:, ['stimulus_vert', 'stimulus_horiz',
                                  'response_left', 'response_right',
                                  'switch', 'abs_switch',
                                  'belief', 'abs_belief',
                                  'LLR', 'abs_LLR',
                                  'surprise']]
        linreg = LinearRegression()
        print('fit')
        linreg.fit(behav.values, voxels.values)
        predict = linreg.predict(behav.values)
        for i, parameter in enumerate(behav.columns):
            reg_result = np.concatenate(([linreg.coef_[:, i].flatten()], [linreg.intercept_],
                                         [r2_score(voxels, predict, multioutput='raw_values')],
                                         [mean_squared_error(voxels, predict, multioutput='raw_values')]), axis=0)
            new_shape = np.stack([reg_result[i, :].reshape(self.nifti_shape[0:3]) for i in range(reg_result.shape[0])], -1)
            new_image = nib.Nifti1Image(new_shape, affine=self.nifti_affine)
            self.voxel_regressions[parameter + '_glm'] = new_image

    def vol_2surf(self, radius=.3):
        for param, img in self.voxel_regressions.items():
            for hemisphere in ['L', 'R']:
                pial = join(self.flex_dir, 'fmri', 'completed_preprocessed',
                            self.subject,
                            'fmriprep', self.subject, 'anat', '{0}_T1w_pial.{1}.surf.gii'.format(self.subject, hemisphere))
                surface = vol_to_surf(img, pial, radius=radius, kind='line')
                self.surface_textures[param][hemisphere] = pd.DataFrame(surface, columns=['coef_', 'intercept_', 'r2_score', 'mean_squared_error'])


#@memory.cache
def execute(subject, session, runs, flex_dir, BehavAligned, task):
    v = VoxelSubject(subject, session, runs, flex_dir, BehavAligned, task)
    v.linreg_data(denoise=True)
    # v.single_linreg()
    v.glm()
    v.vol_2surf()
    return v.voxel_regressions, v.surface_textures
