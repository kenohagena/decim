import pandas as pd
from nilearn import image, masking
from glob import glob
from os.path import join, expanduser
import numpy as np
from nilearn import surface
import nibabel as ni
import subprocess
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import nibabel as nib

from decim import slurm_submit as slu
from joblib import Memory
cachedir = expanduser('~/joblib_cache')
slu.mkdir_p(cachedir)
memory = Memory(cachedir=cachedir, verbose=0)


class EPI(object):

    def __init__(self, subject, session, run, flex_dir):
        self.subject = subject
        self.session = session
        self.run = run
        self.flex_dir = flex_dir
        self.atlases = {
            'AAN_DR': 'aan_dr',
            'basal_forebrain_4': 'zaborsky_bf4',
            'basal_forebrain_123': 'zaborsky_bf123',
            'LC_Keren_2std': 'keren_lc_2std',
            'LC_standard': 'keren_lc_1std',
            'CIT168': {2: 'NAc', 6: 'SNc', 10: 'VTA'}
        }

    def denoise(self):
        '''
        Regress voxeldata to CompCor nuisance regressors & Subtract predicted noise.

        INPUT: confound.tsv & nifti file
        OUTPUT: _denoise.nii nifti file in preprocessed directory & pandas.csv.
        '''
        confounds = pd.read_table(join(self.flex_dir, 'fmri', 'completed_preprocessed', self.subject, 'fmriprep', self.subject, self.session, 'func',
                                       '{0}_{1}_task-{2}_bold_confounds.tsv'.format(self.subject, self.session, self.run)))
        confounds = confounds[['tCompCor00', 'tCompCor01', 'tCompCor02',
                               'tCompCor03', 'tCompCor04', 'tCompCor05']]
        nifti = nib.load(join(self.flex_dir, 'fmri', 'completed_preprocessed', self.subject, 'fmriprep', self.subject, self.session, 'func',
                              '{0}_{1}_task-{2}_bold_space-T1w_preproc.nii.gz'.format(self.subject, self.session, self.run)))
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
        new_image.to_filename(join(self.flex_dir, 'fmri', 'completed_preprocessed', self.subject, 'fmriprep', self.subject, self.session, 'func',
                                   '{0}_{1}_task-{2}_bold_space-T1w_preproc_denoise.nii.gz'.format(self.subject, self.session, self.run)))
        return new_image

    def load_epi(self):
        '''
        Find and load EPI-files.
        '''
        file = glob(join(self.flex_dir, 'fmri', 'completed_preprocessed', self.subject, 'fmriprep', self.subject, self.session, 'func',
                         '*{}*space-T1w_preproc_denoise.nii.gz'.format(self.run)))
        if len(file) == 1:
            self.EPI = image.load_img(file)
        elif len(file) == 0:
            print('CompCor denoising first...')
            self.EPI = self.denoise()
        else:
            print('More than one EPI found for ', self.subject, self.session, self.run)

    def warp_atlases(self):
        atlas_dir = join(self.flex_dir, 'fmri', 'atlases')
        subprocess.call('warp_masks_MNI_to_T1w_subject_space.sh {0} {1}'.format(self.subject, atlas_dir))

    def load_mask(self):
        '''
        Find and load ROI masks.

        Mult_roi_atlases should take the form of a dict of dicts.
        Outerkey: substring to identify atlas, innerkey: frame within that 4D Nifty, value: name of that ROI.
        '''
        self.masks = {}
        mask_dir = join(self.flex_dir, 'fmri', 'atlases', self.subject)
        for atlas, rois in self.atlases.items():
            mask = glob(join(mask_dir, '*{}*'.format(atlas)))
            if atlas == 'CIT168':
                for index, roi in rois.items():
                    self.masks[roi] = image.index_img(mask[0], index)
            else:
                self.masks[rois] = image.load_img(mask)

    def resample_masks(self):
        '''
        Resample masks to affine/shape of EPIs.
        '''
        self.resampled_masks = {}
        epi_img = self.EPI
        for key, value in self.masks.items():
            self.resampled_masks[key] = image.resample_img(value, epi_img.affine,
                                                           target_shape=epi_img.get_data().shape[0:3])

    def mask(self):
        '''
        Apply all masks to all EPIs.
        '''
        self.epi_masked = {}
        self.weights = {}
        for key, resampled_mask in self.resampled_masks.items():
            thresh = image.new_img_like(resampled_mask, resampled_mask.get_data() > 0.01)
            self.epi_masked[key] = masking.apply_mask(self.EPI, thresh)
            self.weights[key] = masking.apply_mask(resampled_mask, thresh)

    def brainstem(self):
        '''
        INPUT: Roi extracts & weight file (outputs of extract_brainstem_roi)
        --> Normalize weights (sum == 0)
        --> Z-score per voxel
        --> roi * weights
        OUTPUT: 1-D weighted timeseries of ROI
        '''
        weighted_averages = {}
        for key in self.weights.keys():
            if key in self.atlases['CIT168'].values():
                weight = self.weights[key]
            else:
                weight = self.weights[key].T
            roi = self.epi_masked[key]
            roi = (roi - roi.mean()) / roi.std()  # z-score per voxel
            weight = weight / weight.sum()  # normalize weights ...
            weighted = np.dot(roi, weight)
            weighted_averages[key] = weighted.flatten()
        self.brainstem_weighted = pd.DataFrame(weighted_averages)

    def cortical(self):
        '''
        INPUT: surface annotaions & functional file per run in subject space (fsnative)
        --> Z-score per vertex and average across vertices per annotation label.
        OUTPUT: DF w/ labels as columns & timpoints as rows
        '''
        hemispheres = []
        for hemisphere, hemi in zip(['lh', 'rh'], ['L', 'R']):
            annot_path = join(self.flex_dir, 'fmri', 'completed_preprocessed', self.subject,
                              'freesurfer', self.subject, 'label', '{0}.HCPMMP1.annot'.format(hemisphere))
            hemi_func_path = glob(join(self.flex_dir, 'fmri', 'completed_preprocessed', self.subject, 'fmriprep', self.subject,
                                       self.session, 'func', '*{0}*fsnative.{1}.func.gii'.format(self.run, hemi)))[0]
            annot = ni.freesurfer.io.read_annot(annot_path)
            labels = annot[2]
            labels = [i.astype('str') for i in labels]
            affiliation = annot[0]
            surf = surface.load_surf_data(hemi_func_path)
            surf_df = pd.DataFrame(surf).T
            # z-score per vertex
            surf_df = (surf_df - surf_df.mean()) / surf_df.std()
            surf_df = surf_df.T
            surf_df['label_index'] = affiliation
            df = surf_df.groupby('label_index').mean().T
            df.columns = labels
            hemispheres.append(df)
        cortical_rois = pd.concat(hemispheres, axis=1)
        self.cortical = cortical_rois


@memory.cache
def execute(subject, session, run, flex_dir, atlas_warp=False):
    RE = EPI(subject, session, run, flex_dir)
    RE.load_epi()
    if atlas_warp is True:
        RE.warp_atlases()
    RE.load_mask()
    RE.resample_masks()
    RE.mask()
    RE.brainstem()
    RE.cortical()
    return RE.brainstem_weighted, RE.cortical
