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
from decim.adjuvant import slurm_submit as slu
from joblib import Memory
if expanduser('~') == '/home/faty014':
    cachedir = expanduser('/work/faty014/joblib_cache')
else:
    cachedir = expanduser('~/joblib_cache')
slu.mkdir_p(cachedir)
memory = Memory(cachedir=cachedir, verbose=0)

'''
This script ca be used independently for....

1) Just doing nuissance regression of CompCor regressors
    a) Load CompCor confounds
    b) Load preprocessed niftis
    c) Run regression and subtract
    d) Save "denoised" niftis

    Example:
    RE = EPI('sub-17', 'ses-2', 'inference_run-4', flex_dir)
    RE.denoise(save=True)

2) Extract brainstem ROI time series from nifti voxel data.
    a) load preprocessed EPIs ("load_epi")
    b) optionally warp brainstem atlases to subject space using the bash script
            'warp_masks_MNI_to_T1w_subject_space.sh' ("warp_atlases")
    c) load warped brainstem ROI masks ("load_mask")
    d) resample to affine & shape of EPI ("resample_mask")
    e) apply masks to EPI and average using the voxel weights ("brainstem")

3) Extract cortical surface ROI time series
    a) load surface data for both hemispheres ("cortical")
    b) average per Glasser surface label ("cortical")

To combine steps 3 and 4, use function "execute"


'''


class EPI(object):

    def __init__(self, subject, session, run, flex_dir):
        self.subject = subject
        self.session = session
        self.run = run
        self.flex_dir = flex_dir
        self.atlases = {
            'AAN_DR': 'aan_dr',
            '4th_ventricle': '4th_ventricle',
            'basal_forebrain_4': 'zaborsky_bf4',
            'basal_forebrain_123': 'zaborsky_bf123',
            'LC_Keren_2std': 'keren_lc_2std',
            'LC_standard': 'keren_lc_1std',
            'CIT168': {2: 'NAc', 6: 'SNc', 10: 'VTA'}
        }

    def load_epi(self):
        '''
        Find and load EPI-files.

        - Argument:
            a) use denoised data? (default yes, if no --> '')
        '''
        if self.input_nifti == 'mni_retroicor':
            file_identifier = 'retroicor'
        elif self.input_nifti == 'T1w':
            file_identifier = 'space-T1w_preproc.'
        file = glob(join(self.flex_dir, 'fmri', 'completed_preprocessed',
                         self.subject, 'fmriprep', self.subject,
                         self.session, 'func',
                         '*{0}*{1}*nii.gz'.
                         format(self.run, file_identifier)))

        print(file, 'loaded')  # keep to avoid that CompCor is applied unnoticed
        if len(file) != 1:
            print('{0} EPIs found for '.format(len(file)), self.subject,
                  self.session, self.run)
        else:
            self.EPI = image.load_img(file)

    def warp_atlases(self):
        '''
        Warp atlases to subject T1w space
        Has to be done only once...
        '''
        atlas_dir = join(self.flex_dir, 'fmri', 'atlases')
        subprocess.call('warp_masks_MNI_to_T1w_subject_space.sh {0} {1}'.
                        format(self.subject, atlas_dir))

    def load_mask(self):
        '''
        Find and load ROI masks.

        Mult_roi_atlases should take the form of a dict of dicts.
        Outerkey: substring to identify atlas, innerkey: frame within that 4D Nifty, value: name of that ROI.
        '''
        self.masks = {}
        if self.input_nifti == 'mni_retroicor':
            mask_dir = join(self.flex_dir, 'fmri', 'atlases', 'selected')
        elif self.input_nifti == 'T1w':
            mask_dir = join(self.flex_dir, 'fmri', 'atlases', self.subject)
        for atlas, rois in self.atlases.items():
            mask = glob(join(mask_dir, '*{}*'.format(atlas)))
            print(mask)
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
            self.resampled_masks[key] = image.resample_img(value,
                                                           epi_img.affine,
                                                           target_shape=epi_img.
                                                           get_data().shape[0:3])

    def mask(self):
        '''
        Apply all masks to all EPIs.
        '''
        self.epi_masked = {}
        self.weights = {}
        for key, resampled_mask in self.resampled_masks.items():
            thresh = image.new_img_like(resampled_mask,
                                        resampled_mask.get_data() > 0.01)
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
            roi = (roi - roi.mean(axis=0)) / roi.std(axis=0)                    # z-score per voxel
            weight = weight / weight.sum()                                      # normalize weights ...
            weighted = np.dot(roi, weight)
            weighted_averages[key] = weighted.flatten()
        self.brainstem_weighted = pd.DataFrame(weighted_averages)

    def cortical(self):  # does not work for retroicor yet
        '''
        Loads surface annotation and voxel data in surface subject space (fsnative)
        - Output: DF w/ labels as columns & timpoints as rows
        '''
        hemispheres = []
        for hemisphere, hemi in zip(['lh', 'rh'], ['L', 'R']):
            annot_path = join(self.flex_dir, 'fmri',
                              'completed_preprocessed', self.subject,
                              'freesurfer', self.subject,
                              'label', '{0}.HCPMMP1.annot'.format(hemisphere))
            hemi_func_path = glob(join(self.flex_dir, 'fmri',
                                       'completed_preprocessed', self.subject,
                                       'fmriprep', self.subject,
                                       self.session, 'func',
                                       '*{0}*fsnative.{1}.func.gii'.
                                       format(self.run, hemi)))[0]
            annot = ni.freesurfer.io.read_annot(annot_path)
            labels = annot[2]
            labels = [i.astype('str') for i in labels]
            affiliation = annot[0]
            surf = surface.load_surf_data(hemi_func_path)
            surf_df = pd.DataFrame(surf).T
            surf_df = (surf_df - surf_df.mean()) / surf_df.std()                # z-score per vertex
            surf_df = surf_df.T
            surf_df['label_index'] = affiliation
            df = surf_df.groupby('label_index').mean().T
            df.columns = labels
            df = df.rename(columns={'???': '{}_???'.format(hemi)})
            hemispheres.append(df)
        cortical_rois = pd.concat(hemispheres, axis=1)
        self.cortical = cortical_rois


#@memory.cache
def execute(subject, session, run, flex_dir,
            atlas_warp=False, input_nifti='mni_retroicor'):
    RE = EPI(subject, session, run, flex_dir)
    RE.input_nifti = input_nifti
    RE.load_epi()
    if atlas_warp is True:
        RE.warp_atlases()                                                       # has to be done only once
    RE.load_mask()
    RE.resample_masks()
    RE.mask()
    RE.brainstem()
    RE.cortical()
    #RE.cortical = pd.DataFrame([0])
    return RE.brainstem_weighted, RE.cortical
