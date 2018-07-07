import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from os.path import join
import nibabel as nib
from sklearn.linear_model import LinearRegression
from decim import slurm_submit as slu
from nilearn.surface import vol_to_surf
import sys
from glob import glob

runs = ['inference_run-4', 'inference_run-5', 'inference_run-6']
epi_dir = '/home/khagena/FLEXRULE/fmri'
behav_dir = '/home/khagena/FLEXRULE/behavior/behav_fmri_aligned'
out_dir = '/home/khagena/FLEXRULE/fmri/voxel2'
hemis = {'L': 'lh', 'R': 'rh'}


'''
Use script in two steps:

FIRST: Voxel Regressions & Vol2 Surf
    --> per subject & session

SECOND: Concatenate and average magnitude and lateralization
    --> for all
'''


class VoxelSubject(object):
    def __init__(self, sub, session, epi_dir, out_dir, behav_dir):
        self.sub = sub
        self.subject = 'sub-{}'.format(sub)
        self.session = session
        self.epi_dir = epi_dir
        self.out_dir = out_dir
        self.behav_dir = behav_dir
        self.voxel_regressions = {}
        self.surface_textures = []

    def linreg_voxel(self):
        '''
        Concatenate runwise BOLD- and behavioral timeseries per subject-session.
        Regress each voxel on each behavioral parameter.

        Return one Nifti per session, subject & parameter with four frames:
            coef_, intercept_, r2_score, mean_squared_error
        '''
        session_nifti = []
        session_behav = []
        for run in runs:
            nifti = nib.load(join(self.epi_dir, 'completed_preprocessed', self.subject, 'fmriprep', self.subject, self.session, 'func',
                                  '{0}_{1}_task-{2}_bold_space-T1w_preproc_denoise.nii.gz'.format(self.subject, self.session, run)))
            behav = pd.read_csv(join(self.behav_dir, 'beh_regressors_{0}_{1}_{2}'.format(self.subject, self.session, run)),
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
        # Z-Score behavior and voxels
        session_nifti = (session_nifti - session_nifti.mean()) / session_nifti.std()
        session_behav = (session_behav - session_behav.mean()) / session_behav.std()
        assert session_behav.shape[0] == session_nifti.shape[0]
        self.parameters = behav.columns
        for param in self.parameters:
            linreg = LinearRegression()
            linreg.fit(session_behav[param].values.reshape(-1, 1),
                       session_nifti)
            predict = linreg.predict(session_behav[param].values.reshape(-1, 1))
            reg_result = np.concatenate(([linreg.coef_.flatten()], [linreg.intercept_],
                                         [r2_score(session_nifti, predict, multioutput='raw_values')],
                                         [mean_squared_error(session_nifti, predict, multioutput='raw_values')]), axis=0)
            new_shape = np.stack([reg_result[i, :].reshape(shape[0:3]) for i in range(reg_result.shape[0])], -1)
            new_image = nib.Nifti1Image(new_shape, affine=nifti.affine)
            self.voxel_regressions[param] = new_image
            slu.mkdir_p(join(self.out_dir, 'voxel_regressions'))
            new_image.to_filename(join(self.out_dir, 'voxel_regressions',
                                       '{0}_{1}_{2}.nii.gz'.format(self.subject, self.session, param)))

    def vol_2surf(self, radius=.3):
        for param, img in self.voxel_regressions.iteritems():
            for hemisphere in ['L', 'R']:
                pial = join(self.epi_dir, self.subject,
                            'fmriprep', self.subject, 'anat', '{0}_T1w_pial.{1}.surf.gii'.format(self.subject, hemisphere))
                surface = vol_to_surf(img, pial, radius=radius, kind='line')
                self.surface_textures.append(surface)
                slu.mkdir_p(join(self.out_dir, 'surface_textures'))
                pd.DataFrame(surface, columns=['coef_', 'intercept_', 'r2_score', 'mean_squared_error']).\
                    to_csv(join(self.out_dir, 'surface_textures', '{0}_{1}_{2}_{3}.csv'.format(self.subject, self.session, param, hemisphere)))


def lateralize(x):
    x = x.reset_index()
    left = x.query('hemisphere=="L"')
    right = x.query('hemisphere=="R"')
    del left['hemisphere']
    del right['hemisphere']
    left.set_index(['subject', 'parameter', 'names', 'labs'], inplace=True)
    right.set_index(['subject', 'parameter', 'names', 'labs'], inplace=True)
    if all(x.parameter == 'response_left'):
        x = right - left
    elif all(x.parameter == 'response_right'):
        x = left - right
    else:
        raise RuntimeError()
    return x


def concat(input_dir):
    files = glob(input_dir)
    dfs = []
    for file in (files):
        subject = file[file.find('sub-'):file.find('_ses-')]
        parameter = file[file.find('ses-') + 6:file.find('.csv') - 2]
        session = file[file.find('ses-'):file.find(parameter) - 1]
        hemisphere = file[file.find(parameter) + len(parameter) + 1:file.find('.csv')]
        df = pd.read_csv(file, index_col=0)
        df['belief_right'] = df.belief  # if not existent
        aparc_file = '/Volumes/flxrl/FLEXRULE/fmri/completed_preprocessed/{0}/freesurfer/{0}/label/{1}.HCPMMP1.annot'.\
            format(subject, hemis[hemisphere])
        labels, ctab, names = nib.freesurfer.read_annot(aparc_file)
        df['labs'] = labels
        str_names = [str(i) for i in names]
        str_names = [i[2:-1] if i == "b'???'" else i[4:-1] for i in str_names]
        grouped = df.groupby('labs').mean().reset_index()
        grouped['names'] = str_names
        grouped['parameter'] = parameter
        grouped['subject'] = subject
        grouped['session'] = session
        grouped['hemisphere'] = hemisphere
        dfs.append(grouped)
    df = pd.concat(dfs, ignore_index=True)
    return df


def surface_plot_data(grouped_df, lateral_params, marker='coef_'):
    '''
    1. Average across sessions
    2a. Ttest per parameter, roi & hemisphere
    2b. Average across hemispheres
    --> magnitude values of coef_ per ROI
    3a. difference between hemispheres (kontra - ipsi)
    3b. average across conditions (response left & right)
    3c. ttest across subjects.
    --> lateralization values of coef_ for response / rresp
    '''
    df = grouped_df
    ses_mean = df.groupby(['subject', 'parameter', 'names', 'hemisphere']).mean().reset_index()
    mag = ses_mean.groupby(['parameter', 'names', 'hemisphere']).agg(lambda x: ttest(x, 0)[0]).reset_index()
    mag = mag.groupby(['parameter', 'names']).mean().reset_index()
    mag = mag.pivot(columns='parameter', index='names', values=marker)

    for lateral_param in lateral_params:
        lat = ses_mean.loc[ses_mean.parameter.isin(['{}_left'.format(lateral_param), '{}_right'.format(lateral_param)])]
        lat.set_index(['subject', 'parameter', 'names', 'hemisphere', 'labs'], inplace=True)
        lat = lat.groupby(['parameter'], group_keys=False).apply(lateralize).reset_index()
        lat = lat.groupby(['names', 'subject']).mean().reset_index()
        lat = lat.groupby(['names']).agg(lambda x: ttest(x, 0)[0]).reset_index()
        mag['{}_lat'.format(lateral_param)] = lat[marker].values
    return mag


def surface_data(input_dir, lateral_params):
    grouped = concat(input_dir)
    mag_lat = surface_plot_data(grouped, lateral_params)
    mag_lat.to_csv(join(input_dir, 'surface_textures_average.csv'))


if __name__ == '__main__':
    slu.mkdir_p(out_dir)
    for ses in ['ses-2', 'ses-3']:
        slu.mkdir_p(out_dir)
        v = VoxelSubject(sys.argv[1], 'ses-2', epi_dir, out_dir, behav_dir)
        v.linreg_voxel()
        v.vol_2surf()


'''

    slu.mkdir_p(out_dir)
    for session in ['ses-2', 'ses-3']:
        linreg_voxel(sys.argv[1], session, epi_dir, behav_dir, out_dir)


'''
