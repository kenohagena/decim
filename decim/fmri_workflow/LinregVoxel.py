import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from glob import glob
from os.path import join
import nibabel as nib
from sklearn.linear_model import LinearRegression
from nilearn.surface import vol_to_surf
from nilearn import datasets
from collections import defaultdict
from patsy import dmatrix
from scipy.interpolate import interp1d


'''
Script to run GLM
1. Build behvavioral design matrix
    a) import behavioral pd.DataFrame
    b) build matrix using patsy formula
    c) convolve with HRf
    d) downsample
2. Concat runs of session
3. Run GLM
4. vol_2_surf
5. Return as nifti and as surface and design matrix
'''


def hrf(t):
    '''
    Compute hemodynamic response function
    '''
    h = t ** 8.6 * np.exp(-t / 0.547)
    h = np.concatenate((h * 0, h))
    return h / h.sum()


def make_bold(evidence, dt=0.25):
    '''
    Convolve with hemodynamic response function.
    '''
    t = np.arange(0, 20, dt)
    return np.convolve(evidence, hrf(t), 'same')


def interp(x, y, target):
    '''
    Interpolate
    '''
    f = interp1d(x.values.astype(int), y)
    target = target[target.values.astype(int) > min(x.values.astype(int))]
    return pd.DataFrame({y.name: f(target.values.astype(int))}, index=target)


def regular(df, target='16ms'):
    '''
    Set datetime index and resample to target frequency.
    '''
    dt = pd.to_timedelta(df.index.values, unit='ms')
    df = df.set_index(dt)
    target = df.resample(target).mean().index
    return pd.concat([interp(dt, df[c], target) for c in df.columns], axis=1)


#@memory.cache
class VoxelSubject(object):
    '''
    Initialize.

    - Arguments:
        a) subject
        b) session
        c) runs (multiple?)
        d) Flexrule directory
        e) BehavDataFrame
        f) task
    '''

    def __init__(self, subject, session, runs, flex_dir, BehavDataframe, task):
        self.subject = subject
        self.session = session
        self.runs = runs
        self.flex_dir = flex_dir
        self.BehavDataframe = BehavDataframe
        self.voxel_regressions = {}
        self.surface_textures = defaultdict(dict)
        self.task = task

    def design_matrix(self, behav, bfill_rule_resp=False, ffill_stimulus=False,
                      export_desmat_bf_conv=False):
        '''
        Make design matrix per block using Patsy
        Dummy code categorical variables.

        - Arguments:
            a) behavioral pd.DataFrame
            b) rule reponse as boxcar stimulus - response?
            c) stimulus boxcar onset - offset?
            d) export designmatrix before convolution with hrf?
        '''
        print('load glm data...')
        continuous = behav.loc[:, ['belief', 'LLR', 'surprise', 'onset']]       # Split into categorical and numerical regressors
        categorical = behav.loc[:, ['response', 'stimulus', 'switch',
                                    'rule_resp', 'event']]
        if bfill_rule_resp is True:
            categorical.rule_resp = categorical.rule_resp.fillna(method='bfill',
                                                                 limit=1)       # bfill rule_resp at onset of stimulus
            try:                                                                # Then, sanity check I
                assert all(categorical.loc[categorical.event ==
                                           'CHOICE_TRIAL_ONSET'].rule_resp.values ==
                           categorical.loc[categorical.event ==
                                           'CHOICE_TRIAL_RESP'].rule_resp.values)
            except AssertionError:
                print('''Assertion Error:
                    Error in bfilling rule response values from response
                    to stimulus''')
        categorical.rule_resp = categorical.rule_resp.fillna(0.)
        combined = pd.concat([categorical, continuous], axis=1)
        combined = combined.set_index((combined.onset.values * 1000).
                                      astype(int)).drop('onset', axis=1)
        combined = combined.\
            reindex(pd.Index(np.arange(0, combined.index[-1] + 15000, 1)))
        combined = combined.fillna(method='ffill', limit=99)
        if ffill_stimulus is True:
            combined.stimulus = combined.stimulus.fillna(method='ffill',
                                                         limit=1900)            # stimulus lasts until offset --> boxcar regressor
        combined = combined.loc[np.arange(combined.index[0],
                                          combined.index[-1], 100)]
        combined.loc[0] = 0
        combined.loc[:, ['stimulus', 'response', 'switch',
                         'rule_resp', 'surprise', 'LLR']] =\
            combined.loc[:, ['stimulus', 'response', 'switch',
                             'rule_resp', 'surprise', 'LLR']].fillna(0)
        combined.belief = combined.belief.fillna(method='ffill')

        combined.stimulus = combined.stimulus.\
            map({-1: 'vertical', 1: 'horizontal', 0: 'none'})
        combined.response = combined.response.\
            map({-1: 'left', 1: 'right', 0: 'none'})
        combined.rule_resp = combined.rule_resp.\
            map({-1: 'A', 1: 'B', 0: 'none'})
        combined.loc[:, 'response_'] = combined.response + combined.rule_resp
        combined = combined.replace({'response_': {'nonenone': 'none'}})
        print(combined.loc[combined.response_ != 'none'])
        s = ['none', 'vertical', 'horizontal']                                  # levels for patsy formula formulator
        b = ['none', 'left', 'right']
        r = ['none', 'A', 'B']
        t = ['none', 'leftA', 'leftB', 'rightA', 'rightB']
        if self.task == 'instructed':
            design_matrix = dmatrix('''switch + np.abs(switch) +
                            C(stimulus, levels=s) + C(response, levels=b)''',
                                    data=combined)
        elif self.task == 'inference':
            design_matrix = dmatrix('''belief + np.abs(belief) + switch +
                np.abs(switch) + LLR + np.abs(LLR)+ surprise +
                C(response_, levels=t)''', data=combined)
        dm = pd.DataFrame(design_matrix, columns=design_matrix.
                          design_info.column_names, index=combined.index)
        if export_desmat_bf_conv is True:
            dm.to_hdf('/Users/kenohagena/Desktop/design_matrix.hdf',
                      key=self.task)                                            # for checking the design matrix before convolution

        for column in dm.columns:
            print('Align ', column)
            dm[column] = make_bold(dm[column].values, dt=.1)                    # convolve with hrf
        dm = regular(dm, target='1900ms')                                       # resample to EPI frequency
        dm.loc[pd.Timedelta(0)] = 0
        dm = dm.sort_index()
        return dm.drop('Intercept', axis=1)

    def concat_runs(self):
        '''
        Concatenate design matrices per session.

        - Argument:
            a) use denoised nifti?
        '''
        session_nifti = []
        session_behav = []
        for run in self.runs:
            behav = self.design_matrix(self.BehavDataframe[run])
            if self.input_nifti == 'mni_retroicor':
                file_identifier = 'retroicor'
            elif self.input_nifti == 'T1w':
                file_identifier = 'space-T1w_preproc.'
            files = glob(join(self.flex_dir, 'fmri', 'completed_preprocessed',
                              self.subject, 'fmriprep', self.subject,
                              self.session, 'func',
                              '{0}_{1}_task-{2}_*{3}*nii.gz'.
                              format(self.subject, self.session, run,
                                     file_identifier)))
            if len(files) == 1:
                nifti = nib.load(files[0])
            else:
                print('{1} niftis found for {0}, {2}, {3}'.format(self.subject,
                                                                  len(files),
                                                                  self.session,
                                                                  run))

            self.nifti_shape = nifti.get_data().shape
            self.nifti_affine = nifti.affine
            data = nifti.get_data()
            d2 = np.stack([data[:, :, :, i].ravel() for i in range(data.
                                                                   shape[-1])])
            if len(d2) > len(behav):
                d2 = d2[0: len(behav)]
            elif len(d2) < len(behav):
                behav = behav.iloc[0:len(d2)]
            session_behav.append(behav)
            session_nifti.append(pd.DataFrame(d2))
        session_nifti = pd.concat(session_nifti, ignore_index=True)
        session_behav = pd.concat(session_behav, ignore_index=True)
        print(session_behav.std())
        assert session_behav.shape[0] == session_nifti.shape[0]
        self.session_nifti = session_nifti
        self.session_behav = session_behav

    def glm(self, z_score_behav=True):
        '''
        Run GLM on design matrices.

        - Arguments:
            a) z_score behavior?
        '''
        voxels = self.session_nifti
        behav = self.session_behav
        voxels = (voxels - voxels.mean()) / voxels.std()                        # normalize voxels
        voxels = voxels.fillna(0)                                               # because if voxels have std == 0 --> NaNs introduced
        if z_score_behav is True:                                               # normalize behavior
            behav = (behav - behav.mean()) / behav.std()
        self.DesignMatrix = behav
        linreg = LinearRegression()
        print('fit', self.task)
        linreg.fit(behav.values, voxels.values)
        predict = linreg.predict(behav.values)
        for i, parameter in enumerate(behav.columns):
            reg_result = np.concatenate(([linreg.coef_[:, i].flatten()],
                                         [linreg.intercept_],
                                         [r2_score(voxels, predict,
                                                   multioutput='raw_values')],
                                         [mean_squared_error(voxels,
                                                             predict,
                                                             multioutput='raw_values')]),
                                        axis=0)
            new_shape = np.stack([reg_result[i, :].
                                  reshape(self.nifti_shape[0:3])
                                  for i in range(reg_result.shape[0])], -1)
            new_image = nib.Nifti1Image(new_shape, affine=self.nifti_affine)    # make 4D nifti with regression result per parameter
            self.voxel_regressions[parameter] = new_image                       # fourth dimension contains coefficient, r_square, intercept, mean squared error

    def vol_2surf(self):
        '''
        Extract surface data (subject surface) from subject-specific T1w-nifti
        Uses nilearn.vol_to_surf: https://nilearn.github.io/modules/generated/nilearn.surface.vol_to_surf.html
        '''
        for param, img in self.voxel_regressions.items():
            for hemisphere in ['L', 'R']:
                pial = join(self.flex_dir, 'fmri', 'completed_preprocessed',
                            self.subject,
                            'fmriprep', self.subject, 'anat',
                            '{0}_T1w_pial.{1}.surf.gii'.
                            format(self.subject, hemisphere))
                surface = vol_to_surf(img, pial, radius=.3, kind='line')
                self.surface_textures[param][hemisphere] =\
                    pd.DataFrame(surface, columns=['coef_', 'intercept_',
                                                   'r2_score',
                                                   'mean_squared_error'])

    def mni_to_fsaverage(self):
        '''
        Extract surface data (fsaverage) from MNI152-nifti
        Uses nilearn.vol_to_surf: https://nilearn.github.io/modules/generated/nilearn.surface.vol_to_surf.html
        '''
        fs_average = datasets.fetch_surf_fsaverage(mesh='fsaverage')
        for param, img in self.voxel_regressions.items():
            for hemisphere, hemi in {'L': 'left', 'R': 'right'}.items():
                surface = vol_to_surf(img, fs_average['pial_{}'.format(hemi)], radius=.3, kind='line')
                self.surface_textures[param][hemisphere] =\
                    pd.DataFrame(surface, columns=['coef_', 'intercept_',
                                                   'r2_score',
                                                   'mean_squared_error'])


#@memory.cache
def execute(subject, session, runs, flex_dir, BehavDataframe, task):
    v = VoxelSubject(subject, session, runs, flex_dir, BehavDataframe, task)
    v.input_nifti = 'T1w'                                                    # set input-identifier variable ('T1w' or 'mni_retroicor')
    v.concat_runs()
    v.glm()
    v.vol_2surf()                                                            # use when working with T1w-subject space niftis (fmriprep pipeline)
    #v.mni_to_fsaverage()                                                    # use when working with MNI-space niftis (Rudys retroicor pipeline)
    return v.voxel_regressions, v.surface_textures, v.DesignMatrix


'''
from decim.fmri_workflow import BehavDataframe as bd
behav = bd.execute('sub-3', 'ses-2', 'inference_run-4', 'inference', '/Volumes/flxrl/FLEXRULE', pd.read_csv('/Users/kenohagena/Flexrule/fmri/analyses/bids_stan_fits/summary_stan_fits.csv'))
s = VoxelSubject('sub-3', 'ses-2', ['inference_run-4'], '/Volumes/flxrl/FLEXRULE', behav, 'inference')
print(s.design_matrix(behav).columns)
'''
