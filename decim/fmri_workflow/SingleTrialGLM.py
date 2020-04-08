import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from glob import glob
from os.path import join, expanduser
from decim.adjuvant import slurm_submit as slu
import nibabel as nib
from sklearn.linear_model import LinearRegression
from nilearn.surface import vol_to_surf
from nilearn import datasets
from collections import defaultdict
from nilearn.image import resample_img
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import Memory
if expanduser('~') == '/home/faty014':
    cachedir = expanduser('/work/faty014/joblib_cache')
else:
    cachedir = expanduser('~/joblib_cache')
slu.mkdir_p(cachedir)
memory = Memory(location=cachedir, verbose=0)


@memory.cache
def hrf(t):
    '''
    Compute hemodynamic response function
    '''
    h = t ** 8.6 * np.exp(-t / 0.547)
    h = np.concatenate((h * 0, h))
    return h / h.sum()


@memory.cache
def make_bold(evidence, dt=0.25):
    '''
    Convolve with hemodynamic response function.
    '''
    t = np.arange(0, 20, dt)
    return np.convolve(evidence, hrf(t), 'same')


@memory.cache
def regular(df, target='16ms'):
    '''
    Set datetime index and resample to target frequency.
    Use mean() as resampling method
    '''
    dt = pd.to_timedelta(df.index.values, unit='ms')
    df = df.set_index(dt)
    return df.resample(target).mean()


class SingleTrialGLM(object):
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

    def __init__(self, subject, session, runs, flex_dir, BehavDataframe, task, out_dir):
        self.subject = subject
        self.session = session
        self.runs = runs
        self.flex_dir = flex_dir
        self.out_dir = out_dir
        self.BehavDataframe = BehavDataframe
        self.voxel_regressions = {}
        self.surface_textures = defaultdict(dict)
        self.task = task
        self.info = [self.subject, self.session, self.task]

    def design_matrix(self, run, f=1000):
        '''
        Make design matrix per block using Patsy
        Dummy code categorical variables.

        - Arguments:
            a) behavioral pd.DataFrame
        '''
        print('load glm data...')
        behav = self.BehavDataframe[run]
        rule_switches = behav.loc[behav.event == 'REWARDED_RULE_STIM'].reset_index()
        trials = {}
        for i in range(rule_switches.shape[0]):
            start = rule_switches.iloc[i].onset
            if i < rule_switches.shape[0] - 1:
                end = rule_switches.iloc[i + 1].onset
            else:
                end = behav.iloc[-1].onset
            trials[i] = [start, end]
        behav = behav.set_index((behav.onset.values * f).astype(int)).drop('onset', axis=1)
        behav = behav.reindex(pd.Index(np.arange(0, behav.index[-1] + 15000, 1)))
        behav.loc[0] = 0
        for i, start_end in trials.items():
            behav['{0}_trial_{1}'.format(run[-1], i)] = 0
            behav.loc[int(start_end[0] * f): int(start_end[1] * f), '{0}_trial_{1}'.format(run[-1], i)] = 1
            behav['{0}_trial_{1}'.format(run[-1], i)] = make_bold(behav['{0}_trial_{1}'.format(run[-1], i)].values, dt=1 / f)
        trial_bolds = behav.loc[:, ['trial_{}'.format(i) for i in range(len(trials))]]
        dm = regular(trial_bolds, target='1900ms')
        dm.loc[pd.Timedelta(0)] = 0
        return dm.sort_index()

    def concat_runs(self):
        '''
        Concatenate design matrices per session.
        '''
        session_nifti = []
        session_behav = []
        for run in self.runs:
            behav = self.design_matrix(run)

            if self.input_nifti == 'mni_retroicor':
                file_identifier = 'retroicor'
            elif self.input_nifti == 'T1w':
                file_identifier = 'space-T1w_preproc.'
            elif self.input_nifti == 'mni':
                file_identifier = 'space-MNI152NLin2009cAsym_preproc'

            files = glob(join(self.flex_dir, 'fmri', 'completed_preprocessed',
                              self.subject, 'fmriprep', self.subject,
                              self.session, 'func',
                              '{0}_{1}_task-{2}_*{3}*nii.gz'.
                              format(self.subject, self.session, run,
                                     file_identifier)))
            if len(files) == 1:
                nifti = nib.load(files[0])
                if self.input_nifti == 'mni':
                    brain_mask = glob(join(self.flex_dir, 'fmri', 'completed_preprocessed',
                                           self.subject, 'fmriprep', self.subject,
                                           self.session, 'func',
                                           '{0}_{1}_task-{2}_*MNI152NLin2009cAsym_brainmask*nii.gz'.
                                           format(self.subject, self.session, run)))
                    brain_mask = nib.load(brain_mask[0])
                    brain_mask = resample_img(brain_mask, nifti.affine,
                                              target_shape=nifti.shape[:3])
                    nifti = nib.Nifti1Image(np.multiply(nifti.get_fdata().T, brain_mask.get_fdata().T).T, nifti.affine)

            else:
                print(len(files), 'niftis found for', self.info)

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
        session_behav = pd.concat(session_behav, sort=False).fillna(0)
        assert session_behav.shape[0] == session_nifti.shape[0]
        self.session_nifti = session_nifti
        self.session_behav = session_behav

    def sanity_check_trial_regressors(self):
        df = self.session_behav
        f, ax = plt.subplots(len(df.columns), 1, figsize=(10, len(df.columns * .9)))
        for i, col in enumerate(df.columns):
            ax[i].plot(df[col].values)
            ax[i].set(xticks=[], yticks=[])
        sns.despine(bottom=True, left=True)
        slu.mkdir_p(join(self.out_dir, 'sanity_checks'))
        f.savefig(join(self.out_dir, 'sanity_checks',
                       'trial_regressors_{0}_{1}.png'.format(self.subject, self.session)))

    def single_glm(self, trial):

        voxels = self.session_nifti
        behav = self.session_behav
        behav['residual_trials'] = behav.drop(trial, axis=1).sum(axis=1).values
        behav = behav.loc[:, ['residual_trials', trial]]
        voxels = (voxels - voxels.mean()) / voxels.std()                        # normalize voxels
        voxels = voxels.fillna(0)                                               # because if voxels have std == 0 --> NaNs introduced
        behav = (behav - behav.mean()) / behav.std()
        behav = behav.fillna(0)                                                 # missed-reponse regressor can have std=0 --> NaNs introduced
        linreg = LinearRegression()
        print('fit', self.task)
        linreg.fit(behav.values, voxels.values)
        ind = np.where(behav.columns == trial)[0]
        reg_result = [linreg.coef_[:, ind].flatten()].reshape(self.nifti_shape[0:3])
        new_image = nib.Nifti1Image(reg_result, affine=self.nifti_affine)
        self.voxel_regressions[trial] = new_image

    def run_GLMs(self):
        trials = self.session_behav.columns
        for trial in trials:
            self.single_glm(trial)


def execute(subject, session, runs, flex_dir, BehavDataframe, task, out_dir):
    v = SingleTrialGLM(subject, session, runs, flex_dir, BehavDataframe, task, out_dir)
    v.input_nifti = 'T1w'                                                    # set input-identifier variable ('T1w', 'mni_retroicor', 'mni')
    v.concat_runs()
    v.run_GLMs()
    print(v.voxel_regressions)


runs = ['instructed_run-7', 'instructed_run-8']
behav = {run: pd.read_hdf('/home/khagena/FLEXRULE/Workflow/Sublevel_GLM_Climag_2020-01-07/sub-3/BehavFrame_sub-3_ses-2.hdf', key=run) for run in runs}

s = SingleTrialGLM('sub-3', 'ses-2', runs, '/home/khagena/FLEXRULE', behav, 'instructed', '/home/khagena/FLEXRULE')
s.input_nifti = 'T1w'
s.concat_runs()
s.glm()
s.residuals()
