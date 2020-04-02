import pandas as pd
import numpy as np
from os.path import join, expanduser
from decim.adjuvant import slurm_submit as slu
from glob import glob
import nibabel as ni
from nilearn import surface
from scipy.interpolate import interp1d
from collections import defaultdict
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from joblib import Memory
if expanduser('~') == '/home/faty014':
    cachedir = expanduser('/work/faty014/joblib_cache')
else:
    cachedir = expanduser('~/joblib_cache')
slu.mkdir_p(cachedir)
memory = Memory(location=cachedir, verbose=0)
from multiprocessing import Pool
from pymeg import parallel as pbs


hemispheres = ['lh', 'rh']
hemis = ['L', 'R']


@memory.cache
def interp(x, y, target):
    '''
    Interpolate
    '''
    f = interp1d(x.values.astype(int), y)
    target = target[target.values.astype(int) > min(x.values.astype(int))]
    return pd.DataFrame({y.name: f(target.values.astype(int))}, index=target)


class DecodeSurface(object):

    def __init__(self, subject, session,
                 flex_dir='/home/khagena/FLEXRULE', BehavFrame=None,
                 runs=['inference_run-4', 'inference_run-5', 'inference_run-6'],):

        self.subject = subject
        self.session = session
        self.runs = runs
        self.flex_dir = flex_dir
        self.BehavFrame = BehavFrame
        self.whole_cortex = defaultdict()
        self.annotation = defaultdict()
        self.labels = defaultdict()
        self.mean_auc = defaultdict()
        annot_path = join(self.flex_dir, 'fmri',
                          'completed_preprocessed', self.subject,
                          'freesurfer', self.subject,
                          'label', 'lh.HCPMMP1.annot')
        annot = ni.freesurfer.io.read_annot(annot_path)
        labels = [i.astype('str') for i in annot[2]]
        self.labelnames = [i[2:-4] for i in labels[1:]]

    @memory.cache
    def get_data(self):

        for run in self.runs:
            hemisphere_data = []
            for h in [0, 1]:
                annot_path = join(self.flex_dir, 'fmri',
                                  'completed_preprocessed', self.subject,
                                  'freesurfer', self.subject,
                                  'label', '{0}.HCPMMP1.annot'.format(hemispheres[h]))
                hemi_func_path = glob(join(self.flex_dir, 'fmri',
                                           'completed_preprocessed', self.subject,
                                           'fmriprep', self.subject,
                                           self.session, 'func',
                                           '*{0}*fsnative.{1}.func.gii'.
                                           format(run, hemis[h])))[0]
                annot = ni.freesurfer.io.read_annot(annot_path)
                self.annotation[hemis[h]] = annot[0]
                self.labels[hemis[h]] = [i.astype('str') for i in annot[2]]
                surf = surface.load_surf_data(hemi_func_path)
                surf_df = pd.DataFrame(surf)
                surf_df.index = annot[0]
                hemisphere_data.append(surf_df)
            surf_df = pd.concat(hemisphere_data)
            self.whole_cortex[run] = surf_df

    def trim_data(self, roi_str):
        self.features = None
        self.behavioral = None
        neural = []
        behavoral = []
        roi_names = [hemis[0] + '_' + roi_str + '_ROI']
        roi_index = self.labels[hemis[0]].index(roi_names[0])
        for run, surf_df in self.whole_cortex.items():
            roi = surf_df.loc[roi_index].reset_index(drop=True).T
            roi = (roi - roi.mean()) / roi.std()
            dt = pd.to_timedelta(roi.index.values * 1900, unit='ms')
            roi = roi.set_index(dt)
            target = roi.resample('100ms').mean().index
            roi = pd.concat([interp(dt, roi[c], target) for c in roi.columns],
                            axis=1)
            behav = pd.read_hdf(join(self.flex_dir, 'Workflow',
                                     'Sublevel_GLM_Climag_2020-01-07',
                                     self.subject, 'BehavFrame_{0}_{1}.hdf'.
                                     format(self.subject, self.session)),
                                key=run)
            choices = pd.DataFrame({'rule_response': behav.loc[behav.event == 'CHOICE_TRIAL_RESP', 'rule_resp'].values.astype(float),
                                    'stimulus': behav.stimulus.dropna(how='any').values,
                                    'response': behav.loc[behav.event == 'CHOICE_TRIAL_RESP', 'value'].values.astype(float),
                                    'onset': behav.loc[behav.event == 'CHOICE_TRIAL_ONSET'].onset.values.astype(float)})
            choices = choices.loc[~choices.response.isnull()]
            onsets = choices.onset.values.astype(float)
            bl = pd.Timedelta(2000, unit='ms')
            te = pd.Timedelta(12000, unit='ms')
            for onset in onsets:
                cue = pd.Timedelta(onset, unit='s').round('100ms')
                task_evoked = roi.loc[cue - bl: cue + te]
                task_evoked = task_evoked.resample('1900ms').mean()
                task_evoked = task_evoked.iloc[:8]
                assert task_evoked.shape[0] == 8
                neural.append(task_evoked.values)
                behavoral.append(choices.loc[choices.onset == onset])

        self.features = np.transpose(np.dstack(neural))                            # 1st axis (rows): trials; 2nd axis (cols): vertices; 3rd axis: timepoints within trial
        self.behavioral = pd.concat(behavoral)

    def classify(self, parameter, timepoint):

        X = self.features[:, :, timepoint]
        y = self.behavioral[parameter].values
        linear_SVC = svm.LinearSVC(C=.1, max_iter=50000)

        aucs = []
        cv = StratifiedKFold(n_splits=10)
        for i, (train, test) in enumerate(cv.split(X, y)):
            linear_SVC.fit(X[train], y[train])
            viz = roc_auc_score(y[test], linear_SVC.predict(X[test]))
            aucs.append(viz)

        return np.mean(aucs)


def execute(sub, flex_dir):
    subject = 'sub-{}'.format(sub)
    list_of_dicts = []
    for session in ['ses-2', 'ses-3']:
        decoder = DecodeSurface(subject=subject, session=session, flex_dir=flex_dir)
        decoder.get_data()
        for roi_str in decoder.labelnames:
            decoder.trim_data(roi_str)
            for parameter in ['response', 'stimulus', 'rule_responses']:
                for timepoint in range(8):
                    mean_auc = decoder.classify(parameter=parameter, timepoint=timepoint)

                    list_of_dicts.append({
                        'session': session,
                        'subject': subject,
                        'roi': roi_str,
                        'parameter': parameter,
                        'timepoint': timepoint,
                        'roc_auc': mean_auc
                    })
                    print(list_of_dicts[-1])

    pd.DataFrame(list_of_dicts).to_hdf('CorticalDecoding.hdf', key=subject)


def climag_submit(subrange):
    flex_dir = '/home/khagena/FLEXRULE'
    for sub in subrange:
        pbs.pmap(execute, [(sub, flex_dir)],
                 walltime='4:00:00', memory=15, nodes=1, tasks=1,
                 name='decode_{0}'.format(sub))
