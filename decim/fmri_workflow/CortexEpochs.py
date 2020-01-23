import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
from decim.adjuvant import pupil_grating_correct as pgc
from joblib import Memory
from os.path import expanduser
from decim.adjuvant import slurm_submit as slu
if expanduser('~') == '/home/faty014':
    cachedir = expanduser('/work/faty014/joblib_cache')
else:
    cachedir = expanduser('~/joblib_cache')
slu.mkdir_p(cachedir)
memory = Memory(cachedir=cachedir, verbose=0)

'''
Extract pupil, behavioral and ROI time series data per choice epoch and build a gigantic pd.MultiIndex DataFrame

1. Extract behavioral parameters per choice trial ("choice_behavior")
    a) response, rule_response, stimulus, reward, accumulated belief
2. Extract last points seen before choice ("points")
3. Extract pupil time series per choice epoch ("choice_pupil")
    a) locked to onset of grating stimulus (-1s to 3.5s)
    b) locked to choice (-1s to 1.5s)
    c) subtract baseline (-1s until grating) from both
4. Extract BOLD time series for different brainstem ROIS per Epoch
    a) resample from TE (1900ms) to new frequency (100ms)
    b) take onsets from behavior
    c) loop through onsets and extract epochs for brainstem ROIs
    d) epochs: -2s to 12s from onset
    e) subtract baseline (-2s to 2s from onset, see da Gee et al., eLife)

Necessary Input files:
    - preprocessed behavioral pd.DAtaFrame
    - preprocessed pupil pd.DataFrame
    - extracted brainstem ROI time series pd.DataFrame

'''


def interp(x, y, target):
    '''
    Interpolate
    '''
    f = interp1d(x.values.astype(int), y)
    target = target[target.values.astype(int) > min(x.values.astype(int))]
    return pd.DataFrame({y.name: f(target.values.astype(int))}, index=target)


def baseline_correct(grating, choice, length=1000):
    baseline = np.matrix((grating.loc[:, 0:length].mean(axis=1))).T
    return pd.DataFrame(np.matrix(grating) - baseline),\
        pd.DataFrame(np.matrix(choice) - baseline), baseline


class Choiceframe(object):

    def __init__(self, subject, session, run, task,
                 flex_dir, BehavFrame, CortRois):
        '''
        Initialize

        - Arguments:
            a) subject
            b) session
            c) run
            d) task
            e) Flexrule directory
            f) behavioral pd.DataFrame
            g) pupil pd.DataFrame
            e) extracted brainstem ROI time series
        '''
        self.subject = subject
        self.session = session
        self.run = run
        self.task = task
        self.flex_dir = flex_dir
        BehavFrame.onset = BehavFrame.onset.astype(float)
        BehavFrame = BehavFrame.sort_values(by='onset')
        self.BehavFrame = BehavFrame
        self.CortRois = CortRois

    def choice_behavior(self):
        df = self.BehavFrame
        choices = pd.DataFrame({'rule_response': df.loc[df.event == 'CHOICE_TRIAL_RESP', 'rule_resp'].values.astype(float),
                                'rt': df.loc[df.event == 'CHOICE_TRIAL_RESP'].rt.values.astype(float),
                                'stimulus': df.stimulus.dropna(how='any').values,
                                'response': df.loc[df.event == 'CHOICE_TRIAL_RESP', 'value'].values.astype(float),
                                'reward': df.loc[df.event == 'CHOICE_TRIAL_RESP'].reward.values.astype(float),
                                'onset': df.loc[df.event == 'CHOICE_TRIAL_ONSET'].onset.values.astype(float)})
        if self.task == 'inference':
            df.belief = df.belief.fillna(method='ffill')
            choices['trial_id'] =\
                df.loc[df.event == 'CHOICE_TRIAL_ONSET'].trial_id.values.astype(int)
            choices['accumulated_belief'] =\
                df.loc[df.event == 'CHOICE_TRIAL_ONSET'].belief.values.astype(float)
            choices['rewarded_rule'] =\
                df.loc[df.event == 'CHOICE_TRIAL_ONSET'].gen_side.values + 0.5
        elif self.task == 'instructed':
            choices['trial_id'] = np.arange(1, len(choices) + 1, 1)
            choices['accumulated_belief'] = np.nan
            choices['rewarded_rule'] =\
                df.loc[df.event == 'CHOICE_TRIAL_ONSET'].rewarded_rule.values
        self.choice_behavior = choices

    def fmri_epochs(self, basel=2000, te=12000, freq='100ms',
                    ROIs=[['RSC_ROI', 'POS2_ROI', 'PCV_ROI', '7Pm_ROI', '8BM_ROI', 'AVI_ROI',
                           'IP2_ROI', 'IP1_ROI', 'FOP5_ROI', 'a32pr_ROI']]):
        '''
        Loop through choice trial and extract fmri epochs for brainstem ROIs


        - Arguments:
            a) basline period in ms (baseline -basel to +basel from onset)
            b) epoch length from onset on in ms
            c) target frequency for resampling the ROI time series
            d) list of ROI names
        '''
        roi = self.CortRois
        roi = roi.loc[:, ROIs]
        dt = pd.to_timedelta(roi.index.values * 1900, unit='ms')
        roi = roi.set_index(dt)
        target = roi.resample(freq).mean().index
        roi = pd.concat([interp(dt, roi[c], target) for c in roi.columns], axis=1)
        behav = self.choice_behavior
        onsets = behav.onset.values
        evoked_run = defaultdict(list)
        bl = pd.Timedelta(basel, unit='ms')
        te = pd.Timedelta(te, unit='ms')
        for onset in onsets:
            cue = pd.Timedelta(onset, unit='s').round('ms')
            baseline = roi.loc[cue - bl: cue + bl].mean()
            task_evoked = roi.loc[cue - bl: cue + te] - baseline
            for col in task_evoked.columns:
                evoked_run[col].append(task_evoked[col].values)
        for key, values in evoked_run.items():
            df = pd.DataFrame(values)
            evoked_run[key] = df
        self.roi_epochs = evoked_run

    def merge(self):
        '''
        Merge everything into one pd.MultiIndex pd.DataFrame.
        '''

        self.choice_behavior.columns =\
            pd.MultiIndex.from_product([['behavior'], ['parameters'],
                                        self.choice_behavior.columns],
                                       names=['source', 'type', 'name'])
        master = self.choice_behavior
        master = master.set_index([master.behavior.parameters.trial_id])
        singles = []
        for key, frame in self.roi_epochs.items():
            frame.columns = pd.MultiIndex.from_product([['fmri'], [key],
                                                        frame.columns],
                                                       names=['source', 'type',
                                                              'name'])
            singles.append(frame)
        fmri = pd.concat(singles, axis=1)
        self.master = pd.merge(fmri.set_index(master.index, drop=True).
                               reset_index(), master.reset_index())


#@memory.cache
def execute(subject, session, run, task,
            flex_dir, BehavFrame, CortRois):
    '''
    Execute per subject, session, task and run.

    Moreover need to give
        - Flexrule directory
        - preprocessed behavioral pd.DAtaFrame
        - preprocessed pupil pd.DataFrame
        - extracted brainstem ROI time series pd.DataFrame
    '''
    c = Choiceframe(subject, session, run, task,
                    flex_dir, BehavFrame, CortRois)
    c.choice_behavior()
    c.fmri_epochs()
    c.merge()
    return c.master


__version__ = '2.0'
'''
2.0
-Input linear pupilframes
-recquires BIDS
1.2
-triallocked period now 1000ms before offset and total of 4500ms
-if rt > 2000ms choicelocked is set to np.nan
'''
