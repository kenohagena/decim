import pandas as pd
import numpy as np
from decim.fmri_workflow import BehavDataframe as fa
from collections import defaultdict


class Choiceframe(object):

    def __init__(self, subject, session, run, flex_dir, BehavFrame, PupilFrame, BrainstemRois):
        '''
        Initialize
        '''
        self.subject = subject
        self.session = session
        self.run = run
        self.flex_dir = flex_dir
        BehavFrame.onset = BehavFrame.onset.astype(float)
        BehavFrame = BehavFrame.sort_values(by='onset')
        self.BehavFrame = BehavFrame
        self.PupilFrame = PupilFrame
        self.BrainstemRois = BrainstemRois

    def choice_behavior(self):
        df = self.BehavFrame
        switches = pd.DataFrame({'onset': df.loc[df.switch != 0].onset.values,
                                 'direction': df.loc[df.switch != 0].switch.values,
                                 'switch_index': df.loc[df.switch != 0].index.values})
        self.switch_behavior = switches

    def points(self):
        '''
        Add belief values of 11 surrounding samples to later find true switches
        '''
        df = self.BehavFrame
        points = df.loc[(df.event == 'GL_TRIAL_LOCATION')].reset_index()
        p = []
        for i, row in self.switch_behavior.iterrows():
            switch_point = points.loc[points['index'] == row.switch_index].index[0]
            if switch_point < 5:
                trial_points = points.iloc[0:switch_point + 6]
                p.append(np.append(np.zeros(11 - len(trial_points)), trial_points.belief.values))
            else:
                trial_points = points.iloc[switch_point - 5:switch_point + 6]
                p.append(trial_points.belief.values)
        self.point_kernels = pd.DataFrame(p)

    def choice_pupil(self, artifact_threshhold=.2, tw=4500):
        '''
        Takes existing pupilframe and makes choicepupil frame.
        If there is no existing pupilframe, a new one is created.
        '''
        df = self.BehavFrame
        behav_onsets = self.BehavFrame.loc[self.BehavFrame.event ==
                                           'CHOICE_TRIAL_ONSET'].onset.values
        pupil_onsets = self.PupilFrame.loc[self.PupilFrame.message ==
                                           'CHOICE_TRIAL_ONSET'].time.values
        difference = pupil_onsets / 1000 - behav_onsets
        assert difference.std() < 0.05
        switch_onsets = df.loc[df.switch != 0].onset.values
        switch_onsets_pupil = (switch_onsets + difference.mean()) * 1000
        switch_indices = self.PupilFrame.loc[self.PupilFrame.time.
                                             isin(switch_onsets_pupil.astype(int))].index
        assert len(switch_indices) == len(switch_onsets)
        df = self.PupilFrame.loc[:, ['message', 'biz', 'message_value',
                                     'blink', 'run', 'trial_id']]
        pupil_switch_lock = []
        blink_mean = []
        for switch in switch_indices:
            '''
            Extract gratinglocked pupilresponse, choicelocked pupil response & choice parameters
            '''
            if len(df.iloc[switch: switch + 3500, :].loc[df.message == 'RT', 'message_value']) == 0:
                continue
            else:
                pupil_switch_lock.append(df.loc[np.arange(switch - 1000, switch + tw - 1000).
                                                astype(int), 'biz'].values)
                blink_mean.append(df.loc[np.arange(switch - 500, switch + 1500), 'blink'].mean())
        pupil_switch_lock = pd.DataFrame(pupil_switch_lock)
        baseline = np.matrix((pupil_switch_lock.loc[:, 0:1000].mean(axis=1))).T
        pupil_switch_lock = pd.DataFrame(np.matrix(pupil_switch_lock) - baseline)
        self.pupil_switch_lock = pupil_switch_lock
        self.pupil_parameters = pd.DataFrame({'blink': blink_mean})
        self.pupil_parameters['TPR'] = self.pupil_switch_lock.loc[:, 500:2500].mean(axis=1)

    def fmri_epochs(self, basel=2000, te=6):
        roi = self.BrainstemRois
        roi = roi.loc[:, ['aan_dr', 'zaborsky_bf4',
                          'zaborsky_bf123', 'keren_lc_1std', 'NAc', 'SNc',
                          'VTA', '4th_ventricle']]
        dt = pd.to_timedelta(roi.index.values * 1900, unit='ms')
        roi = roi.set_index(dt)
        target = roi.resample('100ms').mean().index
        roi = pd.concat([fa.interp(dt, roi[c], target) for c in roi.columns], axis=1)
        behav = self.switch_behavior
        onsets = behav.onset.values
        evoked_run = defaultdict(list)
        for onset in onsets:
            cue = pd.Timedelta(onset, unit='s').round('ms')
            bl = pd.Timedelta(basel, unit='ms')
            baseline = roi.loc[cue - bl: cue + bl].mean()
            task_evoked = roi.loc[cue - bl: cue + bl * te] - baseline
            for col in task_evoked.columns:
                evoked_run[col].append(task_evoked[col].values)
        for key, values in evoked_run.items():
            df = pd.DataFrame(values)
            evoked_run[key] = df
        self.roi_epochs = evoked_run

    def merge(self):
        '''
        Merge everything. And Save.
        '''
        self.pupil_switch_lock.columns =\
            pd.MultiIndex.from_product([['pupil'], ['switch_lock'],
                                        range(self.pupil_switch_lock.shape[1])],
                                       names=['source', 'type', 'name'])
        self.pupil_parameters.columns =\
            pd.MultiIndex.from_product([['pupil'], ['parameters'],
                                        self.pupil_parameters.columns],
                                       names=['source', 'type', 'name'])
        self.switch_behavior.columns =\
            pd.MultiIndex.from_product([['behavior'], ['parameters'],
                                        self.switch_behavior.columns],
                                       names=['source', 'type', 'name'])
        self.point_kernels.columns =\
            pd.MultiIndex.from_product([['behavior'], ['points'],
                                        range(self.point_kernels.shape[1])],
                                       names=['source', 'type', 'name'])
        master = pd.concat([self.pupil_switch_lock,
                            self.pupil_parameters,
                            self.switch_behavior,
                            self.point_kernels], axis=1)
        master = master.set_index([master.behavior.parameters.onset])
        singles = []
        for key, frame in self.roi_epochs.items():
            frame.columns = pd.MultiIndex.from_product([['fmri'], [key],
                                                        frame.columns],
                                                       names=['source', 'type', 'name'])
            singles.append(frame)
        fmri = pd.concat(singles, axis=1)
        self.master = pd.merge(fmri.set_index(master.index, drop=True).reset_index(),
                               master.reset_index())


def execute(subject, session, run, task, flex_dir,
            BehavFrame, PupilFrame, BrainstemRois):
    c = Choiceframe(subject, session, run, flex_dir,
                    BehavFrame, PupilFrame, BrainstemRois)
    c.choice_behavior()
    if task == 'inference':
        c.points()
    elif task == 'instructed':
        c.point_kernels = pd.DataFrame(np.zeros((c.choice_behavior.shape[0], 20)))
    c.choice_pupil()
    c.fmri_epochs()
    c.merge()
    return c.master


'''
behav = pd.read_hdf('/Volumes/flxrl/FLEXRULE/SubjectLevel/sub-17/BehavFrame_sub-17_ses-2.hdf', key='inference_run-4')
pupil = pd.read_hdf('/Volumes/flxrl/FLEXRULE/SubjectLevel/sub-17/PupilFrame_sub-17_ses-2.hdf', key='inference_run-4')
brain = pd.read_hdf('/Volumes/flxrl/FLEXRULE/SubjectLevel/sub-17/BrainstemRois_sub-17_ses-2.hdf', key='inference_run-4')
master = execute('sub-17', 'ses-2', 'inference_run-4', 'inference', '/Volumes/flxrl/FLEXRULE/', behav, pupil, brain)
'''
