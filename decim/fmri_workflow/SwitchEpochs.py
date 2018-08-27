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
        switches = pd.DataFrame({'switch_left': df.loc[df.switch == True].
                                 switch_left.values,
                                 'switch_right': df.loc[df.switch == True].
                                 switch_right.values,
                                 'switch_surprise': df.loc[df.switch == True].
                                 surprise.values,
                                 'onset': df.loc[df.switch == True].
                                 onset.values.astype(float)})
        self.switch_behavior = switches

    def points(self, n=20):
        '''
        Add last n points before choice onset.
        '''
        df = self.BehavFrame
        points = df.loc[(df.event == 'GL_TRIAL_LOCATION')]
        p = []
        for i, row in self.switch_behavior.iterrows():
            trial_points = points.loc[points.onset.astype('float') < row.onset]
            if len(trial_points) < 20:
                trial_points = np.full(20, np.nan)
            else:
                trial_points = trial_points.value.values[len(trial_points) - 20:len(trial_points)]
            p.append(trial_points)
        points = pd.DataFrame(p)
        self.point_kernels = points

    def choice_pupil(self, artifact_threshhold=.2, tw=4500):
        '''
        Takes existing pupilframe and makes choicepupil frame.
        If there is no existing pupilframe, a new one is created.
        '''
        behav_onsets = self.BehavFrame.loc[self.BehavFrame.event ==
                                           'CHOICE_TRIAL_ONSET'].onset.values
        pupil_onsets = self.PupilFrame.loc[self.PupilFrame.message ==
                                           'CHOICE_TRIAL_ONSET'].time.values
        difference = pupil_onsets / 1000 - behav_onsets
        assert difference.std() < 0.05
        switch_onsets = df.loc[df.switch == True].onset.values
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
                pupil_switch_lock.append(df.loc[np.arange(switch - 1000, switch + tw - 1000).\
                astype(int), 'biz'].values)
                blink_mean.append(df.loc[np.arange(switch - 500, switch + 1500), 'blink'].mean())
        pupil_switch_lock = pd.DataFrame(pupil_switch_lock)
        baseline = np.matrix((pupil_switch_lock.loc[:, 0:1000].mean(axis=1))).T
        pupil_switch_lock = pd.DataFrame(np.matrix(pupil_switch_lock) - baseline)
        self.pupil_switch_lock = pupil_switch_lock
        self.pupil_parameters = pd.DataFrame({'blink': blink_mean})
        self.pupil_parameters['TPR'] = self.pupil_choice_lock.loc[:, 500:2500].mean(axis=1)

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
        master = master.set_index([master.pupil.parameters.onset])
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
    c = Choiceframe(subject, session, run, task, flex_dir,
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
