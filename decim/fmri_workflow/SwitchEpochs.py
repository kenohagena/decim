import pandas as pd
import numpy as np
from decim.fmri_workflow import LinregVoxel as fa
from collections import defaultdict

'''
This script is used to build pd.DataFrame for switch epochs

1. Extract switch information (onset, direction, index) from behavioral pd.DataFrame
2. Extract the Glaze belief for 11 timepoints surrounding the switch to validate only "true switches"
3. Extract pupil time series per switch epoch
    a) locked to switch from BehavFrame (-1s to 3.5s)
    c) subtract baseline (-1s until grating)
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


class Choiceframe(object):

    def __init__(self, subject, session, run, flex_dir,
                 BehavFrame, PupilFrame, BrainstemRois):
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
        switches = pd.DataFrame({'onset': df.loc[df.switch == True].
                                 onset.values,
                                 'direction': df.loc[df.switch == True].
                                 switch.values,
                                 'switch_index': df.loc[df.switch == True].
                                 index.values})
        self.switch_behavior = switches

    def points(self):
        '''
        Add belief values of 11 surrounding samples to later find true switches
        '''
        df = self.BehavFrame
        points = df.loc[(df.event == 'GL_TRIAL_LOCATION')].reset_index()
        p = []
        for i, row in self.switch_behavior.iterrows():
            switch_point = points.loc[points['index'] ==
                                      row.switch_index].index[0]
            if switch_point < 5:
                trial_points = points.iloc[0:switch_point + 6]
                p.append(np.append(np.zeros(11 - len(trial_points)),
                                   trial_points.belief.values))
            else:
                trial_points = points.iloc[switch_point - 5:switch_point + 6]
                p.append(trial_points.belief.values)
        self.point_kernels = pd.DataFrame(p)

    def choice_pupil(self, tw=4500):
        '''
        Extract pupil time series epochs at switches in Glaze belief

        - Arguments:
            a) total epoch length in ms
        '''
        df = self.BehavFrame
        behav_onsets = self.BehavFrame.loc[self.BehavFrame.event ==
                                           'CHOICE_TRIAL_ONSET'].onset.values
        pupil_onsets = self.PupilFrame.loc[self.PupilFrame.message ==
                                           'CHOICE_TRIAL_ONSET'].time.values
        difference = pupil_onsets / 1000 - behav_onsets                         # Pupil and behavioral data have different time logs
        assert difference.std() < 0.05                                          # Sanity Check I: Transformation between both time scales works
        swo_behav = df.loc[df.switch != 0].onset.values                         # Switch time points extracted from behavioral data
        swo_pupil = (swo_behav + difference.mean()) * 1000                      # Time points transformed to pupil time scale
        sw_indices = self.PupilFrame.loc[self.PupilFrame.time.
                                         isin(swo_pupil.
                                              astype(int))].index
        assert len(sw_indices) == len(swo_behav)                                # Sanity Check II: After transformation same amount of switch indices
        df = self.PupilFrame.loc[:, ['message', 'biz', 'message_value',
                                     'blink', 'run', 'trial_id']]
        pupil_swl = []
        blink_mean = []
        for switch in sw_indices:
            pupil_swl.append(df.loc[np.arange(switch - 1000,
                                              switch + tw - 1000).
                                    astype(int), 'biz'].values)                 # Pupil epoch
            blink_mean.append(df.loc[np.arange(switch - 500, switch + 1500),
                                     'blink'].mean())                           # Mean artifact during 2 most important seconds
        pupil_swl = pd.DataFrame(pupil_swl)
        baseline = np.matrix((pupil_swl.loc[:, 0:1000].mean(axis=1))).T
        pupil_swl = pd.DataFrame(np.matrix(pupil_swl) -
                                 baseline)
        self.pupil_switch_lock = pupil_swl
        self.pupil_parameters = pd.DataFrame({'blink': blink_mean})
        self.pupil_parameters['TPR'] = self.pupil_switch_lock.loc[:, 500:2500].\
            mean(axis=1)

    def fmri_epochs(self, basel=2000, te=12000, freq='100ms',
                    ROIs=['aan_dr', 'zaborsky_bf4', 'zaborsky_bf123',
                          'keren_lc_1std', 'NAc', 'SNc', 'VTA', '4th_ventricle']):
        '''
        Loop through switches and extract fmri epochs for brainstem ROIs


        - Arguments:
            a) basline period in ms (baseline -basel to +basel from onset)
            b) epoch length from onset on in ms
            c) target frequency for resampling the ROI time series
            d) list of ROI names
        '''
        roi = self.BrainstemRois
        roi = roi.loc[:, ROIs]
        dt = pd.to_timedelta(roi.index.values * 1900, unit='ms')
        roi = roi.set_index(dt)
        target = roi.resample(freq).mean().index
        roi = pd.concat([fa.interp(dt, roi[c], target) for c in roi.columns],
                        axis=1)
        onsets = self.switch_behavior.onset.values
        evoked_run = defaultdict(list)
        bl = pd.Timedelta(basel, unit='ms')
        te = pd.Timedelta(te, unit='ms')
        for onset in onsets:
            cue = pd.Timedelta(onset, unit='s').round('ms')
            bl = pd.Timedelta(basel, unit='ms')
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
        Merge everything
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
        print(master.head(), master.index, master.shape)
        master = master.set_index([master.behavior.parameters.onset])
        print(master.head(), master.index, master.shape)
        singles = []
        for key, frame in self.roi_epochs.items():
            frame.columns = pd.MultiIndex.from_product([['fmri'], [key],
                                                        frame.columns],
                                                       names=['source', 'type', 'name'])
            singles.append(frame)
        fmri = pd.concat(singles, axis=1)
        print(fmri.head(), fmri.shape, master.index, fmri.index)
        self.master = pd.merge(fmri.set_index(master.index, drop=True).reset_index(),
                               master.reset_index())


def execute(subject, session, run, task, flex_dir,
            BehavFrame, PupilFrame, BrainstemRois):
    '''
    Execute per subject, session, task and run.

    Moreover need to give
        - Flexrule directory
        - preprocessed behavioral pd.DAtaFrame
        - preprocessed pupil pd.DataFrame
        - extracted brainstem ROI time series pd.DataFrame
    '''
    c = Choiceframe(subject, session, run, flex_dir,
                    BehavFrame, PupilFrame, BrainstemRois)
    c.choice_behavior()
    if task == 'inference':
        c.points()
    elif task == 'instructed':
        c.point_kernels = pd.DataFrame(np.zeros((c.switch_behavior.shape[0], 11)))
    c.choice_pupil()
    c.fmri_epochs()
    c.merge()
    return c.master


'''
behav = pd.read_hdf('/Volumes/flxrl/FLEXRULE/SubjectLevel/sub-17/BehavFrame_sub-17_ses-2.hdf', key='inference_run-4')
pupil = pd.read_hdf('/Volumes/flxrl/FLEXRULE/pupil/linear_pupilframes/PupilFrame_17_ses-2.hdf', key='/inference_run-4')
brain = pd.read_hdf('/Volumes/flxrl/FLEXRULE/SubjectLevel/sub-17/BrainstemRois_sub-17_ses-2.hdf', key='inference_run-4')
master = execute('sub-17', 'ses-2', 'inference_run-4', 'inference', '/Volumes/flxrl/FLEXRULE/', behav, pupil, brain)
'''
