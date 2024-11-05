import pandas as pd
import numpy as np
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
                 BehavFrame, PupilFrame):
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

    def choice_behavior(self, type='belief'):
        '''
        Indicate whether 'belief' switches or 'generative' switches
        '''
        df = self.BehavFrame

        df.loc[:, 'gen_switch'] = np.roll(df.gen_side.values, 1)
        df.loc[:, 'gen_switch'] = df.loc[:, 'gen_side'] - df.loc[:, 'gen_switch']
        df.loc[0, 'gen_switch'] = 0

        if type == 'belief':

            self.onsets = df.loc[df.switch.isin([-1, 1])].onset.values
            onsets = pd.DataFrame({'onset': self.onsets,
                                   'direction': df.loc[df.switch.isin([-1, 1])].
                                   switch.values,
                                   'switch_index': df.loc[df.switch.isin([-1, 1])].
                                   index.values})
        elif type == 'generative':
            self.onsets = df.loc[df.gen_switch.isin([-1, 1])].onset.values
            onsets = pd.DataFrame({'onset': self.onsets,
                                   'direction': df.loc[df.gen_switch.isin([-1, 1])].
                                   gen_switch.values,
                                   'switch_index': df.loc[df.gen_switch.isin([-1, 1])].
                                   index.values})


        self.switch_behavior = onsets


    def choice_pupil(self, tw=16000):
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
        swo_behav = self.onsets                                                 # Switch time points extracted from behavioral data
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

        master = pd.concat([self.pupil_switch_lock,
                            self.pupil_parameters,
                            self.switch_behavior], axis=1)
        master = master.set_index([master.behavior.parameters.onset])
        self.master = master

def execute(subject, session, run, task, flex_dir,
            BehavFrame, PupilFrame, switch_type='belief'):
    '''
    Execute per subject, session, task and run.

    Moreover need to give
        - Flexrule directory
        - preprocessed behavioral pd.DAtaFrame
        - preprocessed pupil pd.DataFrame
        - extracted brainstem ROI time series pd.DataFrame
    '''
    c = Choiceframe(subject, session, run, flex_dir,
                    BehavFrame, PupilFrame)
    c.choice_behavior(switch_type)
    c.choice_pupil()
    c.merge()
    return c.master

__version__= 1.1
