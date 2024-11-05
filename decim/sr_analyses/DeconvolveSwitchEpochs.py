import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.signal import convolve, detrend
from numpy.linalg import pinv


'''
Use deconcolution to construct event evoked responses

Keysteps:
1. Get event timestamps and convert to pupil frame time
2. Downsample pupil to 50 Hz (comparable to fmri which can be upsampled)
3. Deconvolve:
    3.1 Y is a matrix of shape Nx1, where N is the length of timeseries (e.g. basically the time series of a run)
    3.2 X is a design matrix of the shape NxP (P is the number of peri event samples, here 15 s of interest)
    3.3 make the pseudoinverse of X (X+) and multiply X+ with Y to have the deconvovled resposne

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

    def get_behavior_onsets(self, type='belief'):
        '''
        Indicate whether 'belief' switches or 'generative' switches
        Gets onsets in behavior time
        '''
        df = self.BehavFrame

        df.loc[:, 'gen_switch'] = np.roll(df.gen_side.values, 1)
        df.loc[:, 'gen_switch'] = df.loc[:, 'gen_side'] - df.loc[:, 'gen_switch']
        df.loc[0, 'gen_switch'] = 0

        if type == 'belief':

            self.behav_onsets = df.loc[df.switch.isin([-1, 1])].onset.values

        elif type == 'generative':
            self.onsets = df.loc[df.gen_switch.isin([-1, 1])].onset.values

    def choice_pupil(self, tw=15):
        '''
        Extract pupil time series epochs at switches in Glaze belief

        - Arguments:
            a) total epoch length in seconds
        '''
        self.PupilFrame.index = pd.to_datetime(self.PupilFrame.index, unit='ms')
        df = self.BehavFrame
        behav_onsets = self.BehavFrame.loc[self.BehavFrame.event ==
                                           'CHOICE_TRIAL_ONSET'].onset.values
        pupil_onsets = self.PupilFrame.loc[self.PupilFrame.message ==
                                           'CHOICE_TRIAL_ONSET'].time.values
        difference = pupil_onsets / 1000 - behav_onsets                         # Pupil and behavioral data have different time logs
        assert difference.std() < 0.05                                          # Sanity Check I: Transformation between both time scales works
        pupil_onsets = (self.behav_onsets + difference.mean()) * 1000                      # Time points transformed to pupil time scale
        switch_indices = self.PupilFrame.loc[self.PupilFrame.time.
                                         isin(pupil_onsets.
                                              astype(int))].index
        assert len(switch_indices) == len(self.behav_onsets)                                # Sanity Check II: After transformation same amount of switch indices

        pupil_resampled = self.PupilFrame.resample('20ms').mean() # resample pupil to 50 Hz
        self.resampled_signal = detrend(pupil_resampled.biz.values) # detrend
        switch_indices_resampled = switch_indices.round('20ms') #round switch indices to 50Hz
        self.positions = [pupil_resampled.index.get_loc(target) for target in switch_indices_resampled if target in pupil_resampled.index]

    def deconvolve(self, tw=15, freq=50):
        '''
        Deconvolution, 'tw': window length in seconds, 'freq': sampling frequency in Hz
        '''
        X = np.zeros((self.resampled_signal.shape[0], tw*freq))
        for i, event in enumerate(self.positions):
            for j in range(tw*freq):
                try:
                    X[event+j, j] = 1
                except IndexError:
                    continue
        X_pseudo_inv = pinv(X)  # Pseudoinverse of the design matrix
        self.R_estimated = X_pseudo_inv @ self.resampled_signal  # Estimated neural response


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
    c.get_behavior_onsets(switch_type)
    c.choice_pupil()
    c.deconvolve()
    return c.R_estimated

__version__= 1.1



