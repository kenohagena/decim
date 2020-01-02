from pyedfread import edf
import numpy as np
import pandas as pd
import math
from scipy import signal


class Localizer(object):

    def __init__(self, file):
        self.file = file

    def load_pupil(self):
        '''
        Loads sa, ev & m of edf file
        '''
        return edf.pread(self.file)

    def get_pupil(self, sa):
        '''
        Extracts the pupilsize, and gazelocations from the sa file.
        '''
        assert (sa.mean().pa_right > 0) or (sa.mean().pa_left > 0)
        if (sa.mean().pa_right < 0) & (sa.mean().pa_left > 0):
            pupilside = 'left'
        else:
            pupilside = 'right'

        sa_frame = sa.loc[:, ['pa_{}'.format(pupilside), 'time',
                              'gx_{}'.format(pupilside),
                              'gy_{}'.format(pupilside)]]

        return sa_frame

    def get_events(self, ev):
        '''
        Loads automatically detected blinks and saccades from ev file.

        Returns np.arrays with orig. time loactions of blinked/saccaded samples
        '''
        sac_frame = ev[ev.type == 'saccade'].loc[:, ['start', 'end']]
        sactime = []
        for i, row in sac_frame.iterrows():
            sactime.append(list(range(row.start, row.end)))
        sactime = sum(sactime, [])

        blink_frame = ev[ev.blink == True].loc[:, ['start', 'end']]
        blinktime = []
        for i, row in blink_frame.iterrows():
            blinktime.append(list(range(row.start, row.end)))
        blinktime = sum(blinktime, [])
        return blinktime, sactime

    def get_messages(self, m):
        '''
        Takes message frame and returns a one with just time, message and value
        '''
        message_frame = m.loc[:, ['CHOICE_TRIAL_ONSET',
                                  'CHOICE_TRIAL_ONSET_time',
                                  'CHOICE_TRIAL_STIMOFF',
                                  'CHOICE_TRIAL_STIMOFF_time',
                                  'LOCALIZER_STIM', 'LOCALIZER_STIM_time']]

        onset = message_frame.loc[:, ['CHOICE_TRIAL_ONSET',
                                      'CHOICE_TRIAL_ONSET_time']].\
            dropna(how='all')
        onset = onset.rename(columns={'CHOICE_TRIAL_ONSET_time': 'time',
                                      'CHOICE_TRIAL_ONSET': 'stim_ID'})
        onset['message'] = 'CHOICE_TRIAL_ONSET'
        onset['stim_ID'] = message_frame.loc[:, 'LOCALIZER_STIM'].dropna()

        stimoff = message_frame.loc[:, ['CHOICE_TRIAL_STIMOFF',
                                        'CHOICE_TRIAL_STIMOFF_time']].\
            dropna(how='all')
        stimoff = stimoff.rename(columns={'CHOICE_TRIAL_STIMOFF_time': 'time',
                                          'CHOICE_TRIAL_STIMOFF': 'stim_ID'})
        stimoff['message'] = 'CHOICE_TRIAL_STIMOFF'
        stimoff['stim_ID'] = message_frame.loc[:, 'LOCALIZER_STIM'].dropna()

        return pd.concat([onset, stimoff])

    def basicframe(self, events=True, messages=True, save_path=None):
        '''
        Loads pupil data and optionally events & messages.

        Merges and concatenates to one dataframe (per subject, session, block).
        '''
        sa, ev, m = self.load_pupil()
        pupil_frame = self.get_pupil(sa)
        self.pupilside = pupil_frame.columns[0][3:]
        if events is True:
            blinktime, sactime = self.get_events(ev)
            pupil_frame['saccade'] = pupil_frame['time'].isin(sactime)
            pupil_frame['blink'] = pupil_frame['time'].isin(blinktime)
        if messages is True:
            messages = self.get_messages(m)
            pupil_frame = pd.merge(pupil_frame, messages,
                                   how='left', on=['time'])

        pupil_frame = pupil_frame.drop(pupil_frame[pupil_frame.time == 0]
                                       .index)

        self.pupil_frame = pupil_frame

    def gaze_angle(self, screen_distance=600, monitor_width=1920,
                   monitor_height=1200, pixelsize=.252):
        '''
        Computes angle of gaze in degrees.

        Takes gaze coordniates, screen distance and monitor parameters.
        '''
        self.pupil_frame['gaze_angle'] = ((((self.pupil_frame['gx_{}'.format(self.pupilside)] -
                                             monitor_width / 2)**2 + (self.pupil_frame['gy_{}'.format(self.pupilside)] -
                                                                      monitor_height / 2)**2)**.5 * pixelsize) / screen_distance)
        self.pupil_frame['gaze_angle'] = self.pupil_frame.gaze_angle.apply(np.arctan).values
        self.pupil_frame['gaze_angle'] = self.pupil_frame.gaze_angle.apply(math.degrees).values

    def all_artifacts(self, blink=True, saccade=True, gaze_angle=True, gaze_thresh=4, plot_column=False):
        '''
        Adds column with all methods of artifact detection enbaled.

        True means 'is an artifact'
        '''
        self.pupil_frame['all_artifacts'] = bool(0)
        if blink is True:
            self.pupil_frame.all_artifacts = self.pupil_frame.blink == True
        if saccade is True:
            self.pupil_frame.all_artifacts = (self.pupil_frame.all_artifacts == True) | (self.pupil_frame.saccade == True)
        if gaze_angle is True:
            self.pupil_frame.all_artifacts = (self.pupil_frame.all_artifacts == True) | (self.pupil_frame.gaze_angle > gaze_thresh)

    def small_fragments(self, crit_frag_length=100):
        '''
        Detects leftover fragments smaller than threshhold.

        Sets those detected fragments to NaN in order to make linear interpolation cleaner.
        '''
        convolved = np.convolve(self.pupil_frame.all_artifacts, [0.5, 1], 'same')
        ev_start = np.where(convolved == .5)[0]
        fragment_ends = ev_start
        if convolved[0] != 0:
            fragment_ends = fragment_ends[1:len(fragment_ends)]
        if convolved[len(convolved) - 1] == 0:
            fragment_ends = np.append(fragment_ends, len(self.pupil_frame))

        ev_end = np.where(convolved == 1)[0]
        if convolved[0] == 0:
            fragment_starts = np.append(0, ev_end)
        else:
            fragment_starts = ev_end
        assert len(fragment_ends) == len(fragment_starts)
        fragment_length = fragment_ends - fragment_starts
        wh = np.where(fragment_length < crit_frag_length)
        smallfrag_ends = fragment_ends[wh]
        smallfrag_starts = fragment_starts[wh]
        for i in range(len(smallfrag_starts)):
            self.pupil_frame.all_artifacts = (self.pupil_frame.all_artifacts == True) | (self.pupil_frame.index.isin(range(smallfrag_starts[i], smallfrag_ends[i] + 1)))

    def interpol(self, margin=100):
        convolved = np.convolve(self.pupil_frame.all_artifacts, [0.5, 1], 'same')
        ev_start = np.where(convolved == .5)[0]
        ev_end = np.where(convolved == 1)[0]
        if convolved[len(convolved) - 1] > 0:
            ev_end = np.append(ev_end, len(self.pupil_frame) - 1)
        pupil_interpolated = np.array(self.pupil_frame['pa_{}'.format(self.pupilside)].copy())
        for b in range(len(ev_start)):
            if ev_start[b] < margin:
                start = 0
            else:
                start = ev_start[b] - margin + 1
            if ev_end[b] + margin + 1 > len(self.pupil_frame) - 1:
                end = len(self.pupil_frame) - 1
            else:
                end = ev_end[b] + margin + 1
            interpolated_signal = np.linspace(pupil_interpolated[start],
                                              pupil_interpolated[end],
                                              end - start,
                                              endpoint=False)
            pupil_interpolated[start:end] = interpolated_signal
        self.pupil_frame['interpol'] = pupil_interpolated

    def filter(self, highpass=.01, lowpass=6, sample_rate=1000):
        '''
        Apply 3rd order Butterworth bandpass filter.
        '''
        # High pass:
        pupil_interpolated = self.pupil_frame.interpol
        hp_cof_sample = highpass / (sample_rate / 2)
        bhp, ahp = signal.butter(3, hp_cof_sample, btype='high')
        pupil_interpolated_hp = signal.filtfilt(bhp, ahp, pupil_interpolated)
        # Low pass:
        lp_cof_sample = lowpass / (sample_rate / 2)

        blp, alp = signal.butter(3, lp_cof_sample)
        pupil_interpolated_lp = signal.filtfilt(blp, alp, pupil_interpolated)
        # Band pass:
        pupil_interpolated_bp = signal.filtfilt(blp, alp, pupil_interpolated_hp)

        self.pupil_frame['bp_interpol'] = pupil_interpolated_bp

    def z_score(self):
        '''
        Normalize over session
        '''
        self.pupil_frame['biz'] = (self.pupil_frame.bp_interpol -
                                   self.pupil_frame.bp_interpol.mean()) /\
            self.pupil_frame.bp_interpol.std()

    def demean(self):
        '''
        Subtract mean. Instead of z_score.
        '''
        self.pupil_frame['demean'] = self.pupil_frame.bp_interpol - self.pupil_frame.bp_interpol.mean()

    def reframe(self, tw=4500, baseline=1000, which='biz'):
        '''
        Reframe: One row per 'Trial'

        'tw' is length of windo of interest per trial
        'baseline' is length of interval before onset of grating to be included
        '''
        df = self.pupil_frame.loc[:, ['message', which, 'stim_ID', 'blink', 'all_artifacts', 'gaze_angle', 'time']]
        choicepupil = []

        for onset in df.loc[df.message == "CHOICE_TRIAL_ONSET"].index:
            trial = df.loc[np.arange(onset - baseline, onset + tw - baseline).astype(int), which].values
            trial = np.append(trial, [df.loc[np.arange(onset, onset + 2000), 'all_artifacts'].mean(),
                                      df.loc[np.arange(onset, onset + 2000), 'blink'].mean(),
                                      df.iloc[onset: onset + 3000, :].loc[df.message == 'CHOICE_TRIAL_STIMOFF', 'time'].index[0] - onset,
                                      df.iloc[onset].stim_ID])

            choicepupil.append(trial)
        choicedf = pd.DataFrame(choicepupil)

        # subtract baseline per trial
        for i, row in choicedf.iterrows():
            choicedf.iloc[i, 0:tw] = (choicedf.iloc[i, 0:tw] - choicedf.iloc[i, 0:1000].mean())

        choicedf = choicedf.rename(index=str, columns={tw: "all_artifacts", tw + 1: "blink", tw + 2: "stimoff", tw + 3: 'stimulus_ID'})
        self.reframe = choicedf
