from pyedfread import edf
from pyedfread import edfread
import numpy as np
import pandas as pd
from glob import glob
from os.path import join
import math
from scipy import signal


class Pupilframe(object):

    def __init__(self, subject, session, block, base_path, pupil_frame=None):
        self.subject = 'VPIM0{}'.format(subject)
        sesdict = {1: 'A', 2: 'B', 3: 'C'}
        self.session = sesdict[session]
        self.block = block
        self.base_path = base_path
        self.pupil_frame = pupil_frame
        if pupil_frame is None:
            self.pupilside = []
        else:
            self.pupilside = self.pupil_frame.columns[0][3:]

    def load_pupil(self):
        '''
        Concatenates path and file name and loads sa, ev, and m file.

        Recquires subject code, session, phase and block.
        '''
        sessions = {'A': 1, 'B': 3, 'C': 5}
        phase = sessions[self.session]
        directory = join(self.base_path,
                         "{}".format(self.subject),
                         "{}".format(self.session),
                         'PH_' + "{}".format(phase) +
                         'PH_' + "{}".format(self.block))
        files = glob(join(directory, '*.edf'))
        if len(files) > 1:
            raise RuntimeError(
                'More than one log file found for this block: %s' % files)
        elif len(files) == 0:
            raise RuntimeError(
                'No log file found for this block: %s, %s, %s, %s' %
                (self.subject, self.session, phase, self.block))
        return edf.pread(files[0], trial_marker=b'trial_id')

    def get_pupil(self, sa):
        '''
        Extracts the pupilsize, and gazelocations from the sa file.
        '''
        assert (sa.mean().pa_right > 0) or (sa.mean().pa_left > 0)
        if (sa.mean().pa_right < 0) & (sa.mean().pa_left > 0):
            pupilside = 'left'
        else:
            pupilside = 'right'

        sa_frame = sa.loc[:, ['pa_{}'.format(pupilside), 'time', 'gx_{}'.format(pupilside), 'gy_{}'.format(pupilside)]]

        return sa_frame

    def get_events(self, ev):
        '''
        Loads automatically detected blinks and saccades from ev file.

        Returns np.arrays with original time loactions of blinked/saccaded samples
        '''
        sac_frame = ev[ev.type == 'saccade'].loc[:, ['start', 'end']]
        sactime = []
        for i, row in sac_frame.iterrows():
            sactime.append(list(range(row.start, row.end)))
        sactime = sum(sactime, [])  # flatten the nested list

        blink_frame = ev[ev.blink == True].loc[:, ['start', 'end']]
        blinktime = []
        for i, row in blink_frame.iterrows():
            blinktime.append(list(range(row.start, row.end)))
        blinktime = sum(blinktime, [])
        return blinktime, sactime

    def get_messages(self, m):
        '''
        Takes message frame and returns a simpler frame containing the fields time, message and value.
        '''
        message_frame = m.loc[:, ['CHOICE_TRIAL_ONSET', 'CHOICE_TRIAL_ONSET_time',
                                  'CHOICE_TRIAL_RESP', 'CHOICE_TRIAL_RESP_time',
                                  'CHOICE_TRIAL_STIMOFF', 'CHOICE_TRIAL_STIMOFF_time',
                                  'GL_TRIAL_REWARD', 'GL_TRIAL_REWARD_time',
                                  'IR_TRIAL_REWARD', 'RT', 'RT_time',
                                  'gener_side', 'gener_side_time',
                                  'location', 'location_time',
                                  'stim_id', 'stim_id_time', 'trialid ']]

        onset = message_frame.loc[:, ['CHOICE_TRIAL_ONSET', 'CHOICE_TRIAL_ONSET_time']].dropna(how='all')
        onset = onset.rename(columns={'CHOICE_TRIAL_ONSET_time': 'time', 'CHOICE_TRIAL_ONSET': 'message_value'})
        onset['message'] = 'CHOICE_TRIAL_ONSET'
        onset['trial_id'] = [int(i[9:]) for i in message_frame.loc[~message_frame.CHOICE_TRIAL_ONSET.isnull(), 'trialid '].values]

        resp = message_frame.loc[:, ['CHOICE_TRIAL_RESP', 'CHOICE_TRIAL_RESP_time']].dropna(how='all')
        resp = resp.rename(columns={'CHOICE_TRIAL_RESP_time': 'time', 'CHOICE_TRIAL_RESP': 'message_value'})
        resp['message'] = 'CHOICE_TRIAL_RESP'

        stimoff = message_frame.loc[:, ['CHOICE_TRIAL_STIMOFF', 'CHOICE_TRIAL_STIMOFF_time']].dropna(how='all')
        stimoff = stimoff.rename(columns={'CHOICE_TRIAL_STIMOFF_time': 'time', 'CHOICE_TRIAL_STIMOFF': 'message_value'})
        stimoff['message'] = 'CHOICE_TRIAL_STIMOFF'

        reward = message_frame.loc[:, ['GL_TRIAL_REWARD', 'GL_TRIAL_REWARD_time']].dropna(how='all')
        reward = reward.rename(columns={'GL_TRIAL_REWARD_time': 'time', 'GL_TRIAL_REWARD': 'message_value'})
        reward['message'] = 'GL_TRIAL_REWARD'

        rt = message_frame.loc[:, ['RT', 'RT_time']].dropna(how='all')
        rt = rt.rename(columns={'RT_time': 'time', 'RT': 'message_value'})
        rt['message'] = 'RT'

        location = message_frame.loc[:, ['location', 'location_time']].dropna(how='all')
        location = location.rename(columns={'location_time': 'time', 'location': 'message_value'})
        location['message'] = 'location'

        return pd.concat([onset, resp, stimoff, reward, rt, location])

    def basicframe(self, events=True, messages=True):
        '''
        Loads pupil data and optionally events, messages, velocity and acceleration.

        merges and concatenates to one dataframe (per subject, session & block).
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
            pupil_frame = pd.merge(pupil_frame, messages, how='left', on=['time'])

        pupil_frame = pupil_frame.drop(pupil_frame[pupil_frame.time == 0].index)

        self.pupil_frame = pupil_frame

    def get_velocity(self, Hz=1000):
        '''
        Compute velocity of eye-movements.

        'x' and 'y' specify the x,y coordinates of gaze location. The function
        assumes that the values in x,y are sampled continously at a rate specified
        by 'Hz'.
        '''

        x = np.array(self.pupil_frame['gx_{}'.format(self.pupilside)])
        y = np.array(self.pupil_frame['gy_{}'.format(self.pupilside)])

        Hz = float(Hz)
        distance = ((np.diff(x) ** 2) +
                    (np.diff(y) ** 2)) ** .5
        distance = np.hstack(([distance[0]], distance))
        win = np.ones((3)) / float(3)
        velocity = np.convolve(distance, win, mode='same')
        velocity = velocity / (3 / Hz)
        acceleration = np.diff(velocity) / (1. / Hz)
        acceleration = np.hstack(([acceleration[0]], acceleration))
        self.pupil_frame['vel'] = velocity
        self.pupil_frame['acc'] = acceleration

    def gaze_angle(self, screen_distance=600, monitor_width=1920, monitor_height=1200, pixelsize=.252):
        '''
        Computes angle of gaze in degrees based on gaze coordniates, screen distance and monitor parameters.
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

    def chop(self):
        '''
        If time series starts or ends with an aritfact/blink it will be cut off from the whole frame.
        '''
        convolved = np.convolve(self.pupil_frame.all_artifacts, [0.5, 1], 'same')
        ev_start = np.where(convolved == .5)[0]
        ev_end = np.where(convolved == 1)[0]
        if ev_start[0] == 0:
            self.pupil_frame = self.pupil_frame.iloc[ev_end[0] + 1:, :]
        if convolved[len(convolved) - 1] > 0:
            self.pupil_frame = self.pupil_frame.iloc[:ev_start[len(ev_start) - 1], :]

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
        self.pupil_frame['biz'] = (self.pupil_frame.bp_interpol - self.pupil_frame.bp_interpol.mean()) / self.pupil_frame.bp_interpol.std()


#p = Pupilframe(1, 3, 4, '/Users/kenohagena/Documents/immuno/data/vaccine/')
# p.basicframe()
# print(p.pupil_frame.loc[~p.pupil_frame.message.isnull()])
