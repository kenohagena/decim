import math
import numpy as np
import pandas as pd

from glob import glob
from os.path import join
from pyedfread import edf
from scipy import signal
from decim import slurm_submit as slu
from tqdm import tqdm
import matplotlib.pyplot as plt

print('loaded_2')
class Pupilframe(object):

    def __init__(self, sub, ses, run_index, flex_dir, pupil_frame=None):
        self.sub = sub
        self.ses = ses
        self.subject = 'sub-{}'.format(sub)
        self.session = 'ses-{}'.format(ses)
        inf_runs = ['inference_run-4',
                    'inference_run-5',
                    'inference_run-6']
        self.run_index = run_index
        self.run = inf_runs[run_index]
        self.flex_dir = flex_dir
        self.pupil_frame = pupil_frame
        if pupil_frame is None:
            self.pupilside = []
        else:
            self.pupilside = self.pupil_frame.columns[0][3:]

    def load_pupil(self, directory=None):
        '''
        Concatenates path and file name and loads sa, ev, and m file.

        Recquires subject code, session, phase and block.
        '''
        if directory is not None:
            files = directory
        else:
            directory = join(self.flex_dir, 'raw', 'bids_mr_v1.1',
                             self.subject, self.session, 'func')
            files = glob(join(directory, '*{}*.edf'.format(self.run)))
        if len(files) > 1:
            raise RuntimeError(
                'More than one log file found for this block: %s' % files)
        elif len(files) == 0:
            raise RuntimeError(
                'No log file found for this block: %s, %s' %
                (self.subject, self.session))
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

    def basicframe(self, events=True, messages=True, directory=None):
        '''
        Loads pupil data and optionally events, messages, velocity and acceleration.

        merges and concatenates to one dataframe (per subject, session & block).
        '''
        sa, ev, m = self.load_pupil(directory=directory)
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

    def interpol(self, source, pupil='orig', margin=100):
        convolved = np.convolve(self.pupil_frame[source], [0.5, 1], 'same')
        ev_start = np.where(convolved == .5)[0]
        ev_end = np.where(convolved == 1)[0]
        if convolved[len(convolved) - 1] > 0:
            ev_end = np.append(ev_end, len(self.pupil_frame) - 1)
        if pupil == 'orig':
            pupil_interpolated = np.array(self.pupil_frame['pa_{}'.format(self.pupilside)].copy())
        elif pupil == 'interpol':
            pupil_interpolated = np.array(self.pupil_frame['interpol'].copy())
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
        return pupil_interpolated

    def blink_interpol(self):
        pupil_interpolated = self.interpol(source='blink')
        self.pupil_frame['interpol'] = pupil_interpolated

    def man_deblink(self):
        '''
        Adds column with all methods of artifact detection enbaled.

        True means 'is an artifact'
        '''
        def plot(column, shift=1):
            f, ax = plt.subplots(figsize=(250, 5))
            ax.plot(self.pupil_frame[column].values * shift)
            ax.xaxis.set_major_locator(plt.MultipleLocator(5000))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(1000))
            ax.grid(which='minor')
            plt.show()
        plot('interpol')
        dirty = []
        start = 0
        while start != 123000:
            start = float(input('start ')) * 1000
            end = float(input('end ')) * 1000
            dirty.append(np.arange(start, end))
        dirty = np.hstack(dirty)
        self.pupil_frame['man_deblink'] = False
        self.pupil_frame.loc[dirty, 'man_deblink'] = True
        interpolated = self.interpol(source='man_deblink', pupil='interpol')
        self.pupil_frame['man_deblink'] = interpolated
        plot('interpol')
        plot('man_deblink')
        valid = input('valid? yes/no ')
        if valid == 'yes':
            self.pupil_frame['interpol'] = interpolated

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


if __name__ == '__main__':
    for sub in [22]:
        for ses in [2, 3]:
            for ri in [0, 1, 2]:
                try:
                    flex_dir = '/Volumes/flxrl/FLEXRULE/'
                    out_dir = join(flex_dir, 'pupil', 'linear_pupilframes')
                    slu.mkdir_p(out_dir)
                    p = Pupilframe(sub, ses, ri, flex_dir)
                    p.basicframe()
                    p.gaze_angle()
                    p.all_artifacts()
                    p.small_fragments()
                    p.interpol()
                    p.filter()
                    p.z_score()
                    p.pupil_frame.to_csv(join(out_dir, 'pupilframe_{}_{}_{}.csv'.format(p.subject, p.session, p.run)))
                except RuntimeError:
                    continue
