import numpy as np
import pandas as pd
from glob import glob
from os.path import join, expanduser
from pyedfread import edf
from scipy import signal
import matplotlib.pyplot as plt
from joblib import Memory
from decim.adjuvant import slurm_submit as slu
if expanduser('~') == '/home/faty014':
    cachedir = expanduser('/work/faty014/joblib_cache')
else:
    cachedir = expanduser('~/joblib_cache')
slu.mkdir_p(cachedir)
memory = Memory(cachedir=cachedir, verbose=0)


'''
This script loads and preprocesses pupil data per subject, session and run:

1. Load pupil metrics, events, messages from raw data
2. Do algorithmic blink detection and linearly interpolate
3. Optioannly do manual blink detection in interactive jupyter notebook (highly recommended)
4. band pass filter
4. z_score
6. Return in the following format:
    - rows: time (ms)
    - columns:
        a) pa_left / pa_right (raw pupil diameter)
        b) gxright, gyright (gaze coordinates)
        c) saccade, blink (True, False)
        d) message, message value
        e) trial_id
        f) interpol (blinks interpolated by algorithm)
        g) man_deblink (blinks interpolated manually)
        h) bp_interpol (bandpassed, interpolated pupil)
        i) biz (z-scored, bandpassed, interpolated pupil)
'''


class PupilFrame(object):

    def __init__(self, subject, session, run, flex_dir):
        self.subject = subject
        self.session = session
        self.run = run
        self.type = run[:-6]
        self.flex_dir = flex_dir

    def load_pupil(self):
        '''
        Load pupil, events, messages from raw directory.
        '''
        directory = join(self.flex_dir, 'raw', 'bids_mr_v1.2',
                         self.subject, self.session, 'func')
        files = glob(join(directory, '*{}*.edf'.format(self.run)))
        if len(files) > 1:
            raise RuntimeError(
                'More than one log file found for this block: %s' % files)
        elif len(files) == 0:
            raise RuntimeError(
                'No log file found for this block: %s, %s, %s' %
                (self.subject, self.session, self.run))
        if self.type == 'inference':
            return edf.pread(files[0], trial_marker=b'trial_id')                # 'inference' and 'instructed' run files use different trial_markers
        elif self.type == 'instructed':
            return edf.pread(files[0], trial_marker=b'TRIALID')

    def get_pupil(self, sa):
        '''
        Extracts the pupil diameter and gaze coordinates from the pupil file

        Argument:
            a) sa, pupil file loaded in "load_pupil"
        '''
        assert (sa.mean().pa_right > 0) or (sa.mean().pa_left > 0)
        if (sa.mean().pa_right < 0) & (sa.mean().pa_left > 0):
            pupilside = 'left'
        else:
            pupilside = 'right'

        sa_frame = sa.loc[:, ['pa_{}'.format(pupilside), 'time',
                              'gx_{}'.format(pupilside), 'gy_{}'.format(pupilside)]]

        return sa_frame

    def get_events(self, ev):
        '''
        Loads automatically detected blinks and saccades from events file
        Returns np.arrays with timepoints of blinked/saccaded samples

        Argument:
            a) ev, events file loaded in "load_pupil"
        '''
        sac_frame = ev[ev.type == 'saccade'].loc[:, ['start', 'end']]
        sactime = []
        for i, row in sac_frame.iterrows():
            sactime.append(list(range(row.start, row.end)))
        sactime = sum(sactime, [])                                              # flatten the nested list
        blink_frame = ev[ev.blink == True].loc[:, ['start', 'end']]
        blinktime = []
        for i, row in blink_frame.iterrows():
            blinktime.append(list(range(row.start, row.end)))
        blinktime = sum(blinktime, [])
        return blinktime, sactime

    def get_messages(self, m):
        '''
        Reduces message file to relevant information
        Choice grating onset, offset, response and RT.

        Argument:
            a) m, message file loaded in "load_pupil"
        '''
        mess = ['CHOICE_TRIAL_ONSET',
                'CHOICE_TRIAL_RESP',
                'CHOICE_TRIAL_STIMOFF', 'RT']
        messages = mess + [i + '_time' for i in mess] + ['trialid ']
        message_frame = m.loc[:, messages]

        columns = {}
        for message_label in mess:
            columns[message_label] = message_frame.loc[:, [message_label,
                                                           message_label +
                                                           '_time']].\
                dropna(how='all')
            columns[message_label] = columns[message_label].\
                rename(columns={message_label + '_time': 'time',
                                message_label: 'message_value'})
            columns[message_label]['message'] = message_label
            trial_id = [int(i[9:13]) for i in message_frame.
                        loc[~message_frame[message_label].isnull(),
                            'trialid '].values]
            if len(trial_id) != len(columns[message_label]):
                continue
            else:
                columns[message_label]['trial_id'] = trial_id
        return pd.concat(list(columns.values())).sort_values(by='time').\
            fillna(method='ffill', limit=1)

    def basicframe(self, events=True, messages=True):
        '''
        1. Loads pupil data and optionally events & messages.
        2. Merges and concatenates into one pd.DataFrame
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

    def small_fragments(self, array, threshhold=200):
        '''
        Detects leftover fragments smaller than threshhold.

        Sets those detected fragments to NaN to make linear interpolation cleaner.
        '''
        convolved = np.convolve(array, [0.5, 1], 'same')
        ev_start = np.where(convolved == .5)[0]
        fragment_ends = ev_start
        if convolved[0] != 0:
            fragment_ends = fragment_ends[1:len(fragment_ends)]
        if convolved[len(convolved) - 1] == 0:
            fragment_ends = np.append(fragment_ends, len(array))
        ev_end = np.where(convolved == 1)[0]
        if convolved[0] == 0:
            fragment_starts = np.append(0, ev_end)
        else:
            fragment_starts = ev_end
        if (convolved[-2] == 1.5) & (convolved[-1] == 1):
            fragment_starts = fragment_starts[0:-1]
        assert len(fragment_ends) == len(fragment_starts)
        fragment_length = fragment_ends - fragment_starts
        wh = np.where(fragment_length < threshhold)
        smallfrag_ends = fragment_ends[wh]
        smallfrag_starts = fragment_starts[wh]
        for start, end in zip(smallfrag_starts, smallfrag_ends):
            array[start:end + 1] = True
        return array

    def interpol(self, array, orig_pupil, margin=250):
        '''
        Linear interpolation of blinks and artifacts.

        - Arguments:
            a) array (pupil with blinks set to NaN)
            b) orig_pupil (origninal pupil)
            c) margin used at star/end of interpolation
        '''
        convolved = np.convolve(array, [0.5, 1], 'same')                        # Use convolution to detect start and endpoint of interpolation
        ev_start = np.where(convolved == .5)[0]
        ev_end = np.where(convolved == 1)[0]
        if convolved[len(convolved) - 1] > 0:
            ev_end = np.append(ev_end, len(self.pupil_frame) - 1)
        pupil_interpolated = np.array(orig_pupil.copy())                        # Copy of original pupil to interpolate
        for b in range(len(ev_start)):
            if ev_start[b] < margin:
                start = 0
            else:
                start = ev_start[b] - margin + 1
            if ev_end[b] + margin + 1 > len(self.pupil_frame) - 1:
                end = len(self.pupil_frame) - 1
            else:
                end = ev_end[b] + margin + 1
            interpolated_signal = np.linspace(pupil_interpolated[start],        # Inteprolate
                                              pupil_interpolated[end],
                                              end - start,
                                              endpoint=False)
            pupil_interpolated[start:end] = interpolated_signal
        return pupil_interpolated

    def blink_interpol(self, crit_frags=200):
        '''
        Do automatic blink detection and interpolation

        -Argument:
            a) length of small fragments to be interpolated
        '''

        array = self.pupil_frame['pa_{}'.format(self.pupilside)] < 100          # detects blinks by absolute pupil diameter threshold
        array = self.small_fragments(array, crit_frags=crit_frags)              # moreover, sets small fragments to NaN
        pupil_interpld =\
            self.interpol(array=array,
                          pupil=self.pupil_frame['pa_{}'.
                                                 format(self.pupilside)],
                          margin=250)                                           # interpolate detected blinks and small fragments linearly
        self.pupil_frame['interpol'] = pupil_interpld

    def man_deblink(self):
        '''
        Function allows in interactive environment to manually report
        - samples that have been false negatively not reported as artifacts/blinks
        - samples that have been false positively been reported as artifacts/blinks

        Returns pd.DataFrame with
            - new column 'man_deblink'
            - updated column 'blink', which now tags every interpolated sample with True
        '''
        pf = self.pupil_frame.copy()

        def plot(compare=False):
            f, ax = plt.subplots(figsize=(250, 5))
            if compare is False:
                ax.plot(pf['pa_{}'.format(self.pupilside)].values - 500)
                ax.plot(pf['interpol'].values)
            else:
                ax.plot(pf['pa_{}'.format(self.pupilside)].values)
                ax.plot(pf['man_deblink'].values + 500)
            ax.xaxis.set_major_locator(plt.MultipleLocator(5000))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(1000))
            ax.grid(which='minor')
            plt.show()
        plot()
        valid = 'no'
        while valid == 'no':
            dirty = []
            clean = []
            start = 0
            while start != 123000:                                              # Always input start and end of artifact segment
                start = float(input('start ')) * 1000                           # Terminate with input "123"
                end = float(input('end ')) * 1000
                dirty.append(np.arange(start, end))
            while start != 321000:                                              # Input start and end of false positive segements (automatically interpolated despite actually ok)
                start = float(input('start ')) * 1000                           # Terminate with input "321"
                end = float(input('end ')) * 1000
                clean.append(np.arange(start, end))
            dirty = np.hstack(dirty)
            clean = np.hstack(clean)
            pf['blink'] = pf['pa_{}'.format(self.pupilside)] < 100
            pf.loc[dirty, 'blink'] = True
            pf.loc[clean, 'blink'] = False
            interpolated = self.interpol(source=pf['blink'],
                                         pupil=pf['pa_{}'.
                                                  format(self.pupilside)])      # interpolate
            pf['man_deblink'] = interpolated
            plot(compare=True)
            valid = input('valid? yes/no ')
        if valid == 'yes':                                                      # Terminate manual interface by typing 'yes'
            self.pupil_frame = pf                                               # Initiate second manual interface by typing "no"
            self.pupil_frame['interpol'] = interpolated

    def filter(self, highpass=.01, lowpass=6, sample_rate=1000):
        '''
        Apply 3rd-order Butterworth bandpass filter.
        '''
        pupil_interpolated = self.pupil_frame.interpol                          # High pass:
        hp_cof_sample = highpass / (sample_rate / 2)
        bhp, ahp = signal.butter(3, hp_cof_sample, btype='high')
        pupil_interpolated_hp = signal.filtfilt(bhp, ahp, pupil_interpolated)
        lp_cof_sample = lowpass / (sample_rate / 2)                             # low pass
        blp, alp = signal.butter(3, lp_cof_sample)
        pupil_interpolated_bp = signal.filtfilt(blp, alp,
                                                pupil_interpolated_hp)          # band pass

        self.pupil_frame['bp_interpol'] = pupil_interpolated_bp

    def z_score(self):
        '''
        Normalize
        '''
        self.pupil_frame['biz'] = (self.pupil_frame.bp_interpol -
                                   self.pupil_frame.bp_interpol.mean()) /\
            self.pupil_frame.bp_interpol.std()


#@memory.cache
def execute(subject, session, run, flex_dir, manual=True):
    '''
    Execute this script.

    - Arguments;
        a) subject (e.g. 'sub-17')
        b) session (e.g. 'ses-2')
        c) run (e.g. 'inference_run-4')
        e) Flexrule directory
        f) manual (True recommended)

    - Output: pd.DataFrame with pupil
    '''
    pf = PupilFrame(subject, session, run, flex_dir)
    pf.basicframe()
    pf.blink_interpol()
    if manual is True:
        pf.man_deblink()
    pf.filter()
    pf.z_score()
    return pf.pupil_frame
