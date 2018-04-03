import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from scipy.io import loadmat
import glaze2 as gl
import math

sessions = {1: 'A', 2: 'B', 3: 'C'}


class Nassarframe(object):
    '''
    PRD_TRIAL_START   ------> trial start, value = value of sample to come (identical with show sample)
    PRD_PREDICT_SAMPLE_ON  -> value of last sample (new value, than prediction. this is the X)
    PRD_PRED  --------------> new prediction
    PRD_ERR  ---------------> difference between prediction and last seen value (thus == "PRD_PREDICT_SAMPLE_ON" - prediction)
    PRD_SHOW_SAMPLE  -------> show trial_start value
    PRD_SHOW_FIXERR  -------> fixation error, 0 == no, 1 == yes (hash)
    PRD_SHOW_OFF  ----------? 0 or 1; seems to be identical to fixation error
    PRD_ERRORAT  -----------> prediction error made (thus == "PRD_PRED" - sample)
    '''

    def __init__(self, subject, session, path, blocks=[1, 2, 3, 4, 5, 6, 7]):
        '''
        Initialize.
        '''
        self.subject = 'VPIM0{}'.format(subject)
        self.sub = subject
        self.session = sessions[session]
        self.ses = session
        self.blocks = blocks
        self.path = path
        self.phase = session * 2
        self.trials = {}

    def get_trials(self):
        '''
        Load data and extract samples ('X'), preditcions and blink info.
        '''
        for block in self.blocks:
            source = gl.log2pd(gl.load_log(self.subject, self.session, self.phase, block, self.path), block)
            trials = pd.DataFrame({
                'prediction': source.loc[source.message == 'PRD_PRED'].value.values,
                'X': source.loc[source.message == 'PRD_SHOW_SAMPLE'].value.values,
                'fixation_err': source.loc[source.message == 'PRD_SHOW_FIXERR'].value.values})
            trials['new_prediction'] = np.roll(trials.prediction, -1)
            trials['prediction_error'] = trials.X - trials.prediction
            trials['learning_rate'] = (trials.new_prediction - trials.prediction) / trials.prediction_error
            trials.loc[len(trials) - 1, 'learning_rate'] = np.nan
            self.trials[block] = trials.loc[:, ['prediction',
                                                'X',
                                                'fixation_err',
                                                'prediction_error',
                                                'learning_rate']]

    def concat(self):
        '''
        concatenate blocks
        '''
        df = []
        for key, value in self.trials.items():
            value['block'] = key
            df.append(value)
        df = pd.concat(df, ignore_index=True)
        self.sessiontrials = df


def load_sequences(subject, session, block, sequencefile_path):
    sequences_mat = loadmat(sequencefile_path)
    sessions = ['A', 'B', 'C']
    subjects = ['VPIM01', 'VPIM02', 'VPIM03', 'VPIM04', 'VPIM05', 'VPIM06', 'VPIM07', 'VPIM08', 'VPIM09']
    if subject == 'VPIM01':
        seq == sessions.index(session)

    elif subject in subjects[1:4] and session == 'A':
        seq = subjects.index(subject) + 2
    else:
        seq = subjects.index(subject)
    print(seq, sessions.index(session) * 2 + 1, block - 1)
    return sequences_mat["sequences"][0, seq][0, (sessions.index(session) * 2 + 1)][0, block - 1]['mu'][0][0][0]

#n = Nassarframe(2, 2, '/Users/kenohagena/Documents/immuno/data/vaccine')
# n.get_trials()
# n.concat()
# print(n.sessiontrials)
