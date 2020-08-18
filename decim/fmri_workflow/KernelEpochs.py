import pandas as pd
import numpy as np
from collections import defaultdict
from joblib import Memory
from os.path import expanduser
from decim.adjuvant import slurm_submit as slu
if expanduser('~') == '/home/faty014':
    cachedir = expanduser('/work/faty014/joblib_cache')
else:
    cachedir = expanduser('~/joblib_cache')
slu.mkdir_p(cachedir)
memory = Memory(cachedir=cachedir, verbose=0)

'''
Extract pupil, behavioral and ROI time series data per choice epoch and build a gigantic pd.MultiIndex DataFrame

1. Extract behavioral parameters per choice trial ("choice_behavior")
    a) response, rule_response, stimulus, reward, accumulated belief
2. Extract LLR, CPP and psi of n (default=12) last samples.

Necessary Input files:
    - preprocessed behavioral pd.DAtaFrame
'''


class Choiceframe(object):

    def __init__(self, subject, session, run, task,
                 flex_dir, BehavFrame):
        '''
        Initialize

        - Arguments:
            a) subject
            b) session
            c) run
            d) task
            e) Flexrule directory
            f) behavioral pd.DataFrame
            g) pupil pd.DataFrame
            e) extracted brainstem ROI time series
        '''
        self.subject = subject
        self.session = session
        self.run = run
        self.task = task
        self.flex_dir = flex_dir
        BehavFrame.onset = BehavFrame.onset.astype(float)
        BehavFrame = BehavFrame.sort_values(by='onset')
        self.BehavFrame = BehavFrame
        self.n_samples = 15
        self.parameters = ['LLR', 'surprise', 'psi']
        self.kernels = defaultdict()

    def choice_behavior(self):
        df = self.BehavFrame
        choices = pd.DataFrame({'rule_response': df.loc[df.event == 'CHOICE_TRIAL_RESP', 'rule_resp'].values.astype(float),
                                'rt': df.loc[df.event == 'CHOICE_TRIAL_RESP'].rt.values.astype(float),
                                'stimulus': df.stimulus.dropna(how='any').values,
                                'response': df.loc[df.event == 'CHOICE_TRIAL_RESP', 'value'].values.astype(float),
                                'reward': df.loc[df.event == 'CHOICE_TRIAL_RESP'].reward.values.astype(float),
                                'onset': df.loc[df.event == 'CHOICE_TRIAL_ONSET'].onset.values.astype(float)})
        if self.task == 'inference':
            df.belief = df.belief.fillna(method='ffill')
            choices['trial_id'] =\
                df.loc[df.event == 'CHOICE_TRIAL_ONSET'].trial_id.values.astype(int)
            choices['accumulated_belief'] =\
                df.loc[df.event == 'CHOICE_TRIAL_ONSET'].belief.values.astype(float)
            choices['rewarded_rule'] =\
                df.loc[df.event == 'CHOICE_TRIAL_ONSET'].gen_side.values + 0.5
        self.choices = choices

    def kernel_samples(self, parameter, log=False, zs=False):
        '''
        Add last n points before choice onset.
        '''
        df = self.BehavFrame
        points = df.loc[(df.event == 'GL_TRIAL_LOCATION')]
        if log is True:
            points.loc[1:, parameter] = np.log(points.loc[1:, parameter])                   # first surprise is 0 --> inf introduced
        if zs is True:
            points[parameter] = (points[parameter] - points[parameter].mean()) / points[parameter].std()
        p = []
        for i, row in self.choices.iterrows():
            trial_points = points.loc[points.onset.astype('float') < row.onset]
            if len(trial_points) < self.n_samples:
                trial_points = np.full(self.n_samples, np.nan)
            else:
                trial_points = trial_points[parameter].values[len(trial_points) - self.n_samples:len(trial_points)]
            p.append(trial_points)
        points = pd.DataFrame(p)
        points['trial_id'] = self.choices.trial_id.values
        self.kernels[parameter] = points

    def merge(self):
        '''
        Merge everything into one pd.MultiIndex pd.DataFrame.
        '''
        self.choices.columns =\
            pd.MultiIndex.from_product([['behavior'], ['parameters'],
                                        self.choices.columns],
                                       names=['source', 'type', 'name'])
        for p in self.parameters:
            self.kernels[p].columns =\
                pd.MultiIndex.from_product([['behavior'], [p],
                                            list(range(self.kernels[p].shape[1] - 1)) + ['trial_id']],
                                           names=['source', 'type', 'name'])

        master = pd.concat([self.kernels[p] for p in self.parameters] + [self.choices], axis=1)
        self.master = master.set_index([master.behavior.parameters.trial_id])


#@memory.cache
def execute(subject, session, run, task,
            flex_dir, BehavFrame):
    '''
    Execute per subject, session, task and run.

    Moreover need to give
        - Flexrule directory
        - preprocessed behavioral pd.DAtaFrame
        - preprocessed pupil pd.DataFrame
        - extracted brainstem ROI time series pd.DataFrame
    '''
    c = Choiceframe(subject, session, run, task,
                    flex_dir, BehavFrame)
    c.choice_behavior()
    c.kernel_samples(parameter='LLR')
    c.kernel_samples(parameter='psi', zs=True)
    c.kernel_samples(parameter='surprise', zs=True, log=True)
    c.merge()
    print(c.master.behavior.surprise)
    return c.master


__version__ = '2.0'
'''
2.0
-Input linear pupilframes
-recquires BIDS
1.2
-triallocked period now 1000ms before offset and total of 4500ms
-if rt > 2000ms choicelocked is set to np.nan
'''
#execute('sub-17', 'ses-2', 'inference_run-4', 'inference', 'flex_dir', pd.read_hdf('/Users/kenohagena/Desktop/BehavFrame_sub-17_ses-2.hdf', key='inference_run-4'))
