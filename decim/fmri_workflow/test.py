import numpy as np
import pandas as pd
from glob import glob
from os.path import join, expanduser
from decim.adjuvant import slurm_submit as slu
import nibabel as nib
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from pymeg import parallel as pbs
from matplotlib import pyplot as plt
import seaborn as sns


class SingleTrialGLM(object):
    '''
    Initialize.

    - Arguments:
        a) subject
        b) session
        c) runs (multiple?)
        d) Flexrule directory
        e) BehavDataFrame
        f) task
    '''

    def __init__(self):
        self.df = pd.read_hdf('/home/khagena/FLEXRULE/Workflow/Sublevel_GLM_Climag_2020-04-09/sub-17/trial_regressors_sub-17.hdf', key='ses-2')

    def sanity_check_trial_regressors(self):
        df = self.df
        f, ax = plt.subplots(len(df.columns), 1, figsize=(10, len(df.columns) * .9))
        for i, col in enumerate(df.columns):
            ax[i].plot(df[col].values)
            ax[i].set(xticks=[], yticks=[])
        sns.despine(bottom=True, left=True)
        f.savefig(join(self.out_dir, 'trial_regressors.png'))


def execute(subject, session, runs):
    v = SingleTrialGLM()
    v.sanity_check_trial_regressors()


def submit(sub, env='Climag'):
    for ses in [2, 3]:
        pbs.pmap(execute, [(sub, ses, env)], walltime='4:00:00',
                 memory=40, nodes=1, tasks=2, name='subvert_sub-{}'.format(sub))
