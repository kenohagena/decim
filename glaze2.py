# IMPORT STUFF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glob import glob
from os.path import join

from scipy import optimize as opt
from scipy.io import loadmat
from scipy.stats import norm
from scipy.special import expit  # logistic function

# 1. LOAD MATLAB FILE AND EXTRACT RELEVANT DATA IN ARRAY OR DATAFRAME

# 1.1 Preparation: Get path/file-name from subjectcode, etc.


def load_log(sub_code, session, phase, block, base_path):
    '''
    returns the correct path + file of the matlab file in a specific folder.

    Input: subjects code, session A/B or C, phase 1 or 2, block 1 - 7.
    '''
    directory = join(base_path,
                     "{}".format(sub_code),
                     "{}".format(session),
                     'PH_' + "{}".format(phase) +
                     'PH_' + "{}".format(block))
    files = glob(join(directory, '*.mat'))
    if len(files) > 1:
        raise RuntimeError(
            'More than one log file found for this block: %s' % files)
    elif len(files) == 0:
        raise RuntimeError(
            'No log file found for this block: %s, %s, %s, %s' %
            (sub_code, session, phase, block))
    return loadmat(files[0])


def row2dict(item):
    '''
    Convert a single row of the log to a dictionary.
    '''
    return {'time': item[0, 0][0, 0],
            'message': item[0, 1][0],
            'value': item[0, 2].ravel()[0],
            'phase': item[0, 3][0, 0],
            'block': item[0, 4][0, 0]}


def log2pd(log, block, key="p"):
    '''takes loaded matlab file and returns panda dataframe.

    contains only logs of that block.
    '''
    log = log["p"]['out'][0][0][0, 0][0]
    pl = [row2dict(log[i, 0]) for i in range(log.shape[0])]

    df = pd.DataFrame(pl)
    df.loc[:, 'message'] = df.message.astype('str')
    df.loc[:, 'value'] = df.value
    return df.query('block==%i' % block)


def LLR(value, e1=0.5, e2=-0.5, sigma=1):
    '''
    returns log likelihood ratio.

    Needs means of both distributions and variance and a given value.
    '''
    LLR = np.log(norm.pdf(value, e1, sigma)) - \
        np.log(norm.pdf(value, e2, sigma))
    return LLR


def prior(b_prior, H):
    '''
    returns weighted prior belief.

    given belief at t = n-1 and hazard rate.
    '''
    psi = b_prior + np.log((1 - H) / H + np.exp(-b_prior)) - \
        np.log((1 - H) / H + np.exp(b_prior))
    return psi


def belief(df, H):
    '''
    Returns models Belief at a given time.

    loc is a pandas Series that indexes into the overall
    log file
    '''
    locs = (df.loc[df.message == "GL_TRIAL_LOCATION", 'value']
            .astype(float))
    belief = 0 * locs.values
    for i, value in enumerate(locs):
        if i == 0:
            belief[i] = LLR(value)
        else:
            belief[i] = prior(belief[i - 1], H) + LLR(value)
    return pd.Series(belief, index=locs.index)


def cross_entropy_error(df, H):
    '''
    Compute cross entropy error
    '''

    choices = (df.loc[df.message == "CHOICE_TRIAL_RULE_RESP", 'value']
               .astype(float))
    choices = choices.dropna()

    # Find last point location before choice trial
    belief_indices = df.loc[choices.index - 12].index.values
    pnm = -belief(df, H).loc[belief_indices].values
    pnm = expit(pnm)
    pn = choices.values
    return -np.sum(((1 - pn) * np.log(1 - pnm)) + (pn * np.log(pnm)))


def optimal_H(df):
    '''
    returns hazard rate with best cross entropy error.

    uses simple scalar optimization algorithm. time recquired: 50s.
    '''
    point_locations = (df.loc[df.message == "GL_TRIAL_LOCATION", 'value']
                       .astype(float))
    choices = (df.loc[df.message == "CHOICE_TRIAL_RULE_RESP", 'value']
               .astype(float))
    choices = choices.dropna()

    # Find last point location before choice trial
    belief_indices = df.loc[choices.index - 12].index.values

    def error_function(x): return cross_entropy_error(
        df, x)
    o = opt.minimize_scalar(error_function,
                            bounds=(0, 1), method='bounded')
    return o


__version__ = '3.2'
'''
2.1.0
Version sperated loading functions from calculating functions and extracting
functions.
Now matlab file has to be loaded and converted to dataframe in a first step.
All other functions take this dataframe as an input.
Thus, functions dont have to access matlab file and are supposedly faster
2.1.1
minor fix to filter_dec function
2.1.2
fix to model_choice function
H now obligatory parameter for model_choice
3.0 shortening by N
3.1
fix in log2pd: deleted the operation astype(float) on values, because ValueError was raised on subjects answers ('x' or 'm')
in cross-entropy-error: inserted the definitions of choices and belief_indices from optimal_H
3.2
deleted obsolete plotting section
'''

