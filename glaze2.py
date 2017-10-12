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
    '''loads the matlab file and creates np.array from it

    input_file is path/name of matlab file, key is key in input_file.keys()
    '''
    log = log["p"]['out'][0][0][0, 0][0]
    pl = [row2dict(log[i, 0]) for i in range(log.shape[0])]

    df = pd.DataFrame(pl)
    df.loc[:, 'message'] = df.message.astype('str')
    df.loc[:, 'value'] = df.message.astype('float')
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


def belief(loc, H, e1=0.5, e2=-0.5, sigma=1):
    '''
    Returns models Belief at a given time.

    loc is a pandas Series that indexes into the overall
    log file
    '''
    belief = 0 * loc.values
    for i, value in enumerate(loc):
        if i == 0:
            belief[i] = LLR(value)
        else:
            belief[i] = prior(belief[i - 1], H) + LLR(value)
    return pd.Series(belief, index=loc.index)


def cross_entropy_error(H, choices, point_locations, belief_indices):
    '''
    Compute cross entropy error
    '''

    pnm = belief(point_locations, H).loc[belief_indices].values
    pnm = expit(pnm)
    pn = -choices.values + 1
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

    error_function = lambda x: cross_entropy_error(
        x, choices, point_locations, belief_indices)
    o = opt.minimize_scalar(error_function,
                            bounds=(0, 1), method='bounded')
    return o


# 4.PLOTTING

# 4.1 Plot cross entropy error on hazard rate
def plot_ce_H(DataFrame, x):
    '''returns simple plot with cross entropy errors plotted on Hazard rates.

    x is the step interval made.'''
    t = np.arange(0.00001, 1.0, x)

    def ce_errors(x):
        result = []
        for i in x:
            result.append(ce_error(DataFrame, (i)))
        return result
    s = ce_errors(t)
    plt.plot(t, s)
    plt.xlabel('Hazard rate (H)')
    plt.ylabel('cross entropy error')
    plt.title('cross-entropy error')
    plt.grid(True)
    plt.show()


__version__ = '2.1.2'
'''
2.1.0
Version sperated loading functions from calculating functions and extracting
functions.
Now matlab file has to be loaded and converted to dataframe in a first step.
All other functions take this dataframe as an input.
Thus, functions dont have to access matlab file and are supposedly faster
2.1.1
minor fix to filter_dec function
2.2.2
fix to model_choice function
H now obligatory parameter for model_choice
'''
