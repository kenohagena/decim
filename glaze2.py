# IMPORT STUFF

import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from scipy import optimize as opt
from scipy.io import loadmat
from scipy.stats import norm

# 1. LOAD MATLAB FILE AND EXTRACT RELEVANT DATA IN ARRAY OR DATAFRAME

# 1.1 Preparation: Get path/file-name from subjectcode, etc.


def path_name(sub_code, session, phase, block):
    '''returns the correct path + file of the matlab file in a specific folder.

    Input: subjects code, session A/B or C, phase 1 or 2, block 1 - 7.'''
    directory = os.path.join('/Users/kenohagena/Documents/Data/',
                             "{}".format(sub_code),
                             "{}".format(session), 'PH_' +
                             "{}".format(phase) + 'PH_'
                             + "{}".format(block))
    for file in os.listdir(directory):
        if file.endswith(".mat"):
            return os.path.join(directory, file)

# Get path/file-name of previous block


def former_path_name(sub_code, session, phase, block):
    '''returns the correct path + file of the matlab file one block before.

    Input: subjects code, session A/B or C, phase 1 or 2, block 1 - 7.'''
    if block == 1:
        directory = os.path.join('/Users/kenohagena/Documents/Data/',
                                 "{}".format(
                                     sub_code), "{}".format(session),
                                 'PH_' + "{}".format(phase) +
                                 'PH_' + "{}".format(block))
    else:
        directory = os.path.join('/Users/kenohagena/Documents/Data/',
                                 sub_code,
                                 session,
                                 'PH_' +
                                 "{}".format(phase) +
                                 'PH_' +
                                 "{}".format(block - 1))
    for file in os.listdir(directory):
        if file.endswith(".mat"):
            return os.path.join(directory, file)

# 1.2 load matlab file and create an NUMPY ARRAY

# function to load all data from the matlab file


def row2dict(item):
    '''
    Convert a single row of the log to a dictionary.
    '''
    return {'time': item[0, 0][0, 0],
            'message': item[0, 1][0],
            'value': item[0, 2].ravel()[0],
            'block': item[0, 3][0, 0],
            'phase': item[0, 4][0, 0]}


def load_np_unfiltered(sub_code, session, phase, block, key="p"):
    '''loads the matlab file and creates np.array from it

    input_file is path/name of matlab file, key is key in input_file.keys()'''
    input_file = path_name(sub_code, session, phase, block)
    x = loadmat(input_file)
    log = x["p"]['out'][0][0][0, 0][0]

    pl = [row2dict(log[i, 0]) for i in range(log.shape[0])]

    return np.array(pl)

# function to load only "new" data from matlab file


def load_np_filtered(sub_code, session, phase, block, key="p"):
    '''loads the matlab file and creates pd.DataFrame from new logs

    input_file is path/name of matlab file, key is key in input_file.keys()'''
    fsh = load_np_unfiltered(sub_code, session, phase, block - 1).shape[0]
    ash = load_np_unfiltered(sub_code, session, phase, block).shape[0]
    input_file = path_name(sub_code, session, phase, block)
    x = loadmat(input_file)
    log = x["p"]['out'][0][0][0, 0][0]

    pl = [row2dict(log[i, 0]) for i in range(log.shape[0])]
    da = np.array(pl)
    return da[fsh:ash]

# function to decide automatically which of the two options above shall be
# applied


def load_np(sub_code, session, phase, block, key="p"):
    '''function loads np and decides whether filtered or unfiltered

    shall be applied.'''
    if block > 1:
        return load_np_filtered(sub_code, session, phase, block, key="p")
    else:
        return load_np_unfiltered(sub_code, session, phase, block, key="p")

# 1.3 load matlab file and create PANDA DATAARRAY

# extract all data


def load_pd_unfiltered(sub_code, session, phase, block, key="p"):
    '''loads the matlab file and creates pd.DataFrame from it

    input_file is path/name of matlab file, key is key in input_file.keys()'''
    input_file = path_name(sub_code, session, phase, block)

    x = loadmat(input_file)
    log = x["p"]['out'][0][0][0, 0][0]

    pl = [row2dict(log[i, 0]) for i in range(log.shape[0])]
    df = pd.DataFrame(pl)
    df.loc[:, 'message'] = df.message.astype('str')
    return df

# load Data filtered for only new  elements


def load_pd_filtered(sub_code, session, phase, block, key="p"):
    '''loads the matlab file and creates pd.DataFrame from new logs

    input_file is path/name of matlab file, key is key in input_file.keys()'''
    fsh = load_np_unfiltered(sub_code, session, phase, block - 1).shape[0]
    ash = load_np_unfiltered(sub_code, session, phase, block).shape[0]
    input_file = path_name(sub_code, session, phase, block)
    x = loadmat(input_file)
    log = x["p"]['out'][0][0][0, 0][0]

    pl = [row2dict(log[i, 0]) for i in range(log.shape[0])]
    df = pd.DataFrame(pl)
    df.loc[:, 'message'] = df.message.astype('str')
    return df[fsh:ash]

# decide function whether all data or just new


def load_pd(sub_code, session, phase, block, key="p"):
    '''function loads np and decides whether filtered or unfiltered

    shall be applied.'''
    if block > 1:
        return load_pd_filtered(sub_code, session, phase, block, key="p")
    else:
        return load_pd_unfiltered(sub_code, session, phase, block, key="p")


# 2. EXTRACT ALL KIND OF STUFF FROM THE DATA

# 2.1 Extract all point locations and theri indices
def p_loc(DataFrame):
    '''returns location values of all points as array.

    array contains value of all point locations.'''
    df = DataFrame
    df.message = df.message.astype(str)
    return np.array(df.query('message=="GL_TRIAL_LOCATION"')
                    .value
                    .astype(float))


def p_loc_ind(DataFrame):
    '''returns indices of point locations as an array.

    array contains index of all point locations.'''
    df = DataFrame
    df.message = df.message.astype(str)
    index1 = np.array(
        df.query('message=="GL_TRIAL_LOCATION"').index.astype(float))
    firstindex = np.array(df[:1].index.astype(float))
    return np.subtract(index1, firstindex)

# 2.2 EXTRACT RELEVANT SUBJECT DATA AT DECISION TRIALS

# extract all answers


def answers(DataFrame):
    '''returns numpy array with all subject answers.

    0 for x and 1 for m.'''
    df = DataFrame
    choice = np.array(
        df.query('message=="CHOICE_TRIAL_RESP"').value.astype(float))
    return choice

# extract answer indexes


def answer_ind(DataFrame):
    '''returns numpy array with all indexes of answers.'''
    df = DataFrame
    choice_ind = np.array(
        df.query('message=="CHOICE_TRIAL_RESP"').index.astype(float))
    return choice_ind

# extract rule response


def rule_resp(DataFrame):
    '''returns onedimensional array containing all rule responses.

    1 corresponds with vertical-left, 0 == vertical right.'''
    df = DataFrame
    rresp = np.array(
        df.query('message=="CHOICE_TRIAL_RULE_RESP"').value.astype(float))
    return rresp


# extract stimulus
def stim(DataFrame):
    '''returns array with all stimuli.

    1 for horiyontal, 0 for vertical.'''
    df = DataFrame
    stimulus = np.array(
        df.query('message=="GL_TRIAL_STIM_ID"').value.astype(float))
    return stimulus


# extract generation side
def dist_source(DataFrame):
    '''returns array with all generating sources.

    -0.5 for left source +0.5 for right source.'''
    df = DataFrame
    da = np.array(df)
    # first all indexes of 'generation side'
    dist_ind = np.array(
        df.query('message=="GL_TRIAL_GENSIDE"').index.astype(float))
    # mask to only contain 'gen side' in decision trial, thus only if
    # 'stim_id' follows
    mask = []
    for n in dist_ind.astype(int):
        if da[(n - 1), 1] == "GL_TRIAL_STIM_ID":
            mask.append(n)
    # all values of 'generation side' in decision trials
    dist = (da[mask, 4].astype(float))
    return dist


# extract rewards
def reward(DataFrame):
    '''returns array with all rewards.'''
    df = DataFrame
    rewards = np.array(
        df.query('message=="GL_TRIAL_REWARD"').value.astype(float))
    return rewards


# 3. CALCULATE STUFF

# 3.1 MODELS CHOICES
def mod_choice(DataFrame, H):
    '''calculates belief of model over all trials.

    returns choices model would have made before decision trial. 
    Hazard rate optional argument.'''
    return filter_dec(belief(DataFrame, H), DataFrame)


# Log likelihood ratio
def LLR(value, e1=0.5, e2=-0.5, sigma=1):
    '''returns log likelihood ratio.

    Needs means of both distributions and variance and a given value.'''
    LLR = np.log(norm.pdf(value, e1, sigma)) - \
        np.log(norm.pdf(value, e2, sigma))
    return LLR

# time varying prior expectation psi


def prior(b_prior, H):
    '''returns weighted prior belief.

    given belief at t = n-1 and hazard rate.'''
    psi = b_prior + np.log((1 - H) / H + np.exp(-b_prior)) - \
        np.log((1 - H) / H + np.exp(b_prior))
    return psi

# belief (as in Glaze et al.)


def belief(DataFrame, H, e1=0.5, e2=-0.5, sigma=1):
    '''Returns models Belief at a given time.

    Needs p_loc(Dataframe), means, variance and hazard rate.'''
    result = []
    loc = p_loc(DataFrame)
    for i, value in enumerate(loc):
        if i == 0:
            result.append(LLR(value))
        else:
            belief = prior(result[-1], H) + LLR(value)
            result.append(belief)
    return np.array(result)

# function to filter out beliefs at relevant postitions, i.e. before
# decision trials


def filter_dec(x, DataFrame):
    '''filters only relevant values of belief function (above).

    Returns belief values at those timepoints, when a decision trial follows.
    '''
    # mask contains indices of p_loc in ALL TRIALS of p_locs before decision
    # trial (thus only if 'stim_id' follows 4 logs later)
    loc_indices = p_loc_ind(DataFrame)
    df = DataFrame
    da = np.array(df)
    mask = []
    for n in loc_indices.astype(int):
        try:
            if da[(n + 4), 1] == "GL_TRIAL_STIM_ID":
                mask.append(n)
        except IndexError:
            continue

    # mask contains indices of 'p_loc_ind' in extracted beliefs
    maskc = []
    for i in mask:
        maskc.append(np.where(loc_indices == i))
    return np.ravel(x[maskc])
    # return maskc

# 3.2 MODELS SUCCESS

# first mapping model choice to rule response of subject


def map_rresp(x):
    '''
    function maps input to rule response of subject.

    rule response of subject is mapped as follows:
        1 == vertical-left
        0 == vertical righ.
    '''
    return -(x > 0).astype(float) + 1


def mod_suc(DataFrame, H):
    '''
    counts choices in which the model micmicks succesfully
    the subjects answers.

    returns the count as a fraction of all choices made.'''
    model = mod_choice(DataFrame, H)
    sub = rule_resp(DataFrame)
    difference = map_rresp(model) - sub
    return collections.Counter(difference)[0.0] / len(difference)


# 3.3 CROSS ENTROPY ERROR

# logistic regression of absolute models belief values
def log_reg(x):
    return 1 / (1 + np.exp(-x))

# cross-entropy error function
# rresp 0 == verticl right, 1==vertical left


def ce_error_array(DataFrame, H):
    '''returns an array with single cross-entropy errors.'''
    pn = -rule_resp(DataFrame) + 1
    pnm = log_reg(mod_choice(DataFrame, H))
    return((1 - pn) * np.log(1 - pnm)) + (pn * np.log(pnm))

# filters trials with 'nan' as result


def filter_nan(x):
    '''filters only results that are not nan.'''
    result = []
    for i in x:
        if math.isnan(i) is False:
            result.append(i)
        else:
            continue
    return result


def ce_error(DataFrame, H):
    '''
    returns the cross=entropy error for a given input 
    DataFrame dependent on the hazard rate H.
    '''

    return -sum(filter_nan(ce_error_array(DataFrame, H)))


def count_nan(DataFrame):
    '''returns count of 'nan' in decision trials in input file'''
    def get_nan(x):
        result = []
        for i in x:
            if math.isnan(i):
                result.append(i)
            else:
                continue
        return result
    return len(get_nan(ce_error(DataFrame, 1 / 70)))

# 3.4 OPTIMAL HAZARD RATE


def optimal_H(DataFrame):
    '''returns hazard rate with best cross entropy error.

    uses simple scalar optimization algorithm. time recquired: 50s.'''

    o = opt.minimize_scalar(lambda x: ce_error(DataFrame, x),
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
