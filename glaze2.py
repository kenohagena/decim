# IMPORT STUFF
import numpy as np
import pandas as pd

from glob import glob
from os.path import join

from scipy import optimize as opt
from scipy.io import loadmat
from scipy.stats import norm
from scipy.special import expit  # logistic function

# 1. LOAD MATLAB FILE AND EXTRACT RELEVANT DATA IN ARRAY OR DATAFRAME


def load_log(sub_code, session, phase, block, base_path):
    """
    Concatenates path and file name and loads matlba file.

    Recquires subject code, session, phase and block.
    """
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
    """
    Convert a single row of the log to a dictionary.
    """
    return {'time': item[0, 0][0, 0],
            'message': item[0, 1][0],
            'value': item[0, 2].ravel()[0],
            'phase': item[0, 3][0, 0],
            'block': item[0, 4][0, 0]}


def log2pd(log, block, key="p"):
    """
    Takes loaded matlab log and returns panda dataframe.

    Extracts only logs of the current block.
    """
    log = log["p"]['out'][0][0][0, 0][0]
    pl = [row2dict(log[i, 0]) for i in range(log.shape[0])]

    df = pd.DataFrame(pl)
    df.loc[:, 'message'] = df.message.astype('str')
    df.loc[:, 'value'] = df.value
    return df.query('block==%i' % block)


# CALCULATIONS


def LLR(value, e1=0.5, e2=-0.5, sigma=1):
    """
    Computes log likelihood ratio.

    Takes means of two distributions and the common variance sigma.
    """
    LLR = np.log(norm.pdf(value, e1, sigma)) - \
        np.log(norm.pdf(value, e2, sigma))
    return LLR


def prior(b_prior, H):
    """
    Returns weighted prior belief given blief at time t-1 and hazard rate
    """
    psi = b_prior + np.log((1 - H) / H + np.exp(-b_prior)) - \
        np.log((1 - H) / H + np.exp(b_prior))
    return psi


def belief(df, H, gen_var=1):
    """
    Returns models belief at a given time.

    Takes panda dataframe and a hazardrate.
    """
    locs = (df.loc[df.message == "GL_TRIAL_LOCATION", 'value']
            .astype(float))
    belief = 0 * locs.values
    for i, value in enumerate(locs):
        if i == 0:
            belief[i] = LLR(value) * gen_var
        else:
            belief[i] = prior(belief[i - 1], H) + LLR(value) * gen_var
    return pd.Series(belief, index=locs.index)


def cross_entropy_error(df, H):
    """
    Computes cross entropy error, given dataframe and hazardrate.
    """

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
    """
    Returns hazardrate with lowest cross entropy error.

    Takes dataframe and uses simple scalar optimization algorithm.
    """
    def error_function(x):
        return cross_entropy_error(df, x)
    o = opt.minimize_scalar(error_function,
                            bounds=(0, 1), method='bounded')
    return o


def performance(sub_code, session, phase, block, base_path):
    """
    returns dictionary containing number of decisions, number of NaNs and count of rewards.
    """
    df = log2pd(load_log(sub_code, session, phase, block, base_path), block)
    rews = (df.loc[df.message == "GL_TRIAL_REWARD", 'value'])
    array = np.array(rews.values).astype(float)
    no_answer = np.count_nonzero(np.isnan(array))
    rewards = np.count_nonzero((array)) - np.count_nonzero(np.isnan(array))
    performance = rewards / len(array)
    return {'no_answer': no_answer, 'rewards': rewards, 'decisions': len(array), 'performance': performance}


def mean_rt(sub_code, session, phase, block, base_path):
    """
    Returns mean reaction time of given block.
    """
    df = log2pd(load_log(sub_code, session, phase, block, base_path), block)
    rt = df.loc[df.message == 'CHOICE_TRIAL_RT']['value']
    return rt.mean()


def acc_ev(sub_code, session, phase, block, base_path, H):
    '''
    takes sub, ses, etc. and returns accumulated evidence at decision points and reaction time.

    Accumulated evidence corresponds to the belief strength of the glaze model.
    '''
    df = log2pd(load_log(sub_code, session, phase, block, base_path), block)
    # drop these, because sometimes missing or duplicated

    df = df.loc[df.message != 'BUTTON_PRESS']
    df = df.reset_index()
    choices = (df.loc[df.message == "CHOICE_TRIAL_RULE_RESP", 'value']
               .astype(float))
    belief_indices = df.iloc[choices.index - 11].index.values
    rt = df.loc[df.message == 'CHOICE_TRIAL_RT']['value']
    accum_ev = belief(df, H).loc[belief_indices].values
    obj_accum_ev = belief(df, 1 / 70).loc[belief_indices].values
    decision = df.loc[df.message == 'CHOICE_TRIAL_RULE_RESP']['value']

    try:
        assert len(rt) == len(accum_ev) == len(decision)
    except AssertionError:
        print('''Length of rt, accumulated evidence and/or decisions did not match in block {0},
            session {1} of subject {2}
            '''.format(block, session, sub_code)
              )
    return pd.DataFrame({'reaction_time': rt.values, 'accumulated_evidence': accum_ev,
                         'decision': decision.values, 'objective_accumulated_evidence': obj_accum_ev})


__version__ = '3.3.2'
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
3.2.1
changes for better readability
3.3
performance function
3.3.1
mean RT
3.3.2
-accumulated evidence: now contains decision values in order to calculate choice consistency
-accumulated evidence: contains 'objective evidence', i.e. evidence of model when given true H
'''


#print(acc_ev('VPIM01', 'B', 3, 4, '/users/kenohagena/documents/immuno/data/vaccine', 1 / 20))
