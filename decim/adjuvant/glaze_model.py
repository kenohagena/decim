import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from scipy.stats import norm
import math

'''
This script contains two functions, which are assessed by other scripts.

1. "load_logs_bids"
    - loads behavioral data from the bids data directory
    - returns pd.DataFrame and name of file

2. "belief"
    - takes pd.DataFrame with behavior, Hazard rate and generative variance
    - computes and returns Glaze belief, LLR, psi and Murphy surprise
'''


def load_logs_bids(subject, session, base_path, run='inference'):
    '''
    Loads behavioral data for all runs per session, subject and task
    and returns behavioral pd.DataFrames

    - Arguments:
        a) subject (e.g. 'sub-17')
        b) session (e.g. 'ses-2')
        c) directory of raw dataset in BIDS format
        d) task ('inference' or 'instructed')

    - Output: dictionary with runs as keys and loaded pd.DataFrames as values
    '''
    if session == 'ses-1':
        modality = 'beh'
    else:
        modality = 'func'
    directory = join(base_path,
                     subject,
                     session,
                     modality)
    if run == 'instructed':
        files = sorted(glob(join(directory, '*{}*.csv'.format(run))))
    else:
        files = sorted(glob(join(directory, '*{}*.tsv'.format(run))))
    if len(files) == 0:
        raise RuntimeError(
            'No log file found for this block: %s, %s' %
            (subject, session))
    if run == 'inference':
        log_dictionary = {file[-26:-11]: pd.read_table(file) for file in files}
    elif run == 'instructed':
        log_dictionary = {file[-27:-11]: pd.read_csv(file) for file in files}
    if (subject == 'sub-6') & (session == 'ses-2') & (run == 'inference'):                                 # sub-6 clearly misunderstoode the rules in ses-2, inf_run-4....
        run = 'inference_run-4'
        log_dictionary[run].loc[log_dictionary[run].event == 'CHOICE_TRIAL_RESP', 'value'] =\
            -log_dictionary[run].loc[log_dictionary[run].event == 'CHOICE_TRIAL_RESP'].value.astype(float).values + 1
        log_dictionary[run].loc[log_dictionary[run].event == 'GL_TRIAL_REWARD', 'value'] =\
            -log_dictionary[run].loc[log_dictionary[run].event == 'GL_TRIAL_REWARD'].value.astype(float).values + 1
    return log_dictionary


def LLR(value, e1=0.5, e2=-0.5, sigma=1):
    """
    Computes LLR.

    Takes means of two distributions and the common variance sigma.
    """
    LLR = np.log(norm.pdf(value, e1, sigma)) - \
        np.log(norm.pdf(value, e2, sigma))
    return LLR


def prior(b_prior, H):
    """
    Returns weighted prior belief given blief at time t-1 and hazard rate H.
    """
    psi = b_prior + np.log((1 - H) / H + np.exp(-b_prior)) - \
        np.log((1 - H) / H + np.exp(b_prior))
    return psi


def murphy_surprise(psi, llr):
    '''
    Calculates surprise measure according to Peter.
    '''
    surprise = - (psi * llr)
    return surprise


def pcp(loc, ln_prev, H, e_right=0.5, e_left=-0.5, sigma=1):
    p_left = 1 / (math.exp(ln_prev) + 1)
    p_right = 1 - p_left
    pcp = H * (norm.pdf(loc, e_right, sigma) * p_left + norm.pdf(loc, e_left, sigma) * p_right) /\
        (H * (norm.pdf(loc, e_right, sigma) * p_left + norm.pdf(loc, e_left, sigma) * p_right) +
         (1 - H) * (norm.pdf(loc, e_right, sigma) * p_right + norm.pdf(loc, e_left, sigma) * p_left))
    return pcp


def belief(df, H, gen_var=1, point_message='GL_TRIAL_LOCATION', ident='message'):
    """
    Computes Glaze belief, LLR, psi and Murphy surprise.

    - Arguments:
        a) pd.DataFrame with behavior (Output of load_logs_bids)
        b) subjective hazard rate H of subject
        c) generative variance in the Glaze model
        d) message value in the behavioral pd.DataFrame that marks new point
        e) column name of message values in behavioral pd.DataFrame

    -Output: Glaze belief, LLR, psi, surprise all as pd.Series

    """
    locs = (df.loc[df['{}'.format(ident)] == point_message, 'value']
            .astype(float))
    belief = 0 * locs.values
    psi = 0 * locs.values
    pcp_surprise = 0 * locs.values
    for i, value in enumerate(locs):
        if i == 0:
            belief[i] = LLR(value, sigma=gen_var)
        else:
            belief[i] = prior(belief[i - 1], H) + LLR(value, sigma=gen_var)
            psi[i] = prior(belief[i - 1], H)
            pcp_surprise[i] = pcp(value, belief[i - 1], H)
    #surprise = murphy_surprise(psi, LLR(locs.values))
    return pd.Series(belief, index=locs.index), pd.Series(psi, index=locs.index), pd.Series(LLR(locs.values), index=locs.index), pd.Series(pcp_surprise, index=locs.index)


__version__ = '4.0.1'
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
4.0.1
-reduced to basic functionality
'''

load_logs_bids('sub-6', 'ses-2', '/Volumes/flxrl/FLEXRULE/raw/bids_mr_v1.2')
