import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from scipy.stats import norm

'''
Read in data of session
'''


def load_logs_bids(subject, session, base_path, run='inference'):
    '''
    Returns filenames and pandas frame.
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
        return {file[-26:-11]: pd.read_table(file) for file in files}
    elif run == 'instructed':
        return {file[-27:-11]: pd.read_csv(file) for file in files}


'''
Calcluate params of Glaze model
'''


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


def murphy_surprise(psi, llr):
    '''
    Calculate new surprise measure of Peter
    '''
    surprise = - (psi * llr)
    return surprise


def belief(df, H, gen_var=1, point_message='GL_TRIAL_LOCATION', ident='message'):
    """
    Return pd.Series for belief, psi and LLR at given timepoints
    """
    locs = (df.loc[df['{}'.format(ident)] == point_message, 'value']
            .astype(float))
    belief = 0 * locs.values
    psi = 0 * locs.values
    for i, value in enumerate(locs):
        if i == 0:
            belief[i] = LLR(value, sigma=gen_var)
        else:
            belief[i] = prior(belief[i - 1], H) + LLR(value, sigma=gen_var)
            psi[i] = prior(belief[i - 1], H)
    surprise = murphy_surprise(psi, LLR(locs.values))
    return pd.Series(belief, index=locs.index), pd.Series(prior, index=locs.index), pd.Series(LLR(locs.values), index=locs.index), pd.Series(surprise, index=locs.index)


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
