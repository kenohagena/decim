import numpy as np
import math
from scipy import stats

'''
Helper functions.
'''


def hdi(smaples, cred_mass=0.95):
    '''
    Returns highest density interval.

    Takes MCMC sample array and optional credibility mass.
    '''
    sortedpoints = np.sort(smaples)
    ci = math.ceil(cred_mass * len(smaples))
    nci = len(smaples) - ci
    ci_width = []
    for i in range(nci):
        ci_width.append(sortedpoints[i + ci] - sortedpoints[i])
    hdi_min = sortedpoints[np.argmin(ci_width)]
    hdi_max = sortedpoints[np.argmin(ci_width) + ci]
    return (hdi_min, hdi_max)


def mode(samples, bins, decimals=True):
    '''
    Return bin with highest modal value.

    If decimals == True (Default), rounded to 5 decimals.
    '''
    z = np.argmax(np.histogram(samples, 100)[0])
    x = np.histogram(samples, 100)[1][z]
    if decimals is True:
        return float('%.5f' % (x))
    else:
        return x


def ranova(dataframe, condition, subject, further_groupby=None):
    '''
    Calculates repeated measures ANOVA and returns F, p and degrees of freedom.

    Takes a pd dataframe with condition and subject as hierarchical indices.
    Furhter_groupby takes further indices that do not declare a condition.
    '''
    if further_groupby is None:
        k = dataframe.groupby('{}'.format(subject)).size()[1]   # number of conditions
    else:
        k = dataframe.groupby(['{}'.format(subject), '{}'.format(further_groupby)]).size()[1]   # number of conditions
    n = dataframe.groupby('{}'.format(condition)).size()[1]  # Participants in each condition

    ss_between = sum(((dataframe.groupby('{}'.format(condition)).mean()['value']) - dataframe.mean()['value'])**2) * n
    sum_y_squared = sum([value**2 for value in dataframe['value'].values])
    ss_within = sum_y_squared - sum(dataframe.groupby('{}'.format(condition)).sum()['value']**2) / n
    ss_subject = sum(((dataframe.groupby('{}'.format(subject)).mean()['value']) - dataframe.mean()['value'])**2) * k

    df_between = k - 1
    df_error = (n - 1) * (k - 1)

    ss_error = ss_within - ss_subject
    ms_error = ss_error / df_error
    ms_between = ss_between / df_between
    F = ms_between / ms_error
    p = stats.f.sf(F, df_between, df_error)
    return F, p, df_between, df_error


__version__ = '1.0.1'
'''
1.0
functions for calcuating mode, highest density interval and rANOVA.
'''
