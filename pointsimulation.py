import numpy as np
import pandas as pd
import glaze2 as glaze
import random
from scipy import optimize as opt
from scipy.special import expit, erf  # logistic function
random.seed()
from scipy.stats import expon, norm
import math


# SIMULATE POINT LOCATIONS AND DECISION POINTS

def fast_sim(x, tH=1 / 70, nodec=5, isi=35.):
    """
    Simulates a dataset with x trials and true hazardrate tH. Does so faster.

    nodec = minimum points between decisions. Nodec points are shown, after that 'isi' determines decision probability.
    """
    inter_choice_dists = np.cumsum(expon.rvs(scale=1 / (1 / isi), size=1000))
    inter_choice_dists = np.array([int(j + nodec + nodec * (np.where(inter_choice_dists == j)[0])) for j in inter_choice_dists])  # adds 5 (nodec) points between every decision
    inter_choice_dists = inter_choice_dists[inter_choice_dists < x]

    mus = []
    values = []
    start = random.choice([0.5, -0.5])
    cnt = 0
    while cnt < x:
        i = 1 + int(np.round(expon.rvs(scale=1 / tH)))
        mus.append([start] * i)
        values.append(norm.rvs(start, 1, size=i))
        start *= -1
        cnt += i

    df = pd.DataFrame({'rule': np.concatenate(mus)[:x],
                       'value': np.concatenate(values)[:x]})

    # df.columns = ['rule', 'values']
    df.loc[:, 'message'] = 'GL_TRIAL_LOCATION'
    df.loc[inter_choice_dists, 'message'] = 'decision'
    df.loc[:, 'index'] = np.arange(len(df))
    return df

# FILL IN MISSING INFORMATION


def add_belief(df, H, gen_var=1):
    """
    Computes models belief according to glaze at location trials

    Takes simulated dataframe and hazardrate
    """
    glazes = glaze.belief(df, H, gen_var=gen_var)
    glazesdf = pd.DataFrame(glazes, columns=['belief'])
    df = pd.concat([df, glazesdf], axis=1)
    return df


def fill_decbel(df):
    """
    Fills belief fields at decision trials.
    """
    decision_indices = df.loc[df.message == 'decision'].index
    df.loc[df.message == 'decision', 'belief'] = \
        df.loc[decision_indices - 1, 'belief'].values
    return df


def fill_decrule(df):
    """
    Fills rule field at decision trials.
    """
    decision_indices = df.loc[df.message == 'decision'].index
    df.loc[df.message == 'decision', 'rule'] = \
        df.loc[decision_indices - 1, 'rule'].values
    return df


def dec_choice_depr(df, V=1):
    '''
    Chooses at decision trials between 0 ('left') and 1 ('right') based on belief and internal noise V.
    '''
    df['noisy_belief'] = .5 + .5 * erf(df.belief / (np.sqrt(2) * V))
    df['choice'] = np.random.rand(len(df))
    df['choice'] = df.noisy_belief > df.choice
    df.choice = df.choice.astype(int)
    return df


def dec_choice(df, gauss=1):
    '''
    Chooses at decision trials between 0 ('left') and 1 ('right') based on belief and internal noise V.
    '''
    df['noisy_belief'] = df.belief + np.random.normal(scale=gauss, size=len(df))
    df['choice'] = df.noisy_belief > 0
    df.choice = df.choice.astype(int)
    return df


def complete(df, H, gen_var=1, gauss=1):
    """
    Completes simulated dataframe with message, location, belief, rule and correctness
    """
    return dec_choice(fill_decrule(fill_decbel(add_belief(df, H, gen_var=gen_var))), gauss=gauss)


def cer(df, H):
    """
    Completes simulated dataframe and computes cross entropy error.

    Takes dataframe and hazardrate.
    """
    com = complete(df, H)
    actualrule = com.loc[com.message == 'decision', 'rule'] + 0.5
    modelbelief = expit(com.loc[com.message == 'decision', 'belief'])
    error = -np.sum(((1 - actualrule) * np.log(1 - modelbelief)) +
                    (actualrule * np.log(modelbelief)))
    return error


def opt_h(df):
    """
    Returns hazard rate with best cross entropy error.
    """

    def error_function(x):
        return cer(df, x)
    o = opt.minimize_scalar(error_function,
                            bounds=(0, 1), method='bounded')
    error = cer(df, o.x)
    return o.x, error


def h_iter(I, n, hazardrates, trials=1000):
    """
    Returns numpy matrix containing fitted hazardrates
    for given true hazardrates, iterated n times each.

    Takes number of iterations and hazardrates as a list.
    Saves data in a csv file.
    """
    Ms = []

    for h in hazardrates:
        for i in range(n + 1):
            df = fast_sim(trials, h)
            M, error = opt_h(df)
            true_err = cer(df, h)
            Ms.append({'M': M, 'H': h, 'err': error,
                       'true_error': true_err})
    df = pd.DataFrame(Ms)
    df.loc[:, 'trials'] = trials
    df.to_csv("wt{2}_{0}its{1}hazrates.csv".format(
        I, len(hazardrates), trials))
    return df


__version__ = '2.1'

'''
1.1
changed probability of decision trial to 1/35.
1.1.1
PEP-8 fixes
1.2
added function to compute cross entropy error
added function to calculate optimal model hazardrate
made actual generating hazardrate optional parameter in simulate
2.0
Niklas added fast_sim function to make the modules functionality
considerably faster.
2.1
Size number of inter_change_dist raised to 10 000
Added function to iterate opt_h on a list of tH and an arbitrarily
high number of iterations.
Deleted old simulation functions.
Made the module slightly faster by deleting obsolete functions
calculating correct answers of model.
Readability according to PEP-257
'''
