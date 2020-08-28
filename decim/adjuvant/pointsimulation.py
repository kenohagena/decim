import numpy as np
import pandas as pd
import random
from decim.adjuvant import glaze_model as glaze
from scipy.special import erf
from scipy.special import expit
from scipy.stats import expon
from scipy.stats import norm
random.seed()


'''
This script simulates data.

1. "fast_sim" simuates samples and time points of choice trials.
2. "add_belief" calculates the Glaze belief using "decim.adjuvant.glaze_model.py"
3. "dec_choice_depr", "dec_choice" and "dec_choice_inv" compute choices based on
    - Glaze belief at choice trial time points
    - different parameterizations of internal noise V
4. "complete" employs the functions 2) and 3)
5. "data_from_df" takes output of complete and makes dict for stan fitting


All-in-one use:
   --> data_from_df(complete(fast_sim(total_trials, isi), H, V, gauss, method))
'''


# SIMULATE POINT LOCATIONS AND DECISION POINTS

def fast_sim(x, tH=1 / 70, nodec=5, isi=35., gen_var=1):
    """
    Simulates a dataset with x trials and true hazardrate tH. Does so faster.

    nodec = minimum points between decisions. Nodec points are shown, after that 'isi' determines decision probability.
    """
    inter_choice_dists = np.cumsum(expon.rvs(scale=1 / (1 / isi), size=10000))
    inter_choice_dists = np.array([int(j + nodec + nodec * (np.where(inter_choice_dists == j)[0]))
                                   for j in inter_choice_dists])  # adds 5 (nodec) points between every decision
    inter_choice_dists = inter_choice_dists[inter_choice_dists < x]

    mus = []
    values = []
    start = random.choice([0.5, -0.5])
    cnt = 0
    while cnt < x:
        i = 1 + int(np.round(expon.rvs(scale=1 / tH)))
        mus.append([start] * i)
        values.append(norm.rvs(start, gen_var, size=i))
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
    df['belief'] = glazes[0]
    df['LLR'] = glazes[2]
    df['PCP'] = glazes[3]
    df['psi'] = glazes[1]
    return df


def dec_choice_depr(df, V=1):
    '''
    Chooses at decision trials between 0 ('left') and 1 ('right').

    Based on belief and internal noise V.
    '''
    df['noisy_belief'] = .5 + .5 * erf(df.belief / (np.sqrt(2) * V))
    df['choice'] = np.random.rand(len(df))
    df['choice'] = df.noisy_belief > df.choice
    df.choice = df.choice.astype(int)
    return df


def dec_choice(df, gauss=1):
    '''
    Chooses at decision trials between 0 ('left') and 1 ('right').

    Based on belief and gaussian noise.
    '''
    df['noisy_belief'] = df.belief + np.random.normal(scale=gauss, size=len(df))
    df['choice'] = df.noisy_belief > 0
    df.choice = df.choice.astype(int)
    return df


def dec_choice_inv(df, V=1):
    '''
    Chooses at decision trials between 0 ('left') and 1 ('right').

    Based on belief and internal noise V.
    '''
    df['noisy_belief'] = expit(df.belief / V)
    df = df.fillna(method='ffill')
    df['choice'] = np.random.rand(len(df))
    df['choice'] = df.noisy_belief > df.choice
    df.choice = df.choice.astype(int)
    return df


def complete(df, H, gen_var=1, gauss=1, V=1, method='sign'):
    """
    Completes simulated dataframe with message, location, belief, rule and correctness

    - Arguments:
        a) simulated points (output of fast_sim)
        b) Hazard rate H
        c) generative variance
        d) internal noise (gauss and V)
        e) and a parameterization of internal noise ("sign", 'erf' or "inverse")
    """
    if method == 'sign':
        return dec_choice(add_belief(df, H, gen_var=gen_var), gauss=gauss)
    if method == 'erf':
        return dec_choice_depr(add_belief(df, H, gen_var=gen_var), V=V)
    if method == 'inverse':
        return dec_choice_inv(add_belief(df, H, gen_var=gen_var), V=V)


def data_from_df(df):
    '''
    Returns stan ready data dict from pointsimulation dataframe.
    '''
    decisions = df.loc[df.message == 'decision']
    data = {
        'I': len(decisions),
        'N': len(df),
        'obs_decisions': decisions.choice.values,
        'obs_idx': np.array(decisions.index.values).astype(int) + 1,
        'x': df.value.values,
        'B': 1,
        'b': np.array([0, len(df)]).astype(int)
    }
    return data


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
