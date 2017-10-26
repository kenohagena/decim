import numpy as np
import pandas as pd
import glaze2 as glaze
import random
from scipy import optimize as opt
from scipy.special import expit  # logistic function
random.seed()
from scipy.stats import expon, norm

# SIMULATE POINT LOCATIONS AND DECISION POINTS


def what_trial(i, tH):
    '''
    Determines subsequent trial type according to p(decision trial): 'dt'.

    tH is true hazardrate.
    '''
    dt = 1 / 35  # probability of decision trial dt
    y = random.random()
    if y < dt:
        x = decision(i)
    else:
        x = location(i, tH)
    return x

# decision trial


def decision(i):
    '''
    returns object of type dict containing index, message and rule.
    '''
    x = {'index': i, 'message': 'decision'}
    return x

# location trial:


def location(i, tH):
    '''
    determines whether to change current distribution mean mu
    with probability 'dt' and hazardrate tH

    returns dict containing message, location value and rule.
    '''
    y = random.random()
    if y < tH:
        global mu
        mu = -mu
        location = random.gauss(mu, sigma=1)
        x = {'value': location, 'index': i,
             'rule': mu, 'message': 'GL_TRIAL_LOCATION'}
    else:
        mu = mu
        location = random.gauss(mu, sigma=1)
        x = {'value': location, 'index': i,
             'rule': mu, 'message': 'GL_TRIAL_LOCATION'}
    return x


# create pandas DataFrame
# initialize DataFrame
def simulate(x, tH=1 / 70):
    '''
    simulates a dataset with x trials and true hazardrate tH.

    uses the function 'what_trial'.
    '''
    columns = ['index', 'message', 'rule', 'value']
    index = [0]
    df = pd.DataFrame(index=index, columns=columns)

    # simulate on i runs
    global mu
    mu = random.choice([0.5, -0.5])
    for i in range(x):
        df = df.append(what_trial(i, tH), ignore_index=True)
    return df

# FILL IN MISSING INFORMATION


def fast_sim(x, tH=1 / 70):
    inter_change_dists = expon.rvs(scale=1 / (1 / 70), size=1000)
    inter_choice_dists = np.cumsum(expon.rvs(scale=1 / (1 / 35), size=1000))
    inter_choice_dists = inter_choice_dists[inter_choice_dists < x]
    mus = []
    values = []
    start = random.choice([0.5, -0.5])
    cnt = 0
    for i in inter_change_dists:
        mus.append([start] * int(i))
        values.append(norm.rvs(start, 1, size=int(i)))
        start *= -1
        if cnt > x:
            break
        cnt += i

    df = pd.DataFrame({'rule': np.concatenate(mus)[:x], 'value': np.concatenate(
        values)[:x]})

    #df.columns = ['rule', 'values']
    df.loc[:, 'message'] = 'GL_TRIAL_LOCATION'
    df.loc[inter_choice_dists.astype(int), 'message'] = 'decision'
    df.loc[:, 'index'] = np.arange(len(df))
    return df


def add_belief(df, H):
    '''
    input: dataframe and hazardrate H, computes belief according to glaze.

    returns concatenated dataframe containing former df plus belief.
    '''
    glazes = glaze.belief(df, H)
    glazesdf = pd.DataFrame(glazes, columns=['belief'])
    df = pd.concat([df, glazesdf], axis=1)
    return df


def fill_decbel(df):
    '''
    fills the belief field in each decision row

    uses the belief at the last location trial.
    '''
    decision_indices = df.loc[df.message == 'decision'].index
    df.loc[df.message == 'decision', 'belief'] = \
        df.loc[decision_indices - 1, 'belief'].values
    return df


def fill_decrule(df):
    '''
    fills the rule field in each decision row

    uses the rule at the last location trial.
    '''
    decision_indices = df.loc[df.message == 'decision'].index
    df.loc[df.message == 'decision', 'rule'] = \
        df.loc[decision_indices - 1, 'rule'].values
    return df


def add_correct(df):
    '''
    adds column with boolean index of correctness of models guess.
    '''
    df['correct'] = df['rule'] * df['belief'] > 0
    return df


def complete(df, H):
    '''
    takes simulated raw dataframe and Hazardrate.

    returns dataframe with message, location, belief,
    rule and correctness of belief.
    '''
    return add_correct(fill_decrule(fill_decbel(add_belief(df, H))))


def cordec(df):
    '''
    returns percentage of correct answers at decicion trials.
    '''
    return df.loc[df.message == 'decision', 'correct'].mean()


def cer(df, H):
    '''
    Compute cross entropy error

    compares models belief at decision point with actual rule.
    '''
    com = complete(df, H)
    actualrule = com.loc[com.message == 'decision', 'rule'] + 0.5
    modelbelief = expit(com.loc[com.message == 'decision', 'belief'])
    error = -np.sum(((1 - actualrule) * np.log(1 - modelbelief)) +
                    (actualrule * np.log(modelbelief)))
    return error


def opt_h(df):
    '''
    returns hazard rate with best cross entropy error.

    uses simple scalar optimization algorithm.
    '''

    def error_function(x): return cer(df, x)
    o = opt.minimize_scalar(error_function,
                            bounds=(0, 1), method='bounded')
    return o


__version__ = '1.2'

'''
1.1
changed probability of decision trial to 1/35.
1.1.1
PEP-8 fixes
1.2
added function to compute cross entropy error
added function to calculate optimal model hazardrate
made actual generating hazardrate optional parameter in simulate
'''
