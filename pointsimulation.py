# IMPORT STUFF

import numpy as np
import pandas as pd
import glaze2 as glaze
import matplotlib.pyplot as plt
import random

# SIMULATE POINT LOCATIONS AND DECISION POINTS

# decision trial or location trial


def what_trial(i):
    '''Determines subsequent trial type according to p(decision trial): 'dt'.

    decides what function to use.'''
    dt = 1 / 35  # probability of decision trial dt
    y = random.random()
    if y < dt:
        x = decision(i)
    else:
        x = location(i)
    return x

# decision trial


def decision(i):
    '''returns object of type dict containing index, message and rule.'''
    x = {'index': i, 'message': 'decision'}
    return x

# location trial:


def location(i):
    '''Ddecides whether to keep current distribution mean mu or change it with probability 'dt'.

    returns object of type dict which contains message, location value and rule.'''
    H = 1 / 70
    y = random.random()
    if y < H:
        global mu
        mu = -mu
        location = random.gauss(mu, sigma=1)
        x = {'value': location, 'index': i, 'rule': mu, 'message': 'GL_TRIAL_LOCATION'}
    else:
        mu = mu
        location = random.gauss(mu, sigma=1)
        x = {'value': location, 'index': i, 'rule': mu, 'message': 'GL_TRIAL_LOCATION'}
    return x


# create pandas DataFrame
# initialize DataFrame
def simulate(x):
    columns = ['index', 'message', 'rule', 'value']
    index = [0]
    df = pd.DataFrame(index=index, columns=columns)

    # simulate on i runs
    global mu
    mu = random.choice([0.5, -0.5])
    for i in range(x):
        df = df.append(what_trial(i), ignore_index=True)
    return df

# FILL IN MISSING INFORMATION


def add_belief(df, H):
    '''takes dataframe, computes belief according to glaze model with Hazard rate H.

    returns concatenated Dataframe df containing former df plus belief.'''
    glazes = glaze.belief(df, H)
    glazesdf = pd.DataFrame(glazes, columns=['belief'])
    df = pd.concat([df, glazesdf], axis=1)
    return df


def fill_decbel(df):
    '''fills the belief field in each decision row by taking the belief at the last location trial.'''
    decision_indices = df.loc[df.message == 'decision'].index
    df.loc[df.message == 'decision', 'belief'] = df.loc[decision_indices - 1, 'belief'].values
    return df


def fill_decrule(df):
    '''fills the rule field in each decision row by taking the rule at the last location trial.'''
    decision_indices = df.loc[df.message == 'decision'].index
    df.loc[df.message == 'decision', 'rule'] = df.loc[decision_indices - 1, 'rule'].values
    return df


def add_correct(df):
    '''adds column with boolean index of correctness of models guess.'''
    df['correct'] = df['rule'] * df['belief'] > 0
    return df


def complete(df, H):
    '''takes simulated raw dataframe and Hazardrate.

    returns complete df containing message, location, belief, rule and correctness off belief over all trials.'''
    return add_correct(fill_decrule(fill_decbel(add_belief(df, H))))


def cordec(df):
    '''returns percentage of correct answers at decicion trials.'''
    return df.loc[df.message == 'decision', 'correct'].mean()


__version__ = '1.1'

'''1.1
changed probability of decision trial to 1/35.'''
