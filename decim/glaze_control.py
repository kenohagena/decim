from glob import glob
from os.path import join
import numpy as np
from decim import glaze2 as gl
import pandas as pd


def load_logs_bids(subject, session, base_path):
    '''
    Returns filenames and pandas frame.
    '''
    if session == 1:
        modality = 'beh'
    else:
        modality = 'func'
    directory = join(base_path,
                     'sub-{}'.format(subject),
                     'ses-{}'.format(session),
                     modality)
    files = sorted(glob(join(directory, '*inference*.tsv')))
    if len(files) == 0:
        raise RuntimeError(
            'No log file found for this block: %s, %s' %
            (subject, session))
    logs = []
    for i in range(len(files)):
        logs.append(pd.read_table(files[i]))
    return files, logs


def stan_data_control(subject, session, path, swap=False):
    '''
    Returns dictionary with data that fits requirement of stan model.

    Takes subject, session, phase, list of blocks and filepath.
    '''
    lp = [0]
    logs = load_logs_bids(subject, session, path)[1]
    df = pd.concat(logs)
    lp = [0]
    for i in range(len(logs)):
        d = logs[i]
        block_points = np.array(d.loc[d.event == 'GL_TRIAL_LOCATION',
                                                 'value'].index).astype(int)
        lp.append(len(block_points))
    df = df.loc[df.event != '[0]']
    df = df.loc[df.event != 'BUTTON_PRESS']  # sometimes duplicates
    df.index = np.arange(len(df))
    points = df.loc[df.event == 'GL_TRIAL_LOCATION']['value'].astype(float)
    point_count = len(points)
    decisions = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP',
                                   'value'].astype(float)
    if swap is True:
        decisions = decisions
    else:
        decisions = -(decisions[~np.isnan(decisions)].astype(int)) + 1
    dec_count = len(decisions)

    decisions = decisions.dropna()
    belief_indices = df.loc[decisions.index].index.values
    pointinds = np.array(points.index)
    dec_indices = np.searchsorted(pointinds, belief_indices)  # np.searchsorted looks for position where belief index would fit into pointinds
    data = {
        'I': dec_count,
        'N': point_count,
        'obs_decisions': decisions.values,
        'x': points.values,
        'obs_idx': dec_indices,
        'B': len(logs),
        'b': np.cumsum(lp)
    }

    return data


def performance_control(subject, session, base_path):
    '''
    Returns performance and no_answer percentage.
    '''
    logs = load_logs_bids(subject, session, base_path)[1]
    df = pd.concat(logs)
    df = df.loc[df.event != '[0]']
    df = df.loc[df.event != '0']
    df = df.loc[df.event != 'BUTTON_PRESS']
    df.index = np.arange(len(df))
    rews = (df.loc[df.event == "GL_TRIAL_REWARD", 'value'])
    array = np.array(rews.values).astype(float)
    no_answer = np.count_nonzero(np.isnan(array))
    rresp = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP', 'value']
    rewards_manually = np.array(rresp).astype(float) +\
        np.array(df.loc[rresp.index - 6, 'value']).astype(float)
    performance = np.sum(rewards_manually == 0.5) / len(rewards_manually)
    return performance, no_answer


def mean_rt(subject, session, base_path):
    '''
    Returns mean reaction time of given block.
    '''
    logs = load_logs_bids(subject, session, base_path)[1]
    df = pd.concat(logs)
    rt = df.loc[df.message == 'CHOICE_TRIAL_RT']['value'].astype(float)
    return rt.mean()


def accev(subject, session, base_path, H):
    '''
    returns accumulated evidence and rt at decision points.
    '''
    logs = load_logs_bids(subject, session, base_path)[1]
    df = pd.concat(logs)
    df = df.loc[df.event != '[0]']
    df = df.loc[df.event != '0']
    df = df.loc[df.event != 'BUTTON_PRESS']
    df.index = np.arange(len(df))
    choices = (df.loc[df.event == "CHOICE_TRIAL_RULE_RESP", 'value']
               .astype(float))
    belief_indices = df.loc[choices.index - 11].index.values
    rt = df.loc[df.message == 'CHOICE_TRIAL_RT']['value']
    accum_ev = gl.belief(df, H).loc[belief_indices].values
    return pd.DataFrame({'reaction time': rt,
                         'accumulated evidence': accum_ev})
