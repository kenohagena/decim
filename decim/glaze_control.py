from glob import glob
from os.path import join
from scipy.io import loadmat
import numpy as np
from decim import glaze2 as gl
import difflib
import pandas as pd


def load_logs_control(sub_code, session, base_path):
    """
    Concatenates path and file name and loads matlba file.
    Recquires subject code, session, phase and block.
    Returns path and name of loaded files and the matlab file itself.
    """
    directory = join(base_path,
                     "sub-0{}".format(sub_code),
                     "ses-0{}".format(session),
                     'behav')
    files = sorted(glob(join(directory, '*inference*.mat')))

    if len(files) == 0:
        raise RuntimeError(
            'No log file found for this block: %s, %s' %
            (sub_code, session))
    logs = []
    for i in range(len(files)):
        logs.append(loadmat(files[i]))
    return files, logs


def stan_data_control(subject, session, path, swap=False):
    '''
    Returns dictionary with data that fits requirement of stan model.

    Takes subject, session, phase, list of blocks and filepath.
    '''
    dflist = []
    lp = [0]
    logs = load_logs_control(subject, session, path)[1]
    filenames = load_logs_control(subject, session, path)[0]
    d = difflib.SequenceMatcher(None, filenames[0], filenames[1]).get_matching_blocks()
    first_block = filenames[0][d[0].size]
    for i in range(len(logs)):
        d = gl.log2pd(logs[i], i + float(first_block))
        single_block_points = np.array(d.loc[d.message == 'GL_TRIAL_LOCATION']['value'].index).astype(int)
        dflist.append(d)
        lp.append(len(single_block_points))
    df = pd.concat(dflist)
    df = df.loc[df.message != '[0]']
    df = df.loc[df.message != 'BUTTON_PRESS']  # drop these, because sometimes duplicated
    df.index = np.arange(len(df))
    point_locs = np.array(df.loc[df.message == 'GL_TRIAL_LOCATION']['value']).astype(float)
    point_count = len(point_locs)
    decisions = np.array(df.loc[df.message == 'CHOICE_TRIAL_RULE_RESP']['value']).astype(float)
    if swap == True:
        decisions = decisions
    else:
        decisions = -(decisions[~np.isnan(decisions)].astype(int)) + 1  # '-', '+1', because of mapping of rule response
    dec_count = len(decisions)
    choices = (df.loc[df.message == "CHOICE_TRIAL_RULE_RESP", 'value']
               .astype(float))
    choices = choices.dropna()
    belief_indices = df.loc[choices.index - 11].index.values
    ps = df.loc[df.message == 'GL_TRIAL_LOCATION']['value'].astype(float)
    pointinds = np.array(ps.index)
    dec_indices = np.searchsorted(pointinds, belief_indices) + 1  # '+1' because stan starts counting from 1
    data = {
        'I': dec_count,
        'N': point_count,
        'y': decisions,
        'x': point_locs,
        'D': dec_indices,
        'B': len(logs),
        'b': np.cumsum(lp)
    }

    return data


def performance_control(sub_code, session, base_path):
    """
    returns dictionary containing number of decisions, number of NaNs and count of rewards.
    """
    logs = load_logs_control(sub_code, session, base_path)[1]
    filenames = load_logs_control(sub_code, session, base_path)[0]
    d = difflib.SequenceMatcher(None, filenames[0], filenames[1]).get_matching_blocks()
    first_block = filenames[0][d[0].size]
    dflist = []
    for i in range(len(logs)):
        d = gl.log2pd(logs[i], i + float(first_block))
        dflist.append(d)
    df = pd.concat(dflist)
    df = df.loc[df.message != '[0]']
    df = df.loc[df.message != 'BUTTON_PRESS']  # drop these, because sometimes duplicated
    df.index = np.arange(len(df))
    rews = (df.loc[df.message == "GL_TRIAL_REWARD", 'value'])
    array = np.array(rews.values).astype(float)
    no_answer = np.count_nonzero(np.isnan(array))
    rewards = np.count_nonzero((array)) - np.count_nonzero(np.isnan(array))
    #performance = rewards / len(array)
    # return {'no_answer': no_answer, 'rewards': rewards, 'decisions': len(array), 'performance': performance}
    rresp = df.loc[df.message == 'CHOICE_TRIAL_RULE_RESP', 'value']
    rewards_manually = np.array(rresp) + np.array(df.loc[rresp.index - 6, 'value'])
    return np.sum(rewards_manually == 0.5) / len(rewards_manually)


def mean_rt(sub_code, session, base_path):
    """
    Returns mean reaction time of given block.
    """
    logs = load_logs_control(sub_code, session, base_path)[1]
    filenames = load_logs_control(sub_code, session, base_path)[0]
    d = difflib.SequenceMatcher(None, filenames[0], filenames[1]).get_matching_blocks()
    first_block = filenames[0][d[0].size]
    dflist = []
    for i in range(len(logs)):
        d = gl.log2pd(logs[i], i + float(first_block))
        dflist.append(d)
    df = pd.concat(dflist)
    rt = df.loc[df.message == 'CHOICE_TRIAL_RT']['value']
    return rt.mean()


def acc_ev_ctrl(sub_code, session, base_path, H):
    """returns accumulated evidence and rt at decision points."""
    logs = load_logs_control(sub_code, session, base_path)[1]
    filenames = load_logs_control(sub_code, session, base_path)[0]
    d = difflib.SequenceMatcher(None, filenames[0], filenames[1]).get_matching_blocks()
    first_block = filenames[0][d[0].size]
    dflist = []
    for i in range(len(logs)):
        d = gl.log2pd(logs[i], i + float(first_block))
        dflist.append(d)
    df = pd.concat(dflist)
    df = df.loc[df.message != '[0]']
    df = df.loc[df.message != 'BUTTON_PRESS']  # drop these, because sometimes duplicated
    df.index = np.arange(len(df))
    choices = (df.loc[df.message == "CHOICE_TRIAL_RULE_RESP", 'value']
               .astype(float))
    #choices = choices.dropna()
    belief_indices = df.loc[choices.index - 11].index.values
    rt = df.loc[df.message == 'CHOICE_TRIAL_RT']['value']
    accum_ev = gl.belief(df, H).loc[belief_indices].values
    return pd.DataFrame({'reaction time': rt, 'accumulated evidence': accum_ev})


bp = '/users/kenohagena/documents/immuno/data/control'
print(acc_ev_ctrl(1, 3, bp, 1 / 70))
