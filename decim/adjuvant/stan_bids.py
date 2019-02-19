import decim
import pandas as pd
import numpy as np
import pickle
from itertools import zip_longest
from multiprocessing import Pool
import pystan
from decim.adjuvant import glaze_model as gl
from os.path import join
from glob import glob
import datetime
from decim.adjuvant import slurm_submit as slu
from decim.adjuvant import statmisc


'''
Extract and format behavioral data per subject and session
and do parameter fits using Stan model.

Set options at the beginning of script, than use "submit"
Submit function works with HUMMEL cluster.
'''


# SET OPTIONS
bids_mr = '/work/faty014/FLEXRULE/raw/bids_mr_v1.2'
flex_dir = '/work/faty014/FLEXRULE'
subjects = [1, 2]
sessions = [1, 2, 3]


def keys():
    for subject in subjects:
        for session in sessions:
            yield(subject, session)


def stan_data_control(sub, ses, path, swap=False):
    '''
    Returns dictionary with data that fits requirement of stan model.

    Takes integer subject, integer session and filepath.
    '''
    lp = [0]
    logs = gl.load_logs_bids('sub-{}'.format(sub), 'ses-{}'.format(ses), path)
    df = pd.concat(logs)
    lp = [0]
    for key, value in logs.items():
        block_points = np.array(value.loc[value.event == 'GL_TRIAL_LOCATION',
                                          'value'].index).astype(int)
        lp.append(len(block_points))
    df = df.loc[df.event != '[0]']
    df = df.loc[df.event != 'BUTTON_PRESS']                                     # sometimes duplicates
    df.index = np.arange(len(df))
    points = df.loc[df.event == 'GL_TRIAL_LOCATION']['value'].astype(float)
    point_count = len(points)
    decisions = df.loc[df.event == 'CHOICE_TRIAL_RULE_RESP',
                                   'value']
    decisions = decisions.replace('n/a', np.nan).astype(float)                  # 'n/a' gives error
    if swap is True:
        decisions = decisions
    else:
        decisions = -(decisions[~np.isnan(decisions)].astype(int)) + 1
    dec_count = len(decisions)

    decisions = decisions.dropna().astype(float)
    belief_indices = df.loc[decisions.index].index.values
    pointinds = np.array(points.index)
    dec_indices = np.searchsorted(pointinds, belief_indices)                    # np.searchsorted looks for position where belief index would fit into pointinds
    data = {
        'I': dec_count,                                                         # number of decisions
        'N': point_count,                                                       # number of point samples
        'obs_decisions': decisions.values.astype(int),                          # decisions (0 or 1)
        'x': points.values,                                                     # sample values
        'obs_idx': dec_indices,                                                 # indices of decisions
        'B': len(logs),                                                         # number of total samples
        'b': np.cumsum(lp)                                                      # indices of new block within session (belief --> zero)
    }

    return data


def fit_session(sub, ses, bids_mr=bids_mr, flex_dir=flex_dir):
    '''
    Fit Glaze model for subject and session using Stan.

    - Arguments:
        a) subject (just number)
        b) session (just number)
        c) directory of raw BIDS data set
        d) output directory
    '''
    try:
        data = stan_data_control(sub, ses, bids_mr)
        model_file = decim.get_data('stan_models/inv_glaze_b_fixgen_var.stan')
        compilefile = join(flex_dir, 'inv_glaze_b_fixgen_var_compiled.stan')
        try:                                                                    # reduce memory load by only compiling the model once at the beginning
            sm = pickle.load(open(compilefile, 'rb'))
        except IOError:
            sm = pystan.StanModel(file=model_file)
            pickle.dump(sm, open(compilefile, 'wb'))
        fit = sm.sampling(data=data, iter=5000, chains=4, n_jobs=1)
        d = pd.DataFrame({parameter: fit.extract(parameter)[parameter]
                          for parameter in ['H', 'V']})
        out_dir = join(flex_dir, 'Stan_Fits_{0}'.format(datetime.datetime.now().
                                                        strftime("%Y-%m-%d")))
        slu.mkdir_p(out_dir)
        print(out_dir)
        d.to_hdf(join(out_dir, 'sub-{0}_stanfit.hdf'.
                      format(sub)), key='ses-{}'.format(ses))
    except RuntimeError as e:
        print("No file found for subject {0}, session {1}, path {2}".
              format(sub, ses, bids_mr), e)


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def par_execute(chunk):
    # print(ii, len(chunk))
    chunk = [arg for arg in chunk if arg is not None]
    with Pool(6) as p:
        p.starmap(fit_session, chunk)


def submit():
    for chunk in grouper(keys(), 6):                                            # more than 6 crashes the node
        slu.pmap(par_execute, chunk, walltime='2:00:00',
                 memory=60, nodes=1, tasks=16, name='bids_stan')


def concatenate(input_dir):
    '''
    Concatenate fitted parameters for sessions and subjects. Compute mode and confidence intervals.
    Return dictionary.
    '''
    summary = []
    files = glob(join(input_dir, '.hdf'))
    for file in files:
        for key in file.keys():
            s = pd.read_hdf(file, key=key)
            dr = {'vmode': statmisc.mode(s.V.values, 50, decimals=False),
                  'vupper': statmisc.hdi(s.V.values)[1],
                  'vlower': statmisc.hdi(s.V.values)[0],
                  'hmode': statmisc.mode(s.H.values, 50, decimals=False),
                  'hupper': statmisc.hdi(s.H.values)[1],
                  'hlower': statmisc.hdi(s.H.values)[0],
                  'subject': file[:5], 'session': key}
            summary.append(dr)
    summary = pd.DataFrame(summary)
    summary.to_csv(join(input_dir, 'summary_stan_fits.csv'))
