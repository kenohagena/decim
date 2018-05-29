import decim
import pandas as pd
import numpy as np
from os.path import join, expanduser
import pickle
from decim import glaze_control as gl
from itertools import zip_longest
from multiprocessing import Pool
bids_mr = '/work/faty014/bids_mr/'


def keys():
    for subject in range(22):
        for session in range(3):
            yield(subject + 1, session + 1)


def fit_session(subject, session):
    data = gl.stan_data_control(subject, session, bids_mr)
    model_file = decim.get_data('stan_models/inv_glaze_b_fixgen_var.stan')
    try:
        sm = pickle.load(open(compilefile, 'rb'))
    except IOError:
        sm = pystan.StanModel(file=model_file)
        pickle.dump(sm, open(compilefile, 'wb'))
    fit = sm.sampling(data=data, iter=5000, chains=4, n_jobs=1)
    d = {parameter: fit.extract(parameter)[parameter] for parameter in ['H', 'V']}
    d = pd.DataFrame(d)
    d.to_csv('sub-{0}_ses-{1}_stanfit.csv'.format(subject, session))


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def par_execute(chunk):
    # print(ii, len(chunk))
    chunk = [arg for arg in chunk if arg is not None]
    with Pool(16) as p:
        p.starmap(execute, chunk)


def submit():
    from decim import slurm_submit as slu
    for chunk in grouper(keys(), 16):
        slu.pmap(par_execute, chunk, walltime='2:00:00',
                 memory=60, nodes=1, tasks=16, name='bids_stan')
