import decim
import pandas as pd
import pickle
from decim import glaze_control as gl
from itertools import zip_longest
from multiprocessing import Pool
import pystan

# SET OPTIONS
bids_mr = '/work/faty014/bids_mr_v1.1/'
subjects = []
sessions = []


def keys():
    for subject in subjects:
        for session in sessions:
            yield(subject, session)


def fit_session(subject, session):
    try:
        data = gl.stan_data_control(subject, session, bids_mr)
        model_file = decim.get_data('stan_models/inv_glaze_b_fixgen_var.stan')
        compilefile = '/work/faty014/inv_glaze_b_fixgen_var_compiled.stan'
        try:
            sm = pickle.load(open(compilefile, 'rb'))
        except IOError:
            sm = pystan.StanModel(file=model_file)
            pickle.dump(sm, open(compilefile, 'wb'))
        fit = sm.sampling(data=data, iter=5000, chains=4, n_jobs=1)
        d = {parameter: fit.extract(parameter)[parameter] for parameter in ['H', 'V']}
        d = pd.DataFrame(d)
        d.to_csv('/work/faty014/sub-{0}_ses-{1}_stanfit.csv'.format(subject, session))
    except RuntimeError:
        print("No file found for subject {0}, session {1}".format(subject, session))


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def par_execute(chunk):
    # print(ii, len(chunk))
    chunk = [arg for arg in chunk if arg is not None]
    with Pool(6) as p:
        p.starmap(fit_session, chunk)


def submit():
    from decim import slurm_submit as slu
    for chunk in grouper(keys(), 6):
        slu.pmap(par_execute, chunk, walltime='2:00:00',
                 memory=60, nodes=1, tasks=16, name='bids_stan')
