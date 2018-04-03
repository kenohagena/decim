import glaze_stan_cluster as gs
import pystan
import pandas as pd
import numpy as np
import pickle

sessions = {'A': 1, 'B': 2, 'C': 3}
subjects = ['VPIM01', 'VPIM02', 'VPIM03', 'VPIM04', 'VPIM06', 'VPIM07', 'VPIM09']
model = gs.model_code()
sm = pystan.StanModel(model_code=model)


def keys():
    for sub in subjects:
        for ses in sessions:
            if (sub == 'VPIM03') & (ses == 'B'):
                yield sub, ses, sessions[ses], [1, 2, 3, 4, 5, 6], 'immuno/data/vaccine'
            else:
                yield sub, ses, sessions[ses], [1, 2, 3, 4, 5, 6, 7], 'immuno/data/vaccine'


def execute(x):
    i, j, k, l, m = x
    data = gs.stan_data(i, j, k, l, m)
    fit = sm.sampling(data=data, iter=10000, chains=8)
    with open("immuno/glaze_int_noise_fits260218/model_fit{0}{1}.pkl".format(i, j), "wb") as f:
        pickle.dump({'model': model, 'fit': fit}, f, protocol=-1)


for i in keys():
    execute(i)
