import pystan
import decim 
from decim import pointsimulation as pt
from decim import pystan_workflow as pw
from decim import glaze_stan as gs

# SIMULATE DATA
rH = 1 / 70
rgen_var = 1
rV = 1
trials = 5000
isi = 10


def keys():
    for V in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
        for i in range(20):
            yield(.015, V, 1, i, 'V')

    for H in [.01, .05, .1, .15, .2, .25, .3, .35, .4, .45]:
        for i in range(20):
            yield(H, 1, 1, i, 'H')

    for gen_var in [1, 1.5, 2, 2.5, 3, 3.5]:
        for i in range(20):
            yield(.015, 1, gen_var, i, 'gen_var')

def execute(H, V, gv, i, var):
    model_file = decim.get_data('stan_models/inv_glaze_b.stan')
    sm = pystan.StanModel(file=model_file)
    points = pt.fast_sim(trials, isi=isi)
    data = pt.complete(points, V=V, gen_var=gv, H=H, method='inverse')
    data = gs.data_from_df(data)

    fit = sm.sampling(data=data, iter=100, chains=1)
    name = '{4}{3}V={0}H={1}gv={2}'.format(V, H, gv, i, var)
    fitted = pw.fit_result(fit, name, parameters=['H', 'V', 'gen_var'])
    fitted.samples()
    fitted.to_csv(samples=True)
