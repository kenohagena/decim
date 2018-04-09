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

models = {'vfix': 'stan_models/inv_glaze_b_fixV.stan', 'gvfix':'stan_models/inv_glaze_b_fixgen_var.stan'}

def fix_keys():
    for V in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
        for i in range(20):
            yield(.015, V, 1, i, 'V', models['gvfix'], 'gvfix')

    for H in [.01, .05, .1, .15, .2, .25, .3, .35, .4, .45]:
        for model in models:
            for i in range(20):
                yield(H, 1, 1, i, 'H', models[model], model)

    for gen_var in [1, 1.5, 2, 2.5, 3, 3.5]:
        for i in range(20):
            yield(.015, 1, gen_var, i, 'gen_var', models['vfix'], 'vfix')

def execute(H, V, gv, i, var, model, fixed_variable):
    model_file = decim.get_data(model)
    sm = pystan.StanModel(file=model_file)
    points = pt.fast_sim(trials, isi=isi)
    data = pt.complete(points, V=V, gen_var=gv, H=H, method='inverse')
    data = gs.data_from_df(data)

    fit = sm.sampling(data=data, iter=100, chains=1)
    name = '{5}_{4}{3}_V={0}H={1}gv={2}'.format(V, H, gv, i, var, fixed_variable)
    fitted = pw.fit_result(fit, parameters=['H', 'V', 'gen_var'])
    fitted.samples()
    fitted.to_csv(name=name, samples=True)
