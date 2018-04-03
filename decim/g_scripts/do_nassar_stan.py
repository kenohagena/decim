import glaze_stan as gs
import pystan
import pandas as pd
import numpy as np
import pickle
import nassar_task as nt

lp = [1]
x = []
y = []
for i in range(7):
    xt = nt.make_sequence(1 / 10, 5, 100, range=(0, 300))
    rb = nt.RBayes(5, 1 / 15)
    for i in range(100):
        rb.nassar_update(xt[i])
    yt = pd.DataFrame(rb.history).mu
    lp.append(len(xt))
    x.append(list(xt))
    y.append(list(yt))
sim_data = {
    'I': int(700),
    'B': int(7),
    'b': np.cumsum(lp).astype(int),
    'x': np.array(sum(x, [])),
    'y': np.array(sum(y, []))
}
#data = gs.nassar_data('VPIM04', 'B', 4, [1, 2, 3, 4, 5, 6, 7], '/Users/kenohagena/Documents/immuno/data/vaccine')
model = gs.nassar_model()

sm = pystan.StanModel(model_code=model)

fit = sm.sampling(data=sim_data, iter=5000, chains=4)

with open("nassar_fit.pkl", "wb") as f:
    pickle.dump({'model': sm, 'fit': fit}, f, protocol=-1)
