import numpy as np
import pandas as pd
from decim.adjuvant import patsy_transform_nw as pt
import sympy
import patsy
from sklearn import linear_model

'''
Fit pupil to IRF elicited by the choice grating.
'''

def IRF_pupil(fs=100, dur=4, s=1.0 / (10**26), n=10.1, tmax=.730):  # tmax=0.93
    """
    Canocial pupil impulse fucntion [/from JW]

    dur: length in s

    """

    # parameters:
    timepoints = np.linspace(0, dur, dur * fs)

    # sympy variable:
    t = sympy.Symbol('t')

    # function:
    y = ((s) * (t**n) * (np.math.e**((-n * t) / tmax)))

    # derivative:
    y_dt = y.diff(t)

    # lambdify:
    y = sympy.lambdify(t, y, "numpy")
    y_dt = sympy.lambdify(t, y_dt, "numpy")

    # evaluate and normalize:
    y = y(timepoints)
    y = y / np.std(y)
    y_dt = y_dt(timepoints)
    y_dt = y_dt / np.std(y_dt)

    # dispersion:
    y_dn = ((s) * (timepoints**(n - 0.01)) * (np.math.e**((-(n - 0.01) * timepoints) / tmax)))
    y_dn = y_dn / np.std(y_dn)
    y_dn = y - y_dn
    y_dn = y_dn / np.std(y_dn)

    return y / y.sum(), y_dt / y_dt.sum(), y_dn / y_dn.sum()


class FitTmax(object):

    def __init__(self, grating):
        self.n_trials = grating.shape[0]
        self.grating = grating
        self.y = grating.stack().values
        self.mean_response = grating.mean().values

    def get_dm(self, resp_func):
        l = 13.5
        time = np.linspace(-10, 3.499, int(l * 1000))
        glm_events = pd.DataFrame({'time': time,
                                   'onset': time * 0,
                                   'offset': time * 0,
                                   'rt': time * 0,
                                   'grating': time * 0})
        glm_events.set_index('time', inplace=True)
        glm_events.loc[0:0.05, 'onset'] = 1
        glm_events.loc[2:2.05, 'offset'] = 1
        glm_events.loc[0:2, 'grating'] = 1
        glm_events.loc[0:2, 'grating'] /= glm_events.loc[0:2, 'grating'].sum()
        X = patsy.dmatrix('pt.MF(grating, resp_func) + pt.MF(onset, resp_func) + pt.MF(offset, resp_func) + 1', data=glm_events)
        X = X[9000:]
        return X

    def predict_tmax(self, tmax):
        IRFS = IRF_pupil(fs=1000, tmax=0.93 + tmax)
        self.Xb = np.vstack([self.get_dm(IRFS)] * self.n_trials)
        self.mdl = linear_model.LinearRegression()
        self.fit = self.mdl.fit(self.Xb, self.y)
        self.prediction = np.dot(self.get_dm(IRFS), self.fit.coef_)
        self.err = np.sum((self.mean_response - self.prediction)**2)
        return self.err, self.prediction


'''
fit = FitTmax(grating)
err = []
ts = []
for tmax in np.arange(-0.3, 0, 0.01):
    err.append(fit.predict_tmax(tmax))
    ts.append(tmax)
plt.plot(ts, err)
'''
