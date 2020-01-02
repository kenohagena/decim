import numpy as np
import pandas as pd
import sympy


def IRF_pupil(fs=100, dur=4, s=1.0 / (10**26), n=10.1, tmax=.930):
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

    return y, y_dt, y_dn
