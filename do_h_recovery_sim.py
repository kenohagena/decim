import numpy as np
import pointsimulation as pt


def list_tasks(older_than=None, filter=None):
    for i in range(10):
        for iters in [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200]:
            yield(i, 50, iters)


def execute(x):
    i, N, trials = x
    H = np.linspace(0.001, 0.999, 20)
    pt.h_iter(i, N, H, trials=trials)
