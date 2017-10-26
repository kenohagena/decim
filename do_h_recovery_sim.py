import numpy as np
import pointsimulation as pt


H = np.linspace(0.001, 0.999, 20)


def list_task(filter=None):
    for _ in range(150):
        yield(1000, H)


def execute(x):
    pt.h_iter(x[0], x[1])
