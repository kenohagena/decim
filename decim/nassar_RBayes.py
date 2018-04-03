'''
Implements the reduced bayesian model from Nassar et al.
'''
import numpy as np
import pandas as pd
from scipy.stats import norm


class RBayes(object):

    def __init__(self, generative_sd, H, range=(0, 300)):
        self.generative_sd = float(generative_sd)
        self.H = H
        self.rl = .5
        self.range = range
        self.Ux = H * 1. / (range[1] - range[0])
        self.mu_est = 0
        self.history = []
        self.cnt = 0
        self.cols = []

    @property
    def sigma(self):  # actually squared sigma
        return self.generative_sd**2 + (self.rl * self.generative_sd**2) / (1 - self.rl)

    def CPP(self, X):
        Nx = norm.pdf(X, self.mu_est, self.sigma)
        UxH = self.Ux
        return UxH / (UxH + (Nx * (1 - self.H)))

    def nassar_update(self, X):
        cpp = self.CPP(X)
        #alpha = (1 + cpp * self.rl) / (self.rl + 1.)
        alpha = self.rl + (1 - self.rl) * cpp
        delta = X - self.mu_est
        self.rl = self.tau(X)  # (self.rl + 1) * (1 - cpp) + cpp
        self.mu_est = self.mu_est + alpha * delta

        self.history.append({'mu': self.mu_est, 'sigma': self.sigma,
                             'rl': self.rl, 'cpp': cpp, 'alpha': alpha,
                             'positive': self.ispositive()})
        x = norm.pdf(np.arange(300), self.mu_est, self.sigma)
        cutoff = norm.pdf(self.mu_est + 3 * self.sigma,
                          self.mu_est, self.sigma)
        x[x < cutoff] = np.nan
        self.cols.append(x)
        return self

    def tau(self, X):
        Nsq = self.generative_sd**2
        omega = self.CPP(X)
        tau = self.rl
        mu = self.mu_est
        a = Nsq * omega + \
            (1 - omega) * (tau * Nsq) +\
            omega * (1 - omega) * (tau + mu * (1 - tau) - X)
        b = a + Nsq
        return a / b

    def predictive_distribution(self):
        return norm(loc=self.mu_est, scale=self.sigma)

    def ispositive(self):
        return 1 - self.predictive_distribution().cdf(0)


def make_sequence(H, sigma, length, range=(-100, 100)):
    from scipy.stats import expon
    exp_blocks = length / H
    ibi = list(expon.rvs(scale=1. / H, size=int(5 * exp_blocks)))
    ibi = [max(min(i, 5), 2 * 1. / H) for i in ibi]
    n = []
    while len(n) < length:
        n.extend(norm.rvs(np.random.randint(*range), sigma, size=int(ibi.pop())))
    return n[:length]


#print(make_sequence(0.05, 5, 100, range=(0, 300)))
#rb = RBayes(10, .05)
