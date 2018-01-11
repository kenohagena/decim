import pystan
import numpy as np


# no prior so far
glaze_code = """
data {
    int<lower=0> I; // number of decision trials
    int<lower=0> N; // number of point locations
    int<lower=0, upper=1> y[I]; // subjects answer (0 or 1)
    vector[N] x; // vector with N point locations
    int D[I]; // integer array with indices of decision point locations
}
parameters {
    real<lower=0.0001, upper=0.9999> H; //Hazardrate used in glaze
}
transformed parameters {
    real former_belief = 0.0; //belief before first point location
    real psi[N];
    real llr;
    llr = normal_lpdf(x[1] | 0.5, gen_var) - normal_lpdf(x[1] | -0.5, gen_var);
    psi[1] = llr;

    for (n in 2:N) {
        
        llr = normal_lpdf(x[n] | 0.5, 1) - normal_lpdf(x[n] | -0.5, 1);
        psi[n] = psi[n-1] + log((1 - H) / H + exp(-psi[n-1]))
                - log((1 - H) / H + exp(psi[n-1]));
        psi[n] = psi[n] + llr;
        
        }
    }

model {
    H ~ normal(0.5,1)T[0.0001,0.9999]; //prior on H from truncated normal
    V ~ normal(1,1)T[0.000001, 100]; //prior on internal noise parameter
    gen_var ~ normal(1,1)T[0.000001, 100];
    for (i in 1:I) {
        choice_value = 0.5+0.5*erf(psi[D[i]]/sqrt(2*V));
        y[i] ~ bernoulli_logit(choice_value);
  }
}
"""


def run():
    dat = {'I': 2,  # int
           'N': 10,  # int
           'y': [0, 1],  # int array
           'x': [-2.1, -0.5, -0.2, 0.2, 0.1, -0.1,
                 1.4, 0.5, 0.9, 0.2],  # real array
           'D': [4, 8]}

    sm = pystan.StanModel(model_code=glaze_code)
    fit = sm.sampling(data=dat)
    return fit
