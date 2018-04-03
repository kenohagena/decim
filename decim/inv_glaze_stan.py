import math
import numpy as np
import pandas as pd

from decim import glaze2 as gl
from scipy.special import erf


def model_code():
    '''
    Returns stan modelling code for glaze model for multiple blocks in one session.
    '''
    glaze_code = """
    data {
        int<lower=0> I; // number of decision trials
        int<lower=0> N; // number of point locations
        vector[N] x; // vector with N point locations
        int obs_idx[I];   // integer array with indices of decision point locations
        int obs_decision[I];
        int mis_idx[N-I]; // missing decisions
        }

    parameters {
        real<lower=0, upper=1> H; //Hazardrate used in glaze
        real<lower=1> V; //Variance used in glaze
        real<lower=1> gen_var; //Variance used in glaze
        int mis_decisions[N-I];
    }

    transformed_parameters{
        int decisions[N];
        decisions[obs_idx] = obs_decisions;
        decisions[mis_idx] = mis_decisions;
    }

    transformed parameters {
        real psi[N];
        real choice_value[N];
        real llr;
        llr = normal_lpdf(x[1] | 0.5, gen_var) - \
                              normal_lpdf(x[1] | -0.5, gen_var);
        psi[1] = llr;
        for (i in 2:N) {
                llr = normal_lpdf(x[i] | 0.5, gen_var) - \
                                  normal_lpdf(x[i] | -0.5, gen_var);
                psi[i] = llr + (psi[i-1] + log( (1 - H) / H + exp(-psi[i-1]))
                        - log((1 - H) / H + exp(psi[i-1])));
        }
    }

    model {
        H ~ normal(0.5, 100);
        V ~ normal(1, 20);
        gen_var ~ normal(1, 20);

        for (i in 1:N) {
            decisions[i] ~ bernoulli(inv_logit(psi/V));
        }
    }
    """
    return glaze_code


def data_from_df(df):
    '''
    Returns dictionary with data that fits requirement of stan model.

    Takes subject, session, phase, list of blocks and filepath.
    '''
    decisions = df.loc[df.message == 'decision']
    points = df.loc[df.message == 'GL_TRIAL_LOCATION']
    # Compute decision locations in data stream
    point_idx = points.index.values
    obs_idx = [np.argmin(abs(point_idx - (idx - 1)))
               for idx in decisions.index.values]
    mis_idx = list(set(range(1, len(points))) - set(obs_idx))
    data = {
        'I': len(decisions),
        'N': len(points),
        'obs_decisions': decisions.choice.values,
        'obs_idx': np.array(obs_idx),
        'mis_idx': np.array(mis_idx),
        'x': points.value.values,
    }
    return data


def likelihood(data, parameters):
    '''
    Return likelihood of data given parameter values.

    Data in the format for stan model, parameters is a dictionary.
    Just for one block...
    '''
    points = np.array(data['x'])
    decisions = np.array(data['y'])
    decision_indices = np.array(data['D']) - 1
    belief = 0 * points
    for i, value in enumerate(points):
        if i == 0:
            b = gl.LLR(value, sigma=parameters['gen_var'])
            belief[i] = .5 + .5 * erf(b / math.sqrt(2 * parameters['V']))
        else:
            b = gl.prior(belief[i - 1], parameters['H']) + \
                gl.LLR(value, sigma=parameters['gen_var'])
            belief[i] = .5 + .5 * erf(b / math.sqrt(2 * parameters['V']))
    model_decs = belief[decision_indices]
    df = pd.DataFrame({'model': model_decs.astype(
        float), 'data': decisions.astype(float)})
    df['p'] = np.nan
    for i, row in df.iterrows():
        row.p = math.pow(row.model, row[0]) * \
            math.pow((1 - row.model), (1 - row[0]))
    return math.log(df.product(axis=0)['p'])


__version__ = '2.0'
'''
1.1
Fits hazardrate over multiple blocks.
1.2
HDI functon
1.2.1
added reindex line in stan_data to account for datasets that were interrupted
between blocks and started a new index from that point onwards
2.0
-model contains internal noise parameters
-deleted mode, hdi functions
'''


# sim = pt.complete(pt.fast_sim(1000, 1 / 70), 1 / 70)
# data = data_from_df(sim)

# print(likelihood(data, parameters={'V': 2, 'H': 1 / 10, 'gen_var': 1}))
