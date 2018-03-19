import glaze2 as gl
import numpy as np
import pandas as pd
import math
from scipy.special import erf
import pointsimulation as pt


def model_code():
    '''
    Returns stan modelling code for glaze model for multiple blocks in one session.
    '''
    glaze_code = """
    data {
        int<lower=0> B; // number of blocks
        int b[B+1]; // integer array with indices of last point locations of block
        int<lower=0> I; // number of decision trials
        int<lower=0> N; // number of point locations
        int<lower=0, upper=1> y[I]; // subjects answer (0 or 1)
        vector[N] x; // vector with N point locations
        int D[I]; // integer array with indices of decision point locations
    }
    parameters {
        real<lower=0.0001, upper=0.9999> H; //Hazardrate used in glaze
        real<lower=0.1> V; //Variance used in glaze
        real<lower=0.1, upper=9.9> gen_var; //Variance used in glaze
    }
    transformed parameters {
        real psi[N];
        real choice_value[N];
        real llr;

        for (i in 1:B) {
            llr = normal_lpdf(x[b[i]+1] | 0.5, 1/sqrt(25-gen_var*2.5)) - \
                              normal_lpdf(x[b[i]+1] | -0.5, 1/sqrt(25-gen_var*2.5));
            //print(llr);
            //print(gen_var);
            psi[b[i]+1] = llr;
            choice_value[b[i]+1] = 0.5+0.5*erf(psi[b[i]+1]/(sqrt(2)*1/(10-V)));

            for (n in (b[i]+2):b[i+1]) {

                llr = normal_lpdf(x[n] | 0.5, 1/sqrt(25-gen_var*2.5)) - \
                                  normal_lpdf(x[n] | -0.5, 1/sqrt(25-gen_var*2.5));
                psi[n] = psi[n-1] + log( (1 - H) / H + exp(-psi[n-1]))
                        - log((1 - H) / H + exp(psi[n-1]));

                psi[n] = (psi[n] + llr);
                choice_value[n] = 0.5+0.5*erf(psi[n]/(sqrt(2)*1/(10-V)));
                //print(choice_value[n]);
                }
        }
        //print("H: ", H, " gen_var: ", gen_var, " V: ", V)
    }

    model {
        H ~ uniform(0.0001, 0.9999); // T[0.0001,0.9999]; //prior on H from truncated normal
        V ~ normal(9, 0.5); //gamma(1.1, 10);  // Gamma centered on 1 covering until ~60
        gen_var ~ normal(9.6, 0.4); // Gamma centered on 1 covering until ~30
        for (i in 1:I) {
            y[i] ~ bernoulli(choice_value[D[i]]);
        }
    }
    """
    return glaze_code


def stan_data(subject, session, phase, blocks, path):
    '''
    Returns dictionary with data that fits requirement of stan model.

    Takes subject, session, phase, list of blocks and filepath.
    '''
    dflist = []
    lp = [0]
    for i in range(len(blocks)):
        d = gl.log2pd(gl.load_log(subject, session,
                                  phase, blocks[i], path), blocks[i])
        single_block_points = np.array(d.loc[d.message == 'GL_TRIAL_LOCATION'][
                                       'value'].index).astype(int)
        dflist.append(d)
        lp.append(len(single_block_points))
    df = pd.concat(dflist)
    df.index = np.arange(len(df))
    point_locs = np.array(df.loc[df.message == 'GL_TRIAL_LOCATION'][
                          'value']).astype(float)
    point_count = len(point_locs)
    decisions = np.array(df.loc[df.message == 'CHOICE_TRIAL_RULE_RESP'][
                         'value']).astype(float)
    # '-', '+1', because of mapping of rule response
    decisions = -(decisions[~np.isnan(decisions)].astype(int)) + 1
    dec_count = len(decisions)
    choices = (df.loc[df.message == "CHOICE_TRIAL_RULE_RESP", 'value']
               .astype(float))
    choices = choices.dropna()
    belief_indices = df.loc[choices.index - 12].index.values
    ps = df.loc[df.message == 'GL_TRIAL_LOCATION']['value'].astype(float)
    pointinds = np.array(ps.index)
    # '+1' because stan starts counting from 1
    dec_indices = np.searchsorted(pointinds, belief_indices) + 1
    data = {
        'I': dec_count,
        'N': point_count,
        'y': decisions,
        'x': point_locs,
        'D': dec_indices,
        'B': len(blocks),
        'b': np.cumsum(lp)
    }
    return data


def data_from_df(df):
    '''
    Returns dictionary with data that fits requirement of stan model.

    Takes subject, session, phase, list of blocks and filepath.
    '''
    decisions = df.loc[df.message == 'decision']
    points = df.loc[df.message == 'GL_TRIAL_LOCATION']
    data = {
        'I': len(decisions),
        'N': len(points),
        'y': decisions.choice.values,
        'x': points.value.values,
        'D': [int(j - (np.where(decisions.index.values == j)[0])) for j in decisions.index.values],
        'B': 1,
        'b': [0, len(points)]}
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
            b = gl.prior(belief[i - 1], parameters['H']) + gl.LLR(value, sigma=parameters['gen_var'])
            belief[i] = .5 + .5 * erf(b / math.sqrt(2 * parameters['V']))
    model_decs = belief[decision_indices]
    df = pd.DataFrame({'model': model_decs.astype(float), 'data': decisions.astype(float)})
    df['p'] = np.nan
    for i, row in df.iterrows():
        row.p = math.pow(row.model, row[0]) * math.pow((1 - row.model), (1 - row[0]))
    return math.log(df.product(axis=0)['p'])


def inv_transV(Vt):
    '''
    Takes transformed V, returns original parameter space.
    '''
    return 1 / (10 - Vt)


def inv_transGV(GVt):
    '''
    takes transformed generative variance, returns original.
    '''
    return 1 / (25 - GVt * 2.5)**.5


def transV(V):
    '''
    Input: original V
    Return: transformed
    '''
    return -1 / V + 10


def transGV(GV):
    '''
    Input: original gen_var
    Return: transformed
    '''
    return -1 / 2.5 * (1 / GV)**2 + 10


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


#sim = pt.complete(pt.fast_sim(1000, 1 / 70), 1 / 70)
#data = data_from_df(sim)

#print(likelihood(data, parameters={'V': 2, 'H': 1 / 10, 'gen_var': 1}))
