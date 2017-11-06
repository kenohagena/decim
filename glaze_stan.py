import pystan
import glaze2 as gl
import numpy as np
import pandas as pd

# no prior so far


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
        real<lower=0, upper=1> H; //Hazardrate used in glaze
    }
    transformed parameters {
        real psi[N];
        real llr;
        for (i in 1:B) {
            llr = normal_lpdf(x[b[i]+1] | 0.5, 1) - normal_lpdf(x[b[i]+1] | -0.5, 1);
            psi[b[i]+1] = llr;
            for (n in (b[i]+2):b[i+1]) {

                llr = normal_lpdf(x[n] | 0.5, 1) - normal_lpdf(x[n] | -0.5, 1);
                psi[n] = psi[n-1] + log((1 - H) / H + exp(-psi[n-1]))
                        - log((1 - H) / H + exp(psi[n-1]));
                psi[n] = psi[n] + llr;

                }


        }
    }

    model {
        H ~ uniform(0,1); //prior on H from truncated normal
        for (i in 1:I) {
            y[i] ~ bernoulli_logit((psi[D[i]]));
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
        d = gl.log2pd(gl.load_log(subject, session, phase, blocks[i], path), blocks[i])
        single_block_points = np.array(d.loc[d.message == 'GL_TRIAL_LOCATION']['value'].index).astype(int)
        dflist.append(d)
        lp.append(len(single_block_points))
    df = pd.concat(dflist)
    point_locs = np.array(df.loc[df.message == 'GL_TRIAL_LOCATION']['value']).astype(float)
    point_count = len(point_locs)
    decisions = np.array(df.loc[df.message == 'CHOICE_TRIAL_RULE_RESP']['value']).astype(float)
    decisions = -(decisions[~np.isnan(decisions)].astype(int)) + 1  # '-', '+1', because of mapping of rule response
    dec_count = len(decisions)
    choices = (df.loc[df.message == "CHOICE_TRIAL_RULE_RESP", 'value']
               .astype(float))
    choices = choices.dropna()
    belief_indices = df.loc[choices.index - 12].index.values
    ps = df.loc[df.message == 'GL_TRIAL_LOCATION']['value'].astype(float)
    pointinds = np.array(ps.index)
    dec_indices = np.searchsorted(pointinds, belief_indices) + 1  # '+1' because stan starts counting from 1
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


__version__ = '1.1'
'''
1.1
Fits hazardrate over multiple blocks.
'''
