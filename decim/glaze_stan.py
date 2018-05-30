from decim import glaze2 as gl
import numpy as np
import pandas as pd
import math
from scipy.special import erf
from decim import pointsimulation as pt


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
    belief_indices = df.loc[choices.index].index.values
    ps = df.loc[df.message == 'GL_TRIAL_LOCATION']['value'].astype(float)
    pointinds = np.array(ps.index)
    # '+1' because stan starts counting from 1
    dec_indices = np.searchsorted(pointinds, belief_indices)
    data = {
        'I': dec_count,
        'N': point_count,
        'obs_decisions': decisions,
        'x': point_locs,
        'obs_idx': dec_indices,
        'B': len(blocks),
        'b': np.cumsum(lp)
    }
    return data


def data_from_df(df):
    '''
    Returns stan ready data dict from pointsimulation dataframe.
    '''
    decisions = df.loc[df.message == 'decision']
    data = {
        'I': len(decisions),
        'N': len(df),
        'obs_decisions': decisions.choice.values,
        'obs_idx': np.array(decisions.index.values).astype(int) + 1,
        'x': df.value.values,
        'B': 1,
        'b': np.array([0, len(df)]).astype(int)
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
            b = gl.prior(belief[i - 1], parameters['H']) + gl.LLR(value, sigma=parameters['gen_var'])
            belief[i] = .5 + .5 * erf(b / math.sqrt(2 * parameters['V']))
    model_decs = belief[decision_indices]
    df = pd.DataFrame({'model': model_decs.astype(float), 'data': decisions.astype(float)})
    df['p'] = np.nan
    for i, row in df.iterrows():
        row.p = math.pow(row.model, row[0]) * math.pow((1 - row.model), (1 - row[0]))
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


#sim = pt.complete(pt.fast_sim(1000, 1 / 70), 1 / 70)
#data = data_from_df(sim)

#print(likelihood(data, parameters={'V': 2, 'H': 1 / 10, 'gen_var': 1}))
