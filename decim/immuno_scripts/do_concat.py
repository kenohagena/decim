import pandas as pd
import numpy as np


def sessions():
    for sub in [1, 2, 3, 4, 6, 7, 9]:
        for ses in [1, 2, 3]:
            yield sub, ses


s = []
for sub, ses in sessions():
    df = pd.read_csv('immuno/choiceframes200218/cpf_{0}{1}.csv'.format(sub, ses), header=[0, 1, 2], index_col=[0, 1], dtype=np.float64)
    df['subject'] = sub
    df['session'] = ses
    df = df.set_index(['subject', 'session'], append=True)
    df = df.reorder_levels([2, 3, 0, 1], axis=0)
    s.append(df)

dfbig = pd.concat(s)
dfbig.to_csv('immuno/choiceframes200218/cpf_all.csv', index=True)
