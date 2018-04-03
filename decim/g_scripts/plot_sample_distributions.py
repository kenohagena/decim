import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


for subject in [1, 2, 3, 4, 6, 7, 9]:
    # DATA INPUT
    dfs = []
    summary = []
    for ses in ['A', 'B', 'C']:
        df = pd.read_csv('/Users/kenohagena/Documents/immuno/da/decim/g_behav/gl_in_fits090318/samples_VPIM0{0}{1}'.format(subject, ses))
        df['session'] = ses
        dfs.append(df)
        s = pd.read_csv('/Users/kenohagena/Documents/immuno/da/decim/g_behav/gl_in_fits090318/summary_VPIM0{0}{1}'.format(subject, ses))
        s['session'] = ses
        summary.append(s)
    df = pd.concat(dfs, ignore_index=True)

    H = df.loc[:, ['session', 'H']]
    H['parameter'] = 'H'
    H.columns = ['session', 'samples', 'parameter']
    V = df.loc[:, ['session', 'V']]
    V['parameter'] = 'V'
    V.columns = ['session', 'samples', 'parameter']
    gen_var = df.loc[:, ['session', 'gen_var']]
    gen_var['parameter'] = 'gen_var'
    gen_var.columns = ['session', 'samples', 'parameter']

    df = pd.concat([H, V, gen_var], ignore_index=True)
    summary = pd.concat(summary, ignore_index=False)

    # PLOT
    g = sns.FacetGrid(df, col="parameter", row="session", sharex=False, sharey=False, margin_titles=True)
    g = g.map(plt.hist, "samples", bins=100)

    for axlist, session in zip(g.axes, ['A', 'B', 'C']):
        axlist[0].set_xlim([0, 1])
        axlist[0].set_ylim([0, 500])

        axlist[1].set_xlim([0, 6])
        axlist[1].set_ylim([0, 500])

        axlist[2].set_xlim([0, 1.2])
        axlist[2].set_ylim([0, 1000])

        for ax, col in zip(axlist, [0, 1, 2]):
            ax.set_yticks([])
            ax.annotate('Mode={}'.format('%.3f' % summary.loc[summary.session == session, '50%'][col]),
                        xy=(.4, 0.72),
                        xycoords='axes fraction',
                        fontsize=10)
            ax.annotate('SD={}'.format('%.3f' % summary.loc[summary.session == session, 'sd'][col]),
                        xy=(.4, 0.65),
                        xycoords='axes fraction',
                        fontsize=10)

    g.despine(left=True)
    g.set_xlabels('Parameter space')
    g.savefig('facetgrid_noise_H_{}.png'.format(subject), dpi=160)
