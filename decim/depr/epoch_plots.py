import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
from glob import glob
import seaborn as sns
from scipy.stats import linregress
from decim import slurm_submit as slu
import mne


diverging = sns.diverging_palette(10, 220, sep=80, n=7)

rois = ['AAN_DR', 'basal_forebrain_4', 'basal_forebrain_123', 'LC_standard', 'NAc', 'SNc', 'VTA']
subjects = ['sub-1', 'sub-2', 'sub-3', 'sub-4', 'sub-5', 'sub-6', 'sub-7',
            'sub-8', 'sub-9', 'sub-10', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
            'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22']
out_dir = '/Users/kenohagena/Flexrule/fmri/plots/epoch_plots_new'
slu.mkdir_p(out_dir)
'''
INPUT
'''


def concat_sub_epochs(SL_dir):
    '''
    Input: directory of SubjectLevel data
    Output: concatenated DF
    '''
    files = glob(join(SL_dir, '*', 'CleanEpochs*'))
    dfs = []
    for file in files:
        for session in ['ses-2', 'ses-3']:
            df = pd.read_hdf(file, key=session)
            df['session'] = session
            df['session'] = file[file.find('ses-'):file.find('ses-') + 5]
            df['subject'] = file[file.find('sub-'):file.find('/Clean')]
            dfs.append(df)
    concatenated = pd.concat(dfs, ignore_index=True)
    concatenated.to_hdf('FLEXRULE/Workflow/Sublevel_ChoiceEpochs_Climag_2019-11-06/ChoiceEpochs_Climag_2019-11-06_concat.hdf', key='concat')
    return concatenated


def data(clean_df):
    summary = pd.read_csv('/Users/kenohagena/Flexrule/fmri/analyses/bids_stan_fits/summary_stan_fits.csv', index_col=0)
    linresult = linregress(clean_df.behavior.parameters.rt.values, clean_df.pupil.parameters.TPR.values)
    rt_predicted_tpr = clean_df.behavior.parameters.rt.values * linresult[0] + linresult[1]
    clean_df.loc[:, ('pupil', 'parameters', 'TPR_rt_corr')] = rt_predicted_tpr
    clean_df.loc[:, ('pupil', 'parameters', 'TPR_rt_corr')] = clean_df.pupil.parameters.TPR - clean_df.pupil.parameters.TPR_rt_corr
    clean_df.loc[:, ('pupil', 'parameters', 'tpr_bin')] = pd.qcut(clean_df.pupil.parameters.TPR_rt_corr, [0, .4, .6, 1.], labels=['low', 'med', 'high']).values
    d = clean_df.reset_index().groupby('subject').count().trial_id
    count_series = pd.Series(np.zeros(20), index=subjects)
    count_series[d.index] = d.values
    return clean_df, count_series, linresult, summary


def tpr_rt_scatter(clean_df, linresult):
    f, ax = plt.subplots()
    ax.scatter(clean_df.behavior.parameters.rt, clean_df.pupil.parameters.TPR.values, color=diverging[0], alpha=.3)
    x = np.array([clean_df.behavior.parameters.rt.min(), clean_df.behavior.parameters.rt.max()])
    ax.plot(x, x * linresult[0] + linresult[1], color='black', lw=3, alpha=.8)
    ax.set(xlim=[0, 2], xticks=[.25, .5, .75, 1, 1.25, 1.5, 1.75, 2], xticklabels=['', .5, '', 1, '', 1.5, '', 2],
           ylim=[-4, 4], yticks=np.arange(-4, 5, 2), xlabel='Reaction Time (s)', ylabel='TPR')
    ax.annotate('R=%.2f, p=%.2f' % (linresult[2], linresult[3]), xy=(1.5, -3))
    sns.despine(trim=True)
    f.savefig(join(out_dir, 'TRP_RT_scatter.png'), dpi=160)


def pupil_response(clean_df):
    f, ax = plt.subplots()
    high = clean_df.loc[clean_df.pupil.parameters.tpr_bin == 'high', ('pupil', 'choicelock')]
    low = clean_df.loc[clean_df.pupil.parameters.tpr_bin == 'low', ('pupil', 'choicelock')]
    for i, row in high.iterrows():
        plt.plot(row.values, color=diverging[1], alpha=.1)
    for i, row in low.iterrows():
        plt.plot(row.values, color=diverging[5], alpha=.1)
    plt.plot(high.mean().values, color=diverging[0], lw=4, label='high TPR')
    plt.plot(low.mean().values, color=diverging[6], lw=4, label='low TPR')
    plt.legend()
    ax.set(ylim=[-5, 5], xticks=np.arange(0, 3000, 500), xticklabels=np.arange(-1, 2, .5),
           xlabel='Time from Response (m)', ylabel='Pupil')
    sns.despine(trim=True)
    f.savefig(join(out_dir, 'Pupil_response_trials_binned.png'), dpi=160)


def ROI_pupil_binned(clean_df):
    f, ax = plt.subplots(2, 4, figsize=(12, 6))
    f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                      wspace=.8, hspace=.5)
    for roi, i in zip(rois, range(7)):
        if i < 4:
            j = 0
            k = i
        else:
            j = 1
            k = i - 4
        high = clean_df.loc[clean_df.pupil.parameters.tpr_bin == 'high', ('fmri', roi)].iloc[:, 0:14000]
        low = clean_df.loc[clean_df.pupil.parameters.tpr_bin == 'low', ('fmri', roi)].iloc[:, 0:14000]
        ax[j, k].plot(high.mean().values, color=diverging[0], lw=4)
        ax[j, k].plot(low.mean().values, color=diverging[6], lw=4)
        ax[j, k].set(yticks=[-.05, 0, .05], xticks=np.arange(2000, 15000, 4000), xticklabels=np.arange(0, 13, 4),
                     xlabel='Time from cue (s)', ylabel='fMRI response', title=roi)
        sns.despine(trim=True)
    ax[1, 3].plot([], [], color=diverging[0], label='high TPR')
    ax[1, 3].plot([], [], color=diverging[6], label='low TPR')
    ax[1, 3].legend()
    ax[1, 3].set(xticks=[], yticks=[])
    sns.despine(trim=True, bottom=True, left=True, ax=ax[1, 3])
    f.savefig(join(out_dir, 'ROIs_pupil_binned.png'), dpi=160)


def behavioral_pointplot(df):
    behavior = df.behavior.parameters.drop(['trial_id', 'run'], axis=1).reset_index()
    behav_group = behavior.groupby(['subject', 'session']).mean().reset_index()

    f, ax = plt.subplots(1, 2, figsize=(8, 5))
    sns.pointplot('session', 'reward', data=behav_group, hue='subject', ax=ax[0], palette='Blues')
    sns.pointplot('session', 'rt', data=behav_group, hue='subject', ax=ax[1], palette='Blues')

    ax[0].set(ylim=[.5, 1.0], yticks=[.6, .9],
              xlabel='Session', ylabel='Rewards', title="Rewards")
    ax[1].set(ylim=[.6, 1.1], yticks=[.7, 1],
              yticklabels=[600, 1000], xlabel='Session', ylabel='Reaction Time', title="Reaction Times")
    sns.despine(trim=True, offset=10)
    ax[0].legend('')
    ax[1].legend('')
    #ax[1].legend(loc=5, bbox_to_anchor=(1.5,0.5))
    f.savefig(join(out_dir, 'behavior_basic.png'), dpi=160)


def useable_trials(count_series):

    f, ax = plt.subplots(figsize=(8, 4.5))
    inds = np.arange(len(subjects))
    width = 0.8
    ax.bar(inds, count_series, width, color=diverging[6])

    for ind in inds:
        ax.text(ind, count_series[ind] + 5, "%.0f" % count_series[ind], horizontalalignment='center', fontdict={'fontsize': 10})

    ax.set(yticks=np.arange(0, 200, 50), ylim=[0, 170], xticks=np.arange(0, 21, 1),
           xticklabels=np.concatenate([np.arange(1, 11, 1), np.arange(12, 17, 1), np.arange(18, 23, 1)]),
           xlabel='Subject', ylabel='Clean trials', title='Trials usable after pupil preprocessing')
    sns.despine(trim=True, offset=5)
    f.savefig(join(out_dir, 'Usable_trials_barplot.png'), dpi=160)


def subjects_pupil(clean_df, count_series):
    good_subjects = count_series.loc[count_series > 30].index
    palette = sns.color_palette("GnBu_d", len(good_subjects))
    f, ax = plt.subplots()
    good = clean_df.reset_index().loc[clean_df.reset_index().subject.isin(good_subjects)]
    for sub, i in zip(good_subjects, range(len(good_subjects))):
        pupil = good.loc[good.subject == sub].pupil.choicelock.mean().values
        plt.plot(pupil, color=palette[i], lw=2)
    ax.set(ylim=[-3, 2.5], yticks=np.arange(-2, 4, 2), xticks=np.arange(0, 3000, 500), xticklabels=np.arange(-1, 2, .5),
           xlabel='Time from Response (m)', ylabel='Pupil', title='Pupil response-locked per subject')
    sns.despine(trim=True)
    f.savefig(join(out_dir, 'Subjects_pupil.png'), dpi=160)


def stan_fit_plot(summary):
    data = summary.loc[summary.subject != 'sub-11']
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.pointplot('session', 'hmode', data=data, hue='subject', ax=ax[0], palette='Blues')
    sns.pointplot('session', 'vmode', data=data, hue='subject', ax=ax[1], palette='Blues')

    ax[0].set(ylim=[-.01, .1], yticks=[0, .1], xlabel='Session', ylabel='Hazard rate', title="Hazard rate")
    ax[1].set(ylim=[.3, 7], yticks=[1, 7], xlabel='Session', ylabel='Internal noise', title="Internal noise")
    sns.despine(trim=True, offset=10)
    ax[0].legend('')
    ax[1].legend('')
    #ax[1].legend(loc=5, bbox_to_anchor=(1.5,0.5))
    f.savefig(join(out_dir, 'stan_fits.png'), dpi=160)


def fmri_bin(df, to_bin):
    df['bins'] = to_bin
    df = df.set_index('bins', append=True)
    stacked = df.stack(['type', 'name']).reset_index()
    stacked = stacked.loc[~stacked.name.isin(['onset', 'trial_id', 'run', '140'])]
    stacked.name = stacked.name.astype(int)
    stacked.columns = ['trial_id', 'run', 'subject', 'session',
                       'bins', 'type', 'name', 'b_values']
    stacked.b_values = stacked.b_values.astype(float)
    stacked['bins'] = pd.qcut(stacked.bins, [0, .4, .6, 1], labels=['low', 'mid', 'high'])
    grouped = stacked.groupby(['subject', 'type', 'name', 'bins']).mean().reset_index()
    data = grouped.loc[grouped.bins.isin(['high', 'low'])]
    data.bins = data.bins.astype('str')
    return data


def permutation_plot(fmri, pupil):
    data = fmri_bin(fmri, pupil.choicelock_defitted.mean(axis=1).values)

    sns.set(style='ticks')
    plt.rcParams['pdf.fonttype'] = 3
    plt.rcParams['ps.fonttype'] = 3
    sns.set(style='ticks', font_scale=1, rc={
        'axes.labelsize': 12,
        'axes.titlesize': 15,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
        'axes.linewidth': 1,
        'xtick.major.width': 1,
        'ytick.major.width': 1,
        'ytick.major.width': 1,
        'ytick.major.width': 1,
        'ytick.major.pad': 2.0,
        'ytick.minor.pad': 2.0,
        'xtick.major.pad': 2.0,
        'xtick.minor.pad': 2.0,
        'axes.labelpad': 4.0,
    })

    fg = sns.FacetGrid(data, col='type', col_wrap=4, hue='bins', palette=sns.diverging_palette(220, 20, n=2), sharex=False)
    fg.map(sns.lineplot, 'name', 'b_values', ci=95, lw=5).set_titles('{col_name}')
    fg.add_legend(loc=5, bbox_to_anchor=(.88, .3), title='TPR bin')
    fg.set(xlim=(0, 140), ylim=(-.06, .06),
           xticks=np.arange(20, 180, 60), yticks=[-.05, 0, .05],
           xticklabels=[0, 6, 12], xlabel='Time from cue (s)', ylabel='BOLD signal')
    fg.fig.subplots_adjust(wspace=.3, hspace=.7)
    sns.despine(trim=True, offset=10)

    '''
    Cluster permutations test.
    '''
    axes = fg.axes.flatten()
    for ax in axes:
        title = ax.get_title()
        ax.set_title(rois[title])
        m1 = pd.pivot_table(data.query('bins=="low" & type=="{}"'.format(title)), values='b_values', index='subject', columns='name').values
        p1 = pd.pivot_table(data.query('bins=="high" & type=="{}"'.format(title)), values='b_values', index='subject', columns='name').values
        delta_lc = p1 - m1
        t_tfce, _, p_tfce, H0 = mne.stats.permutation_cluster_1samp_test(delta_lc,
                                                                         threshold=dict(start=0, step=0.2))

        nosig = np.where(p_tfce > .05)[0]
        sig = np.where(p_tfce < .05)[0]
        ax.scatter(nosig, nosig * 0 - .05, color='grey', lw=3, marker='_')
        ax.scatter(sig, sig * 0 - 0.03, color='r', lw=3, marker='_')

    fg.savefig('/Volumes/flxrl/FLEXRULE/pupil/permutation_plot.png', dpi=160)
#df, clean_df, count_series, linresult, summary = data()
#tpr_rt_scatter(clean_df, linresult)
# pupil_response(clean_df)
# ROI_pupil_binned(clean_df)
# behavioral_pointplot(df)
# useable_trials(count_series)
#subjects_pupil(clean_df, count_series)
# stan_fit_plot(summary)


clean_df = concat_sub_epochs('/Volumes/flxrl/FLEXRULE/SubjectLevel/')
clean_df = clean_df.loc[clean_df.task == 'inference']
fmri = clean_df.fmri
pupil = clean_df.pupil
permutation_plot(fmri, pupil)
