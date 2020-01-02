import numpy as np
import pandas as pd
import nibabel as nib
from surfer import Brain
from decim.adjuvant import slurm_submit as slu
from os.path import join
from glob import glob
from scipy.stats import ttest_1samp as ttest
from itertools import product
import matplotlib.pyplot as plt

'''
Script has three 'steps'

1. Get data
    a) import all regression results
    b) lateralize certain parameters (ispsi-kontra)
    c) compute t-test
    d) save p-values and t-statistics as .hdf file

2. Do pysurfer cortex plots
    a) import t-statistic and p_values
    b) do FDR-correction (Benjamini-Hochberg)
    c) do montage-plot (lateral and medial view) with pysurfer

3. Helper functions:
    a) give out list with significant correlated Glasser labels for a parameter
    b) plot certain Glasser labels on a random cortex

'''

hemis = {'L': 'lh', 'R': 'rh'}
mapping = {'C(stimulus, levels=s)[T.vertical]': 'stimulus_vertical',
           'C(stimulus, levels=s)[T.horizontal]': 'stimulus_horizontal',
           'C(response, levels=b)[T.left]': 'response_left',
           'C(response, levels=b)[T.right]': 'response_right',
           'C(rule_resp, levels=r)[T.A]': 'rule_resp_A',
           'C(rule_resp, levels=r)[T.B]': 'rule_resp_B',
           'C(stimulus, levels=s)[T.vertical]:C(rule_resp, levels=r)[T.A]': 'stimulus_vertical_rule_resp_A',
           'C(stimulus, levels=s)[T.horizontal]:C(rule_resp, levels=r)[T.A]': 'stimulus_horizontal_rule_resp_A',
           'C(stimulus, levels=s)[T.vertical]:C(rule_resp, levels=r)[T.B]': 'stimulus_vertical_rule_resp_B',
           'C(stimulus, levels=s)[T.horizontal]:C(rule_resp, levels=r)[T.B]': 'stimulus_horizontal_rule_resp_B',
           'C(response, levels=b)[T.left]:C(rule_resp, levels=r)[T.A]': 'response_left_rule_resp_A',
           'C(response, levels=b)[T.right]:C(rule_resp, levels=r)[T.A]': 'response_right_rule_resp_A',
           'C(response, levels=b)[T.left]:C(rule_resp, levels=r)[T.B]': 'response_left_rule_resp_B',
           'C(response, levels=b)[T.right]:C(rule_resp, levels=r)[T.B]': 'response_right_rule_resp_B',
           'belief': 'belief',
           'np.abs(belief)': 'abs_belief', 'switch': 'switch',
           'np.abs(switch)': 'abs_switch', 'LLR': 'LLR', 'np.abs(LLR)': 'abs_LLR',
           'surprise': 'surprise'}

'''
First: Get data
'''


def lateralize(x):
    '''
    Lateralization function

    Subtract ipsilateral beta-weight from contralateral beta-weight
    '''
    x = x.reset_index()
    left = x.query('hemisphere=="L"')
    right = x.query('hemisphere=="R"')
    del left['hemisphere']
    del right['hemisphere']
    left.set_index(['subject', 'parameter', 'names', 'labs'], inplace=True)
    right.set_index(['subject', 'parameter', 'names', 'labs'], inplace=True)
    if all(x.parameter.str.endswith('left')):
        x = left - right
    elif all(x.parameter.str.endswith('right')):
        x = right - left
    else:
        raise RuntimeError()
    return x


def concat(SJ_dir, task):
    '''
    Concatenate all regression results on the cortical surface.

    Ouput as pD.DataFrames with the columns:
        subject, session, parameter, hemisphere, cortical label, value
    '''
    design_matrix = pd.read_hdf(join(SJ_dir,
                                     'sub-17/DesignMatrix_sub-17_ses-3.hdf'),
                                key=task)                                       # load 'random' DesignMatrix to have the regressor names
    dfs = []
    for sub, hemi in product(range(1, 23), hemis.keys()):
        subject = 'sub-{}'.format(sub)
        aparc_file = join('/Volumes/flxrl/FLEXRULE/fmri/completed_preprocessed/',
                          subject, 'freesurfer', subject, 'label',
                          '{}.HCPMMP1.annot'.format(hemis[hemi]))
        try:
            labels, ctab, names = nib.freesurfer.read_annot(aparc_file)
        except FileNotFoundError:
            print('no aparc_file for ' + subject)
        for ses, param in product([2, 3], design_matrix.columns):
            file = join(SJ_dir, subject,
                        'SurfaceTxt_sub-{0}_ses-{1}_{2}_{3}.hdf'.
                        format(sub, ses, param, hemi))
            parameter = mapping[param]
            # print(task, parameter, subject, ses)                              # uncomment for debug
            session = 'ses-{}'.format(ses)
            try:
                df = pd.read_hdf(file, key=task)
            except (FileNotFoundError, KeyError) as e:
                print('No file found for {0}, {1}, {2}, {3}'.
                      format(subject, session, parameter, hemi))
                continue
            try:
                assert len(df) == len(labels)
            except AssertionError:
                print('len(df) does not match len(labels)')
            df['labs'] = labels
            str_names = [str(i) for i in names]
            str_names = [i[2:-1] if i == "b'???'" else i[4:-1]
                         for i in str_names]
            grouped = df.groupby('labs').mean().reset_index()
            grouped['names'] = str_names
            grouped['parameter'] = parameter
            grouped['subject'] = subject
            grouped['session'] = session
            grouped['hemisphere'] = hemi
            dfs.append(grouped)
    df = pd.concat(dfs, ignore_index=True)
    return df


def surface_glm_data(df, lateral_params, marker='coef_', output='t_stat'):
    '''
    T-test across average beta-weights and lateralized beta-weights.

    - Arguments:
        a) concatenated beta-weight data in a pd.DataFrame with the columns
            subject, session, parameter, hemisphere, cortical label, value
        b) parameters that shall be lateralized (e.g. response)
        c) columns name of beta-weight values (normally 'coef_')
        d) output t-statistic or p-value
    '''
    if output == 't_stat':
        p_or_t = 0
    elif output == 'p_val':
        p_or_t = 1
    ses_mean = df.groupby(['subject', 'parameter', 'names', 'hemisphere']).\
        mean().reset_index()                                                    # average across sessions per subject
    mag = ses_mean.groupby(['parameter', 'names', 'hemisphere']).\
        agg(lambda x: ttest(x, 0)[p_or_t]).reset_index()                        # t-test across across subjects
    mag = mag.groupby(['parameter', 'names']).mean().reset_index()              # average across hemispheres
    mag = mag.pivot(columns='parameter', index='names', values=marker)
    for lateral_param in lateral_params:
        if lateral_param[:8] == 'response':
            lat = ses_mean.loc[ses_mean.parameter.isin(['response_left',
                                                        'response_right'])]
        if lateral_param == 'rule_resp':
            lat = ses_mean.loc[ses_mean.parameter.isin(['rule_resp_A',
                                                        'rule_resp_B'])]
            lat = lat.replace({"parameter": {'rule_resp_A': 'rule_resp_left',
                                             'rule_resp_B': 'rule_resp_right'}})
        elif lateral_param in ['belief', 'LLR', 'switch']:                      # signed numerical parameters
            lat = ses_mean.loc[ses_mean.parameter == '{}'.format(lateral_param)]
            lat_opp = lat.copy()
            lat_opp[marker] = -lat_opp[marker]
            lat = lat.replace({"parameter":
                               {'{}'.format(lateral_param):
                                '{}_left'.format(lateral_param)}})
            lat_opp = lat_opp.replace({"parameter":
                                       {'{}'.format(lateral_param):
                                        '{}_right'.format(lateral_param)}})
            lat = pd.concat([lat, lat_opp], ignore_index=True)
        # print(lat.parameter.unique())                                         # uncomment for debug
        lat = lat.groupby(['parameter'], group_keys=False).\
            apply(lateralize).reset_index()                                     # per parameter apply laterlization function
        lat = lat.groupby(['names', 'subject']).\
            mean().reset_index()
        lat = lat.groupby(['names']).agg(lambda x: ttest(x, 0)[p_or_t]).\
            reset_index()                                                       # t-test acorss subjects
        mag['{}_lat'.format(lateral_param)] = lat[marker].values
    return mag


def surface_data(SJ_dir, lateral_params, task, exclude=['sub-11', 'sub-20']):
    '''
    Load, concat, laterlize and t-test data
    Save as two pd.Dataframes with p-values and t-statistic respectively
    '''
    out_dir = join(SJ_dir, 'GroupLevel')
    slu.mkdir_p(out_dir)
    grouped = concat(SJ_dir, task)
    grouped = grouped.loc[~grouped.subject.isin(exclude)]
    # print(grouped.subject.unique())
    t_stat = surface_glm_data(grouped, lateral_params, output='t_stat')
    p_vals = surface_glm_data(grouped, lateral_params, output='p_val')
    t_stat.to_hdf(join(out_dir, 'Surface_{}.hdf'
                       .format(task)), key='t_statistic')
    p_vals.to_hdf(join(out_dir, 'Surface_{}.hdf'.
                       format(task)), key='p_values')


'''
Second: Do Cortex Plots & FDR-Correction
'''


def get_data(task, in_dir):
    '''
    Get data from .hdf files with p-values and t-statistic
    '''
    glasser_labels = join('/Volumes/flxrl/FLEXRULE/fmri/completed_preprocessed/',
                          'sub-10/freesurfer/fsaverage/label/lh.HCPMMP1.annot')
    glasser_labels_comb = join('/Volumes/flxrl/FLEXRULE/fmri/completed_preprocessed/',
                               'sub-10/freesurfer/fsaverage/label/lh.HCPMMP1_combined.annot')
    labels, ctab, names = nib.freesurfer.read_annot(glasser_labels)
    labels_comb, ctab, names_combined = nib.freesurfer.\
        read_annot(glasser_labels_comb)
    str_names_comb = [str(i)[2:-1] for i in names_combined]
    str_names = [str(i) for i in names]
    str_names = [i[2:-1] if i == "b'???'" else i[4:-1] for i in str_names]
    full_to_comb = pd.DataFrame({'full': labels,
                                 'combined': labels_comb}).\
        groupby('full').mean().reset_index()
    combined = np.array(str_names_comb)[full_to_comb.combined.
                                        astype(int).values]
    t_data = pd.read_hdf(join(in_dir, 'Surface_{}.hdf'.
                              format(task)), key='t_statistic')
    p_data = pd.read_hdf(join(in_dir, 'Surface_{}.hdf'.
                              format(task)), key='p_values')
    t_data = t_data.loc[str_names]
    p_data = p_data.loc[str_names]
    t_data.iloc[0] = 0
    return t_data, p_data, [str_names, combined], labels


def fdr_filter(t_data, p_data, parameter):
    '''
    Apply FDR-correction
    '''
    data = t_data[parameter].values
    filte = benjamini_hochberg(p_data[parameter], 0.05).values
    data[filte != True] = 0
    return data


def benjamini_hochberg(pvals, alpha):
    '''
    Compute Benjamini-Hochberg FDR-Correction
    '''
    p_values = pd.DataFrame({'p': pvals, 'index': np.arange(0, len(pvals))})
    p_values = p_values.sort_values(by='p')
    p_values['rank_'] = np.arange(0, len(p_values)) + 1
    p_values['q'] = (p_values.rank_ / len(p_values)) * alpha
    p_values['reject'] = p_values.p < p_values.q
    thresh = p_values.loc[p_values.reject == True].rank_.max()
    p_values['reject'] = p_values.rank_ <= thresh
    return p_values.sort_values(by='index').reject


def montage_plot(parameter, in_dir, task, fdr_correct=True):
    '''
    Make plots for parameter on the cortical surface using pysurf module

    - Arguments:
        a) parameter
        b) output directory
        c) task (inference or instructed)
        d) FDR correction (boolean)
    '''
    out_dir = join(in_dir, 'pysurf_plots')
    slu.mkdir_p(out_dir)
    fsaverage = "fsaverage"
    hemi = "lh"
    surf = "inflated"
    t_data, p_data, str_names, labels = get_data(task, in_dir)
    if fdr_correct is True:
        data = fdr_filter(t_data, p_data, parameter)
    else:
        data = t_data[parameter].values
    data = data[labels]
    brain = Brain(fsaverage, hemi, surf,
                  background="white", title=parameter + task)
    brain.add_data(data, -10, 10, thresh=None, colormap="RdBu_r", alpha=.8)
    brain.save_imageset(join(out_dir, parameter + '_' + task),
                        ['lateral', 'medial'], colorbar=None)


'''
Third: Helper Functions
'''


def plot_single_roi(rois, views=['lateral']):
    '''
    Function to plot location of an array of Glasser labels in a specified view
    '''
    in_dir = '/Volumes/flxrl/FLEXRULE/Workflow/Sublevel_GLM_Climag_2018-12-21/GroupLevel/'   # need random data
    fsaverage = "fsaverage"
    hemi = "lh"
    surf = "inflated"
    t, p, str_names, labels = get_data('inference',
                                       in_dir=in_dir)
    df = pd.DataFrame({'labs': str_names[0],
                       't': 0})
    df = df.set_index('labs')
    df.loc[rois, 't'] = -4
    data = df.t.values
    data = data[labels]
    brain = Brain(fsaverage, hemi, surf,
                  background="white", views=views)
    brain.add_data(data, -10, 11, thresh=None, colormap="RdBu_r", alpha=.9)
    f, ax = plt.subplots()
    ax = plt.imshow(brain.save_montage(None,
                                       ['lateral'], colorbar=None))
    plt.show()


def roi_names_param(parameter, task, correlation, in_dir):
    '''
    Returns list of labels that show significant correlation with certain parameter

    - Arguments:
        a) parameter
        b) task
        c) correlation ('pos' or 'neg')
        d) inout directory
    '''
    t_data, p_data, str_names, labels = get_data(task, in_dir)
    data = fdr_filter(t_data, p_data, parameter)
    df = pd.DataFrame({'labs': str_names[0],
                       'combined': str_names[1],
                       't': data})
    if correlation[0:3] == 'pos':
        df = df.loc[df.t > 0]
    elif correlation[0:3] == 'neg':
        df = df.loc[df.t < 0]
    return df


# USAGE OF THIS SCRIPT

glm_run = 'Sublevel_GLM_Climag_2019-12-16'                                      # Specify where the GLM results are
# 1. Make t- and p-maps

'''
surface_data(join('/Volumes/flxrl/FLEXRULE/Workflow/', glm_run),
             ['response', 'switch'], 'instructed')
surface_data(join('/Volumes/flxrl/FLEXRULE/Workflow/', glm_run),
             ['response', 'belief', 'switch', 'LLR'], 'inference')
'''
# 2. Do plotting

for task in ['instructed', 'inference']:
    in_dir = join('/Volumes/flxrl/FLEXRULE/Workflow', glm_run, 'GroupLevel')
    cols = pd.read_hdf(join('/Volumes/flxrl/FLEXRULE/Workflow', glm_run, 'GroupLevel',
                            'Surface_{}.hdf'.format(task)), key='t_statistic').columns
    for param in cols:
        montage_plot(param, in_dir=in_dir, task=task, fdr_correct=True)

# 3. Miscellaneous functions
'''
roi_names_param('response_left_multi', 'inference', 'neg')
plot_single_roi('AAIC_ROI')
'''
