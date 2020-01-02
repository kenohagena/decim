import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from scipy.stats import linregress
import sys
from itertools import product


runs = ['inference_run-4', 'inference_run-5', 'inference_run-6']
comb_ROIS = ['23_inside_lh', '19_cingulate_anterior_prefrontal_medial_lh',
             '11_auditory_association_lh', '03_visual_dors_lh',
             '22_prefrontal_dorsolateral_lh', '10_auditory_primary_lh',
             '02_visual_early_lh', '21_frontal_inferior_lh',
             '17_parietal_inferior_lh', '12_insular_frontal_opercular_lh',
             '14_lateral_temporal_lh', '05_visual_lateral_lh',
             '13_temporal_medial_lh', '20_frontal_orbital_polar_lh',
             '07_paracentral_lob_mid_cingulate_lh', '18_cingulate_posterior_lh',
             '09_opercular_posterior_lh', '08_premotor_lh', '01_visual_primary_lh',
             '06_somatosensory_motor_lh', '16_parietal_superior_lh',
             '15_temporal_parietal_occipital_junction_lh', '04_visual_ventral_lh',
             '23_inside_rh', '19_cingulate_anterior_prefrontal_medial_rh',
             '11_auditory_association_rh', '03_visual_dors_rh',
             '22_prefrontal_dorsolateral_rh', '10_auditory_primary_rh',
             '02_visual_early_rh', '21_frontal_inferior_rh',
             '17_parietal_inferior_rh', '12_insular_frontal_opercular_rh',
             '14_lateral_temporal_rh', '05_visual_lateral_rh',
             '13_temporal_medial_rh', '20_frontal_orbital_polar_rh',
             '07_paracentral_lob_mid_cingulate_rh', '18_cingulate_posterior_rh',
             '09_opercular_posterior_rh', '08_premotor_rh', '01_visual_primary_rh',
             '06_somatosensory_motor_rh', '16_parietal_superior_rh',
             '15_temporal_parietal_occipital_junction_rh', '04_visual_ventral_rh',
             'AAN_DR', 'basal_forebrain_4', 'basal_forebrain_123', 'LC_Keren_2std',
             'LC_standard', 'NAc', 'SNc', 'VTA']
parameters = ['belief', 'murphy_surprise', 'switch', 'point', 'response',
              'response_left', 'response_right', 'stimulus_horiz', 'stimulus_vert',
              'stimulus', 'rresp_left', 'rresp_right', 'abs_belief']
roi_path = '/home/khagena/FLEXRULE/fmri/roi_extract/weighted/'
behav_path = '/home/khagena/FLEXRULE/behavior/behav_fmri_aligned/'
example = pd.read_csv(join(roi_path, '1_ses-2_inference_run-4_weighted_rois.csv'), index_col=0)
columns = example.columns


def linreg(subs, sessions, ROIs, params, roi_path, behav_path):
    '''
    Sub, ses, ROI and param have to be lists.
    Perform simple linear regression per session, subject, beh. param. & ROI.
    Return pd.DataFrame if multiple regressions are performed.
    '''
    results = []
    for sub, ses, ROI, param in product(subs, sessions, ROIs, params):
        subject = 'sub-{}'.format(sub)
        roi_zs = []
        behavs = []
        for run in runs:
            roi = pd.read_csv(join(roi_path, '{0}_{1}_{2}_weighted_rois.csv'.format(sub, ses, run)),
                              index_col=0)
            behav = pd.read_csv(join(behav_path, 'beh_regressors_{0}_{1}_{2}'.format(subject, ses, run)),
                                index_col=0)
            behav = behav[param].values
            roi = roi[ROI].values
            if len(roi) > len(behav):
                roi = roi[0: len(behav)]
            elif len(roi) < len(behav):
                behav = behav[0:len(roi)]
            roi_zs.append(pd.Series(roi))
            behavs.append(pd.Series(behav))
        behav = pd.concat(behavs, ignore_index=True)
        roi_z = pd.concat(roi_zs, ignore_index=True)
        slope, intercept, r_value, p_value, std_err = linregress(behav, roi_z)
        result = {'slope': slope, "rhat": r_value, 'p_value': p_value, 'intercept': intercept,
                  'std_err': std_err, 'subject': sub, 'session': ses, 'parameter': param, 'roi': ROI}
        results.apend(result)
    if len(results) < 2:
        return results[0]
    else:
        df = pd.DataFrame(results)
        return df


def import_regressions(path, identifier):
    '''
    Load regression results and apply fisher transform to rhat.
    '''
    dfs = []
    files = glob(join(path, identifier))
    for file in files:
        df = pd.read_csv(file, index_col=[0])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df['rhat_fisher'] = np.arctanh(df.rhat)
    df = df.loc[df.subject != 13]
    df.loc[df.roi.isin(['???', '???.1']), 'roi'] =\
        df.loc[df.roi.isin(['???', '???.1']), 'roi'].map({'???': 'L_???', '???.1': 'R_???'})
    return df


def lat_mag(df, parameters):
    '''
    Average per subject across sessions, inclusde only cortical ROIs.
    '''
    response = df.loc[df.parameter.isin(parameter)]
    resp_ses = response.groupby(['subject', 'parameter', 'roi']).mean().reset_index()
    resp_ses_cort = resp_ses.loc[~resp_ses.roi.isin(brainstem)]
    resp_ses_cort['roi_'] = [i[2:] for i in resp_ses_cort.roi]
    '''
    1. One sample T-test against 0 of rhat magnitudes across subjects.
    2. Average per ROI across 'conditions' (i.e. response left and reponse right)
    '''
    resp_mag = {}
    for reg in resp_ses_cort.roi.unique():
        resp_mag[reg] = ttest(resp_ses_cort.loc[resp_ses_cort.roi == reg].rhat_fisher, 0)[0]

    resp_mag = pd.DataFrame.from_dict(resp_mag, orient='index').reset_index()
    resp_mag.columns = ['roi', 't_statistic']
    resp_mag['roi_'] = [i[2:] for i in resp_mag.roi]
    resp_mag = resp_mag.groupby('roi_').mean().reset_index()
    '''
    1. Replace NaNs with zeros.
    2. Difference between lh and rh per subject, ROI and subject.
    3. Difference / 2 of these between conditions
    4. T-test againsgt 0 of difference across subjects.
    5. Abs() of T-statistic
    '''
    resp_ses_cort = resp_ses_cort.fillna(0)
    resp_cort_lat = resp_ses_cort.groupby(['subject', 'roi_', 'parameter']).agg(lambda x: np.diff(x)).reset_index()
    resp_cort_lat = resp_cort_lat.groupby(['subject', 'roi_']).agg(lambda x: np.diff(x) / 2).reset_index()

    resp_lat = {}
    for reg in resp_cort_lat.roi_.unique():
        tt = ttest(resp_cort_lat.loc[resp_cort_lat.roi_ == reg].rhat_fisher, 0)
        resp_lat[reg] = tt[0]
    resp_lat = pd.DataFrame.from_dict(resp_lat, orient='index').reset_index()
    resp_lat.columns = ['roi_', 't_statistic']
    resp_lat.t_statistic = resp_lat.t_statistic.abs()
    '''
    Save.
    '''
    resp_lat.to_csv('resp_lat.csv')
    resp_mag.to_csv('resp_mag.csv')


if __name__ == '__main__':
    pass


'''
import seaborn as sns
f, ax = plt.subplots(1, 4, figsize=(16, 3))

motorleft = ['02_visual_early_lh', '08_premotor_lh', '01_visual_primary_lh',
       '06_somatosensory_motor_lh']
motorright = ['02_visual_early_rh', '08_premotor_rh', '01_visual_primary_rh',
       '06_somatosensory_motor_rh']
motor_parameters = ['response_left', 'response_right']
motor = df.loc[df.roi.isin(motorleft + motorright) & df.parameter.isin(motor_parameters)]

sns.pointplot('parameter', 'rhat', hue='roi', data=motor.loc[motor.roi.isin(['01_visual_primary_lh',
                                                                                 '01_visual_primary_rh'])],
              dodge=True, ci = 'sd', palette = sns.color_palette("BrBG", 2), ax = ax[0])
sns.pointplot('parameter', 'rhat', hue = 'roi', data = motor.loc[motor.roi.isin(['02_visual_early_lh',
                                                                                 '02_visual_early_rh'])],
              dodge=True, ci = 'sd', palette = sns.color_palette("BrBG", 2), ax = ax[1])
sns.pointplot('parameter', 'rhat', hue = 'roi', data = motor.loc[motor.roi.isin(['08_premotor_lh',
                                                                                 '08_premotor_rh'])],
              dodge=True, ci = 'sd', palette = sns.color_palette("BrBG", 2), ax = ax[2])
sns.pointplot('parameter', 'rhat', hue = 'roi', data = motor.loc[motor.roi.isin(['06_somatosensory_motor_lh',
                                                                                 '06_somatosensory_motor_rh'])],
              dodge=True, ci = 'sd', ax = ax[3], palette = sns.color_palette("BrBG", 2), label=False)
ax[0].set_title('01_visual_primary_lh')
ax[1].set_title('02_visual_early_lh')
ax[2].set_title('08_premotor_lh')
ax[3].set_title('06_somatosensory_motor_lh')

for i in range(4):
    ax[i].legend().set_visible(False)
    ax[i].set_ylim([-.1, .5])
    ax[i].set_yticks([-.1, 0, .1, .2, .3, .4, .5])
sns.despine(trim=True)
'''
