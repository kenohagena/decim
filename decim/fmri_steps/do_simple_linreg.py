import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from scipy.stats import linregress
import sys

subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            12, 14, 15, 16, 19, 20, 21]
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


results = []

roi_path = '/home/khagena/FLEXRULE/fmri/roi_extract/weighted/'
behav_path = '/home/khagena/FLEXRULE/behavior/behav_fmri_aligned/'

example = pd.read_csv(join(roi_path, '1_ses-2_inference_run-4_weighted_rois.csv'), index_col=0)
columns = example.columns


def execute(sub):
    for ses in ['ses-2', 'ses-3']:
        for ROI in columns:
            for param in parameters:
                try:
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

                    slope, intercept, r_value, p_value, std_err = linregress(behav,
                                                                             roi_z)
                    result = {'slope': slope, "rhat": r_value, 'p_value': p_value, 'intercept': intercept,
                              'std_err': std_err, 'subject': sub, 'session': ses, 'parameter': param, 'roi': ROI}
                    results.append(result)
                    print('success')
                except FileNotFoundError:
                    print('file not found')
    df = pd.DataFrame(results)
    df.to_csv(join(roi_path, 'linreg_sub-{}.csv'.format(sub)))


if __name__ == '__main__':
    execute(sys.argv[1])


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
