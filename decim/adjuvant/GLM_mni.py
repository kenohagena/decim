import numpy as np
import nibabel as nib
from os.path import join
from scipy.stats import ttest_1samp
from decim.adjuvant import slurm_submit as slu
from pymeg import parallel as pbs


def execute(h):
    subjects_exclude = [11, 20]
    subjects_include = ['sub-{}'.format(i) for i in range(1, 23) if i not in subjects_exclude]
    glm_run_path = '/home/khagena/FLEXRULE/Workflow/Sublevel_GLM_Climag_2020-01-26'
    out_dir = join(glm_run_path, 'GroupLevel')
    slu.mkdir_p(out_dir)

    for task, regressors in {'instructed': ['np.abs(switch)'],
                             'inference': ['np.abs(belief)', 'np.abs(LLR)', 'surprise']}.items():
        for regressor in regressors:
            t_test = []
            for subject in subjects_include:
                ses_mean = []
                for session in ['ses-2', 'ses-3']:
                    try:
                        nifti = nib.load(join(glm_run_path, subject, 'VoxelReg_{0}_{1}_{2}_{3}.nii.gz'.format(subject, session, regressor, task)))
                        data = nifti.get_fdata()[:, :, :, 0]  # coef_
                        data = np.expand_dims(data, axis=3)
                        ses_mean.append(data)
                    except FileNotFoundError:
                        print('FileNotFoundError', subject, regressor, task, session)
                    try:
                        ses_mean = np.mean(np.concatenate(ses_mean, axis=3), axis=3)
                        ses_mean = np.expand_dims(ses_mean, axis=3)
                        t_test.append(ses_mean)
                    except ValueError:
                        print('no value for', subject, regressor, task)
            t_test = np.concatenate(t_test, axis=3)
            t_stat = np.expand_dims(ttest_1samp(t_test, popmean=0, axis=3)[0], axis=3)
            p_vals = np.expand_dims(ttest_1samp(t_test, popmean=0, axis=3)[1], axis=3)
            new_image = nib.Nifti1Image(np.concatenate([t_stat, p_vals], axis=3), affine=nifti.affine)
            new_image.to_filename(join(out_dir, '{0}_{1}.nii.gz'.format(regressor, task)))

    for task in ['inference', 'instructed']:
        t_test_avg = []
        t_test_diff = []
        for subject in subjects_include:
            resp_mean = []
            for response in ['left', 'right']:
                rule_resp_mean = []
                for rule_resp in ['A', 'B']:
                    ses_mean = []
                    for session in ['ses-2', 'ses-3']:
                        nifti = nib.load(join(glm_run_path, subject, 'VoxelReg_{0}_{1}_C(choice_box, levels=t)[T.{3}{4}]_{2}.nii.gz'.format(subject, session, task, response, rule_resp)))
                        data = nifti.get_fdata()[:, :, :, 0]  # coef_
                        data = np.expand_dims(data, axis=3)
                        ses_mean.append(data)
                    ses_mean = np.mean(np.concatenate(ses_mean, axis=3), axis=3)
                    ses_mean = np.expand_dims(ses_mean, axis=3)
                    rule_resp_mean.append(ses_mean)
                rule_resp_mean = np.mean(np.concatenate(rule_resp_mean, axis=3), axis=3)
                rule_resp_mean = np.expand_dims(rule_resp_mean, axis=3)
                resp_mean.append(rule_resp_mean)
            resp_mean = np.mean(np.concatenate(resp_mean, axis=3), axis=3)
            resp_mean = np.expand_dims(resp_mean, axis=3)
            t_test_avg.append(resp_mean)
            resp_diff = np.diff(np.concatenate(resp_mean, axis=3), axis=3)
            resp_diff = np.expand_dims(resp_diff, axis=3)
            t_test_diff.append(resp_diff)

        for inp, reg in zip([t_test_avg, t_test_diff], ['response_average', 'response_diff']):
            t_test = np.concatenate(inp, axis=3)
            t_stat = np.expand_dims(ttest_1samp(t_test, popmean=0, axis=3)[0], axis=3)
            p_vals = np.expand_dims(ttest_1samp(t_test, popmean=0, axis=3)[1], axis=3)
            new_image = nib.Nifti1Image(np.concatenate([t_stat, p_vals], axis=3), affine=nifti.affine)
            new_image.to_filename(join(out_dir, '{0}_{1}.nii.gz'.format(reg, task)))


def submit(x):
    pbs.pmap(execute, [x], walltime='4:00:00',
             memory=40, nodes=1, tasks=2, name='{}'.format(x))
