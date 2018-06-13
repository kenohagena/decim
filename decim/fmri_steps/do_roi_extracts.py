from decim import roi_extract as re
import pandas as pd
import numpy as np
from os.path import join, expanduser
from glob import glob
from decim import slurm_submit as slu
import sys
from nilearn import surface
import nibabel as ni


epi_dir = '/home/khagena/FLEXRULE/fmri/completed_preprocessed'
atlas_dir = '/home/khagena/FLEXRULE/fmri/atlases'
out_dir = '/home/khagena/FLEXRULE/fmri/roi_extract_130618'
subjects = [12, 13, 14, 19, 20, 21]
sessions = ['ses-2', 'ses-3']
runs = ['inference_run-4',
        'inference_run-5',
        'inference_run-6',
        'instructed_run-7',
        'instructed_run-8']
atlases = {
    'AAN_DR': 'aan_dr',
    'basal_forebrain_4': 'zaborsky_bf4',
    'basal_forebrain_123': 'zaborsky_bf123',
    'LC_Keren_2std': 'keren_lc_2std',
    'LC_standard': 'keren_lc_1std',
    'NAc': 'nac',
    'SNc': 'snc',
    'VTA': 'vta'
}
cit168 = ['nac', 'snc', 'vta']
h = {'lh': 'L',
     'rh': 'R'}


def extract_brainstem_roi(sub, epi_dir, atlas_dir, out_dir):

    slu.mkdir_p(out_dir)
    e = re.EPI(sub, out_dir=out_dir)
    e.load_epi('{1}/sub-{0}/fmriprep/sub-{0}/ses-3/func/'.format(sub, epi_dir),
               identifier='inference*T1w*prepro')
    e.load_epi('{1}/sub-{0}/fmriprep/sub-{0}/ses-3/func/'.format(sub, epi_dir),
               identifier='instructed*T1w*prepro')
    e.load_epi('{1}/sub-{0}/fmriprep/sub-{0}/ses-2/func/'.format(sub, epi_dir),
               identifier='inference*T1w*prepro')
    e.load_epi('{1}/sub-{0}/fmriprep/sub-{0}/ses-2/func/'.format(sub, epi_dir),
               identifier='instructed*T1w*prepro')
    print('{} loaded'.format(sub))
    e.load_mask(expanduser('{1}/sub-{0}'.format(sub, atlas_dir)), mult_roi_atlases={'CIT': {2: 'NAc', 6: 'SNc', 10: 'VTA'}})
    e.resample_masks()
    print('{} resampled'.format(sub))
    e.mask()
    print('{} masked'.format(sub))
    e.save()


def concat_single_rois(sub, out_dir):

    subject = 'sub-{}'.format(sub)
    home = '{1}/{0}/'.format(subject, out_dir)
    roi_dfs = []
    for session in sessions:
        for run in runs:
            runwise = []
            for atlas, name in atlases.items():
                file = sorted(glob(join(home, '*{0}*{1}*{2}*'.format(session, run, atlas))))
                if len(file) == 0:
                    pass
                else:
                    df = pd.read_csv(file[0], index_col=0)
                    cols = pd.MultiIndex.from_product([[name], range(df.shape[1])], names=['roi', 'voxel'])
                    design = pd.DataFrame(np.full(df.shape, np.nan), columns=cols)
                    design[name] = df.values
                    runwise.append(design)
            if len(file) == 0:
                pass
            else:
                concat = pd.concat(runwise, axis=1, ignore_index=False)
                concat['session'] = session
                concat['run'] = run
                roi_dfs.append(concat)

    df = pd.concat(roi_dfs, axis=0)
    df.index.name = 'frame'
    df = df.set_index(['session', 'run', df.index])
    df.to_csv(join(home, '{}_rois_indexed.csv'.format(subject)), index=True)


def extract_cortical_roi(sub, session, run, epi_dir, out_dir):
    '''
    '''
    for hemisphere, hemi in h.items():
        subject = 'sub-{}'.format(sub)
        annot_path = join(epi_dir, subject, 'freesurfer', subject, 'label', '{0}.HCPMMP1.annot'.format(hemisphere))
        lh_func_path = join(epi_dir, subject, 'fmriprep', subject, session, 'func', '*{0}*fsnative.{1}.func.gii'.format(run, hemi))

        annot = ni.freesurfer.io.read_annot(annot_path)
        labels = annot[2]
        labels = [i.astype('str') for i in labels]
        affiliation = annot[0]
        surf = surface.load_surf_data(lh_func_path)
        surf_df = pd.DataFrame(surf)
        surf_df['label_index'] = affiliation
        df = surf_df.groupby('label_index').mean().T
        df.columns = labels
        return df


def weighted_average(atlas, sub, ses, run):
    '''
    '''
    results = []
    binned = []
    for atlas in atlases:
        for sub in subjects:
            for ses in sessions:
                print(atlas, sub, ses)
                rois = pd.read_csv('/Volumes/flxrl/fmri/roi_extract_290518/{0}/{0}_rois_indexed.csv'.format(sub), index_col=[0, 1, 2], header=[0, 1])
                weight = pd.read_csv('/Volumes/flxrl/fmri/roi_extract_290518/{0}/{0}_{1}_weights'.format(sub, atlas), index_col=0)
                weight = weight.iloc[0, 0:-3].values.astype(float)

                roi_zs = []
                behavs = []
                for run in runs:
                    roi = rois.loc[ses, run]
                    roi = roi[atlas]
                    if len(roi) > len(behav):
                        roi = roi.iloc[0: len(behav)]
                    elif len(roi) < len(behav):
                        behav = behav.iloc[0:len(roi)]
                    roi_z = (roi - roi.mean()) / roi.std()
                    roi_zs.append(roi_z)
                    behavs.append(behav)

                behav = pd.concat(behavs, ignore_index=True)
                roi_z = pd.concat(roi_zs, ignore_index=True)

                # normalize weights ...
                weighted = np.dot(roi_z, weight)
                weighted = pd.DataFrame(weighted, index=behav.index)

                slope, intercept, r_value, p_value, std_err = linregress(behav[param].values, weighted.values.flatten())
                result = {'slope': slope, "rhat": r_value, 'p_value': p_value, 'intercept': intercept,
                          'std_err': std_err, 'subject': sub, 'session': ses, 'parameter': param, 'atlas': atlas}
                results.append(result)
                binn = weighted.groupby(pd.cut(behav[param], np.linspace(limits[0], limits[1], 7))).mean().values
                binned.append(binn)

    results = pd.DataFrame(results)
    binned = pd.DataFrame([i.flatten() for i in binned])
    results = pd.concat([binned, results], axis=1)

    results.to_csv('{}_regression_tois.csv'.format(param))


if __name__ == "__main__":
    extract_brainstem_roi(sys.argv[1], epi_dir, atlas_dir, out_dir)  # to embed in shell script
    concat_single_rois(sys.argv[1], out_dir)
    '''
    for sub in subjects:
        extract_brainstem_roi(sub, epi_dir='/Volumes/flxrl/fmri/completed_preprocessed',
                atlas_dir='/Users/kenohagena/Flexrule/fmri/atlases', out_dir='/Volumes/flxrl/fmri/roi_extract-120618')

    '''
