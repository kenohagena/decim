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


def extract_cortical_roi(sub, session, run, epi_dir):
    '''
    '''
    for hemisphere, hemi in h.items():
        subject = 'sub-{}'.format(sub)
        annot_path = join(epi_dir, subject, 'freesurfer', subject, 'label', '{0}.HCPMMP1.annot'.format(hemisphere))
        lh_func_path = glob(join(epi_dir, subject, 'fmriprep', subject, session, 'func', '*{0}*fsnative.{1}.func.gii'.format(run, hemi)))[0]

        annot = ni.freesurfer.io.read_annot(annot_path)
        labels = annot[2]
        labels = [i.astype('str') for i in labels]
        affiliation = annot[0]
        surf = surface.load_surf_data(lh_func_path)
        surf_df = pd.DataFrame(surf).T

        # z-score per vertex
        surf_df = (surf_df - surf_df.mean()) / surf_df.std()
        print(surf_df.mean())
        surf_df = surf_df.T
        surf_df['label_index'] = affiliation
        df = surf_df.groupby('label_index').mean().T
        df.columns = labels
        return df


def weighted_average(sub, session, run, atlas, roi_dir):
    '''
    '''
    subject = 'sub-{}'.format(sub)

    roi = pd.read_csv(glob(join(roi_dir, subject, '{0}*{1}*{2}*{3}*'.format(subject, session, run, atlas)))[0], index_col=0)
    weight = pd.read_csv(glob(join(roi_dir, subject, '{0}_{1}_resampled_weights'.format(sub, atlas)))[0], index_col=0)
    weight = weight.iloc[0, 0:-3].values.astype(float)

    # z-score per voxel
    roi = (roi - roi.mean()) / roi.std()

    # normalize weights ...
    weight = weight / weight.sum()

    weighted = np.dot(roi, weight)
    weighted = pd.DataFrame(weighted)
    return weighted


if __name__ == "__main__":
    for sub in range(1, 23):
        for session in sessions:
            for run in runs:
                df = extract_cortical_roi(sub, session, run, epi_dir)
                for atlas in atlases:
                    df[atlas] = weighted_average(sub, session, run, atlas, out_dir)
                df.to_csv(join(out_dir, '{0}_{1}_{2}_weighted_rois.csv'.format(sub, session, run)))


# To run brainstem extractions embedded in a shell script
    '''
    extract_brainstem_roi(sys.argv[1], epi_dir, atlas_dir, out_dir)  # to embed in shell script
    concat_single_rois(sys.argv[1], out_dir)
    '''
# To run brain extraction within the python script
    '''
    for sub in subjects:
        extract_brainstem_roi(sub, epi_dir='/Volumes/flxrl/fmri/completed_preprocessed',
                atlas_dir='/Users/kenohagena/Flexrule/fmri/atlases', out_dir='/Volumes/flxrl/fmri/roi_extract-120618')

    '''
