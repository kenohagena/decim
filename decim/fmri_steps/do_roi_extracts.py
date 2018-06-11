from decim import roi_extract as re
import pandas as pd
import numpy as np
from os.path import join, expanduser
from glob import glob


subjects = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 19, 20, 21]
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

for sub in subjects:

    e = re.EPI(sub, out_dir=('/Volumes/flxrl/fmri/roi_extract_110618'))
    e.load_epi('/Volumes/flxrl/fmri/completed_preprocessed/sub-{0}/fmriprep/sub-{0}/ses-3/func/'.format(sub),
               identifier='inference*T1w*prepro')
    e.load_epi('/Volumes/flxrl/fmri/completed_preprocessed/sub-{0}/fmriprep/sub-{0}/ses-3/func/'.format(sub),
               identifier='instructed*T1w*prepro')
    e.load_epi('/Volumes/flxrl/fmri/completed_preprocessed/sub-{0}/fmriprep/sub-{0}/ses-2/func/'.format(sub),
               identifier='inference*T1w*prepro')
    e.load_epi('/Volumes/flxrl/fmri/completed_preprocessed/sub-{0}/fmriprep/sub-{0}/ses-2/func/'.format(sub),
               identifier='instructed*T1w*prepro')
    print('{} loaded'.format(sub))
    e.load_mask(expanduser('~/Flexrule/fmri/atlases/sub-{}'.format(sub)), mult_roi_atlases={'CIT': {2: 'NAc', 6: 'SNc', 10: 'VTA'}})
    e.resample_masks()
    print('{} resampled'.format(sub))
    e.mask()
    print('{} masked'.format(sub))
    e.save()

    roi_dfs = []
    subject = 'sub-{}'.format(sub)
    home = expanduser('/Volumes/flxrl/fmri/roi_extract_110618/{}/'.format(subject))
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

    concats = []
    for atlas, name in atlases.items():
        w = []
        for session in sessions:
            for run in runs:
                file = sorted(glob(join(home, '*{0}*{1}*{2}*'.format(session, run, atlas))))
                if len(file) == 0:
                    pass
                else:
                    if name in cit168:
                        weights = pd.read_csv(file[1], index_col=0).T
                    else:
                        weights = pd.read_csv(file[1], index_col=0)
                    weights['atlas'] = name
                    weights['session'] = session
                    weights['run'] = run
                    w.append(weights)
        concat = pd.concat(w, axis=0, ignore_index=True)
        concat.to_csv(join(home, '{1}_{0}_weights'.format(name, subject)))

    df = pd.concat(roi_dfs, axis=0)
    df.index.name = 'frame'
    df = df.set_index(['session', 'run', df.index])
    df.to_csv(join(home, '{}_rois_indexed.csv'.format(subject)), index=True)
