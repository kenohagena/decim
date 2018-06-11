import pandas as pd
import numpy as np
from decim import pupil_frame as pf
from os.path import join, expanduser
from glob import glob
from decim import slurm_submit as slurm


bids_mr = '/Volumes/flxrl/fmri/bids_mr/'
outpath = expanduser('~/Flexrule/fmri/analyses/pupil_dataframes_310518')

for sub in range(1, 23):
    subject = 'sub-{}'.format(sub)
    savepath = join(outpath, subject)
    slurm.mkdir_p(savepath)
    for ses in range(1, 4):
        session = 'ses-{}'.format(ses)
        files = glob(join(bids_mr, subject, session, '*', '*inference*.edf'))
        if len(files) == 0:
            pass
        else:
            for file in files:
                run = file[file.find('inference'):file.find('_phys')]
                raw = pf.Pupilframe(subject, session, None, bids_mr, bids=True)
                raw.basicframe(directory=[file])
                raw.gaze_angle()
                raw.all_artifacts()
                raw.small_fragments()
                raw.interpol()
                raw.filter()
                raw.z_score()
                raw.pupil_frame.to_csv(join(outpath, subject, 'pupil_{0}_{1}_{2}.csv'.format(subject, session, run)))
