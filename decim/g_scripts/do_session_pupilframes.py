import numpy as np
import pandas as pd
from scipy import signal
import math
import ppp_cluster as ppp


def blocks():
    for sub in [1, 2, 3, 4, 6, 7, 9]:
        for ses in [1, 2, 3]:
            for blo in [1, 2, 3, 4, 5, 6, 7]:
                yield(sub, ses, blo)


for sub, ses, blo in blocks():
    if (sub == 3) & (ses == 2) & (blo == 7):
        continue
    else:

        df = pd.read_csv('immuno/basicframes050218/pupil_basicframe_050218_{0}{1}{2}.csv'.format(sub, ses, blo))
        p = ppp.Pupilframe(df)
        p.gaze_angle()
        p.all_artifacts()
        p.small_fragments()
        p.interpol()
        p.chop()
        p.filter()
        p.z_score()
        df = p.pupil_frame
        df['block'] = blo

        if blo == 1:
            dfr = df
        else:
            dfr = pd.concat([dfr, df], ignore_index=True)
        if (blo == 7) | ((sub == 3) & (ses == 2) & (blo == 6)):
            dfr.to_csv('immuno/session_pupilframes_230218/pupil_{0}{1}.csv'.format(sub, ses), index=False)
