import pandas as pd
from scipy import signal
import math
import pupil_pp as ppp


def sessions():
    for sub in [1, 2, 3, 4, 6, 7, 9]:
        for ses in [1, 2, 3]:
            if (sub == 3) & (ses == 2):
                for blo in [1, 2, 3, 4, 5, 6]:
                    yield(sub, ses, blo)
            else:
                for blo in [1, 2, 3, 4, 5, 6, 7]:
                    yield(sub, ses, blo)


for sub, ses, blo in sessions():
    p = ppp.Pupilframe(sub, ses, blo, '/Users/kenohagena/Documents/immuno/data/vaccine')
    p.basicframe()
    p.pupil_frame.to_csv('pupil_basicframe_050218_{0}{1}{2}.csv'.format(sub, ses, blo), index=False)
