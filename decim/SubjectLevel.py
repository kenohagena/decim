import numpy as np
import pandas as pd
from decim import pupil_frame as pf
from decim.fmri_steps import BehavFrame
from collections import defaultdict


class SubjectLevel(object):

    def __init__(self, sub, ses=[2, 3], inference=[4, 5, 6], instructed=[7, 8], environment='Volume'):
        self.sub = sub
        self.subject = 'sub-{}'.format(sub)
        self.ses = ses
        self.sessions = ['ses-{}'.format(i) for i in ses]
        self.inf_runs = inference
        self.instr_runs = instructed
        if environment == 'Volume':
            self.basepath = '/Volumes/flxrl/FLEXRULE'
        elif environment == 'Climag':
            self.basepath = '/home/khgena/FLEXRULE'
        elif environment == 'Hummel':
            self.basepath = '/work/faty014/FLEXRULE'
        else:
            self.basepath = environment

    def Input(self, **kwargs):
        for key in kwargs:
            setattr(SubjectLevel, key, kwargs[key])

    def PupilFrames(self, manual=False):
        self.pupil = defaultdict(dict)
        for ses, session in zip(self.ses, self.sessions):
            for run in self.inf_runs:
                iris = pf.PupilFrame(self.sub, ses, run, self.basepath)
                iris.basicframe()
                iris.blink_interpol()
                if manual is True:
                    iris.man_deblink()
                iris.finish()
                self.pupil[session][run] = iris
            for run in self.instr_runs:
                iris = pf.PupilFrame(self.sub, ses, run, self.basepath)
                iris.basicframe(messages=False)
                iris.blink_interpol()
                if manual is True:
                    iris.man_deblink()
                iris.finish()
                self.pupil[session][run] = iris

    def BehavFrames(self):
        self.behavior = defaultdict(dict)
        for ses, session in zip(self.ses, self.sessions):
            bf = BehavFrame.BehavDataframe(self.sub)
