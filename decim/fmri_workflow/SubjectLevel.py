import numpy as np
import pandas as pd
from decim.fmri_workflow import BehavDataframe as bd
from decim.fmri_workflow import RoiExtract as re
from decim.fmri_workflow import ChoiceEpochs as ce
from decim.fmri_workflow import LinregVoxel as lv
from decim import slurm_submit as slu
from collections import defaultdict
from os.path import join
from glob import glob
from pymeg import parallel as pbs
from multiprocessing import Pool
# from decim.fmri_workflow import PupilLinear as pf


class SubjectLevel(object):

    def __init__(self, sub, ses=[2, 3], runs=[4, 5, 6, 7, 8],
                 environment='Volume'):
        self.sub = sub
        self.subject = 'sub-{}'.format(sub)
        self.ses = ses
        self.sessions = ['ses-{}'.format(i) for i in ses]
        self.run_indices = runs
        run_names = ['inference_run-4',
                     'inference_run-5',
                     'inference_run-6',
                     'instructed_run-7',
                     'instructed_run-8']
        self.runs = {i: i[:-6] for i in np.array(run_names)[np.array(runs) - 4]}
        if environment == 'Volume':
            self.flex_dir = '/Volumes/flxrl/FLEXRULE'
            self.summary = pd.read_csv('/Users/kenohagena/Flexrule/fmri/analyses/bids_stan_fits/summary_stan_fits.csv')
        elif environment == 'Climag':
            self.flex_dir = '/home/khagena/FLEXRULE'
            self.summary = pd.read_csv('/home/khagena/FLEXRULE/behavior/summary_stan_fits.csv')
        elif environment == 'Hummel':
            self.flex_dir = '/work/faty014/FLEXRULE'
            self.summary = pd.read_csv('/work/faty014/FLEXRULE/behavior/summary_stan_fits.csv')
        else:
            self.flex_dir = environment

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def Input(self, **kwargs):
        for key in kwargs:
            setattr(SubjectLevel, key, kwargs[key])

    def PupilFrames(self, manual=False):
        '''
        First Level
        OUTPUT: PupilFrames
        '''
        if hasattr(self, 'PupilFrame'):
            pass
        else:
            self.PupilFrame = defaultdict(dict)
        for session in self.sessions:
            for run in self.runs.keys():
                if run in self.PupilFrame[session].keys():
                    pass
                else:
                    pupil_frame = pf.execute(self.subject, session, run,
                                             self.flex_dir, manual)
                    self.PupilFrame[session][run] = pupil_frame

    def BehavFrames(self):
        '''
        First Level
        OUTPUT: BehavFrames
        '''
        self.BehavFrame = defaultdict(dict)
        for session in self.sessions:
            for run, task in self.runs.items():
                behavior_frame = bd.execute(self.subject, session,
                                            run, task, self.flex_dir, self.summary)
                self.BehavFrame[session][run] = behavior_frame

    def BehavAlign(self):
        '''
        Second Level
        INPUT: BehavFrames
        '''
        self.BehavAligned = defaultdict(dict)
        for session in self.sessions:
            for run, task in self.runs.items():
                BehavFrame = self.BehavFrame[session][run]
                BehavAligned = bd.fmri_align(BehavFrame, task)
                self.BehavAligned[session][run] = BehavAligned

    def RoiExtract(self):
        '''
        '''
        self.CortRois = defaultdict(dict)
        self.BrainstemRois = defaultdict(dict)
        for session in self.sessions:
            for run in self.runs.keys():
                self.CortRois[session][run] =\
                    re.execute(self.subject, session, run, self.flex_dir)[1]
                self.BrainstemRois[session][run] =\
                    re.execute(self.subject, session, run, self.flex_dir)[0]

    def ChoiceEpochs(self):
        self.ChoiceEpochs = defaultdict(dict)
        for session in self.sessions:
            for run, task in self.runs.items():
                print(session, run)
                self.ChoiceEpochs[session][run] =\
                    ce.execute(self.subject, session,
                               run, task, self.flex_dir,
                               self.BehavFrame[session][run],
                               self.PupilFrame[session][run],
                               self.BrainstemRois[session][run])

    def CleanEpochs(self):
        '''
        Concatenate runs within a Session per task.
        '''
        self.CleanEpochs = defaultdict(dict)
        for session in self.sessions:
            per_session = []
            for run, task in self.runs.items():
                run_epochs = self.ChoiceEpochs[session][run]
                run_epochs['run'] = run
                run_epochs['task'] = task
                per_session.append(run_epochs)
            per_session = pd.concat(per_session, ignore_index=True)
            clean = ce.defit_clean(per_session)
            self.CleanEpochs['session'] = clean

    def LinregVoxel(self):
        self.VoxelReg = defaultdict(dict)
        self.SurfaceTxt = defaultdict(dict)
        for task in set(self.runs.values()):
            for session in self.sessions:
                runs = {k: v for (k, v) in self.runs.items() if v == task}
                self.VoxelReg[session][task], self.SurfaceTxt[session][task] = lv.execute(self.subject, session, runs,
                                                                                          self.flex_dir,
                                                                                          self.BehavAligned[session])

    def Output(self):
        output_dir = join(self.flex_dir, 'SubjectLevel', self.subject)
        slu.mkdir_p(output_dir)
        for name, attribute in self.__iter__():
            if name in ['BehavFrame', 'BehavAligned', 'PupilFrame', 'CortRois', 'BrainstemRois', 'ChoiceEpochs']:
                for session in self.sessions:
                    for run in attribute[session].keys():
                        attribute[session][run].to_hdf(join(output_dir, '{0}_{1}_{2}.hdf'.format(name, self.subject, session), key=run))

            elif name == 'CleanEpochs':
                for session in self.sessions:
                    attribute[session].to_hdf(join(output_dir, '{0}_{1}.hdf'.format(name, self.subject), key=session))
            elif name in ['VoxelReg', 'SurfaceTxt']:
                for session in self.sessions:
                    for task in set(self.runs.values()):
                        attribute[session][task].to_hdf(join(output_dir, '{0}_{1}_{2}.hdf'.format(name, self.subject, session), key=task))


def execute(keys):
    sub = keys[0]
    environment = keys[1]
    sl = SubjectLevel(sub, environment=environment)
    sl.PupilFrame = defaultdict(dict)
    files = glob(join(sl.flex_dir, 'pupil/NEW_PUPILFRAMES', '*Frame_{}_*'.format(sl.sub)))
    for file in files:
        ses = file[file.find('ses-'):file.find('.hdf')]
        with pd.HDFStore(file) as hdf:
            k = hdf.keys()
        for run in k:
            sl.PupilFrame[ses][run[run.find('in'):]] = pd.read_hdf(file, key=run)
    sl.BehavFrames()
    sl.RoiExtract()
    sl.BehavAlign()
    sl.ChoiceEpochs()
    sl.CleanEpochs()
    sl.LinregVoxel()
    sl.Output()


def keys(sub, env):
    keys = []
    for s in range(sub, sub + 2):
        keys.append(([s, env]))
    return keys


def par_execute(keys):
    with Pool(2) as p:
        p.starmap(execute, keys)


def submit(sub, env='Hummel'):
    if env == 'Hummel':
        slu.pmap(par_execute, keys(sub, 'Hummel'), walltime='4:00:00',
                 memory=15, nodes=1, tasks=1, name='SubjectLevel')
    elif env == 'Climag':
        pbs.pmap(par_execute, keys(sub, env), walltime='4:00:00',
                 memory=15, nodes=1, tasks=2, name='SubjectLevel')


'''
if __name__ == '__main__':
    execute(sys.argv[1])
'''
