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
        print('Do pupil ', self.subject)
        if hasattr(self, 'PupilFrame'):
            pass
        else:
            self.PupilFrame = defaultdict(dict)
        for session in self.sessions:
            for run in self.runs.keys():
                print('Do pupil ', self.subject, session, run)
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
        print('Do behavior ', self.subject)
        self.BehavFrame = defaultdict(dict)
        for session in self.sessions:
            for run, task in self.runs.items():
                print('Do behav ', self.subject, session, run)
                try:
                    behavior_frame = bd.execute(self.subject, session,
                                                run, task, self.flex_dir, self.summary)
                    self.BehavFrame[session][run] = behavior_frame
                except RuntimeError:
                    print('Runtimeerror for', self.subject, session, run)
                    self.BehavFrame[session][run] = None

    def BehavAlign(self):
        '''
        Second Level
        INPUT: BehavFrames
        '''
        print('Do behavior align', self.subject)
        self.BehavAligned = defaultdict(dict)
        for session in self.sessions:
            for run, task in self.runs.items():
                print('Do behav align', self.subject, session, run)
                BehavFrame = self.BehavFrame[session][run]
                if BehavFrame is not None:
                    BehavAligned = bd.fmri_align(BehavFrame, task)
                    self.BehavAligned[session][run] = BehavAligned
                else:
                    continue

    def RoiExtract(self):
        '''
        '''
        print('Do roi extract', self.subject)
        self.CortRois = defaultdict(dict)
        self.BrainstemRois = defaultdict(dict)
        for session in self.sessions:
            for run in self.runs.keys():
                print('Do roi extract', self.subject, session, run)
                self.CortRois[session][run] =\
                    re.execute(self.subject, session, run, self.flex_dir)[1]
                self.BrainstemRois[session][run] =\
                    re.execute(self.subject, session, run, self.flex_dir)[0]

    def ChoiceEpochs(self):
        print('Do choice epochs', self.subject)
        self.ChoiceEpochs = defaultdict(dict)
        for session in self.sessions:
            for run, task in self.runs.items():
                print('Do choice epochs', self.subject, session, run)
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
        print('Clean epochs')
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
        print('Linreg voxel')
        self.VoxelReg = defaultdict(dict)
        self.SurfaceTxt = defaultdict(dict)
        for task in set(self.runs.values()):
            for session in self.sessions:
                runs = {k: v for (k, v) in self.runs.items() if v == task}
                self.VoxelReg[session][task], self.SurfaceTxt[session][task] = lv.execute(self.subject, session, runs,
                                                                                          self.flex_dir,
                                                                                          self.BehavAligned[session])

    def Output(self):
        print('Output')
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


def execute(sub, environment):
    sl = SubjectLevel(sub, ses=[2], environment=environment)
    sl.PupilFrame = defaultdict(dict)
    files = glob(join(sl.flex_dir, 'pupil/linear_pupilframes', '*Frame_{}_*'.format(sl.sub)))
    for file in files:
        ses = file[file.find('ses-'):file.find('.hdf')]
        with pd.HDFStore(file) as hdf:
            k = hdf.keys()
        for run in k:
            sl.PupilFrame[ses][run[run.find('in'):]] = pd.read_hdf(file, key=run)
    sl.BehavFrames()
    print('ok')
    sl.RoiExtract()
    sl.BehavAlign()
    sl.ChoiceEpochs()
    sl.CleanEpochs()
    sl.LinregVoxel()
    sl.Output()


def submit(sub, env='Hummel'):
    if env == 'Hummel':
        slu.pmap(par_execute, keys(sub, 'Hummel'), walltime='4:00:00',
                 memory=15, nodes=1, tasks=1, name='SubjectLevel')
    elif env == 'Climag':
        pbs.pmap(execute, [(sub, env)], walltime='4:00:00',
                 memory=15, nodes=1, tasks=1, name='SubjectLevel')
