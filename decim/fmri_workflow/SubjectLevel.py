import numpy as np
import pandas as pd
from decim.fmri_workflow import BehavDataframe as bd
from decim.fmri_workflow import RoiExtract as re
from decim.fmri_workflow import ChoiceEpochs as ce
from decim.fmri_workflow import LinregVoxel as lv
from decim.fmri_workflow import SwitchEpochs as se
from decim import slurm_submit as slu
from collections import defaultdict
from os.path import join
from glob import glob
from multiprocessing import Pool
from pymeg import parallel as pbs
try:
    from decim.fmri_workflow import PupilLinear as pf
except ImportError:
    pass


spec_subs = {1: {2: [4, 5, 6, 7], 3: [4, 5, 6]},
             2: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             3: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             4: {2: [4, 5, 6], 3: [4, 5, 6, 7, 8]},
             5: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             6: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             7: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             8: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             9: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             10: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             11: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             12: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             13: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             14: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             15: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             16: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             17: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             18: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             19: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             20: {2: [4, 5, 6, 7], 3: [4, 5, 6, 7, 8]},
             21: {2: [4, 5, 6, 8], 3: [4, 5, 6, 7, 8]},
             22: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]}}


class SubjectLevel(object):

    def __init__(self, sub, ses_runs={2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
                 environment='Volume'):
        self.sub = sub
        self.subject = 'sub-{}'.format(sub)
        self.ses = ses_runs.keys()
        run_names = ['inference_run-4',
                     'inference_run-5',
                     'inference_run-6',
                     'instructed_run-7',
                     'instructed_run-8']
        self.session_runs = {'ses-{}'.format(session):
                             {i: i[:-6] for i in np.array(run_names)[np.array(runs) - 4]}
                             for session, runs in ses_runs.items()}
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
        for session, runs in self.session_runs.items():
            for run in runs.keys():
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
        for session, runs in self.session_runs.items():
            for run, task in runs.items():
                print('Do behav ', self.subject, session, run)
                try:
                    behavior_frame = bd.execute(self.subject, session,
                                                run, task, self.flex_dir,
                                                self.summary)
                    self.BehavFrame[session][run] = behavior_frame
                except RuntimeError:
                    print('Runtimeerror for', self.subject, session, run)
                    self.BehavFrame[session][run] = None

    def BehavAlign(self, fast=False):
        '''
        Second Level
        INPUT: BehavFrames
        '''
        print('Do behavior align', self.subject)
        self.BehavAligned = defaultdict(dict)
        for session, runs in self.session_runs.items():
            for run, task in runs.items():
                print('Do behav align', self.subject, session, run)
                BehavFrame = self.BehavFrame[session][run]
                if BehavFrame is not None:
                    BehavAligned = bd.fmri_align(BehavFrame, task, fast)
                    self.BehavAligned[session][run] = BehavAligned
                else:
                    continue

    def RoiExtract(self, denoise=True):
        '''
        '''
        self.CortRois = defaultdict(dict)
        self.BrainstemRois = defaultdict(dict)
        for session, runs in self.session_runs.items():
            for run in runs.keys():
                print('Do roi extract', self.subject, session, run)
                self.BrainstemRois[session][run], self.CortRois[session][run] =\
                    re.execute(self.subject, session, run,
                               self.flex_dir, denoise=denoise)

    def ChoiceEpochs(self):
        self.ChoiceEpochs = defaultdict(dict)
        for session, runs in self.session_runs.items():
            for run, task in runs.items():
                print('Do choice epochs', self.subject, session, run)
                self.ChoiceEpochs[session][run] =\
                    ce.execute(self.subject, session,
                               run, task, self.flex_dir,
                               self.BehavFrame[session][run],
                               self.PupilFrame[session][run],
                               self.BrainstemRois[session][run])

    def SwitchEpochs(self):
        self.SwitchEpochs = defaultdict(dict)
        for session, runs in self.session_runs.items():
            for run, task in runs.items():
                print('Do switch epochs', self.subject, session, run)
                self.SwitchEpochs[session][run] =\
                    se.execute(self.subject, session,
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
        for session, runs in self.session_runs.items():
            per_session = []
            for run, task in runs.items():
                run_epochs = self.ChoiceEpochs[session][run]
                run_epochs['run'] = run
                run_epochs['task'] = task
                per_session.append(run_epochs)
            per_session = pd.concat(per_session, ignore_index=True)
            clean = ce.defit_clean(per_session)
            self.CleanEpochs[session] = clean

    def LinregVoxel(self):
        print('Linreg voxel')
        self.VoxelReg = defaultdict(dict)
        self.SurfaceTxt = defaultdict(dict)
        for session, runs in self.session_runs.items():
            for task in set(runs.values()):
                print(task, session)
                rs = [r for r in runs.keys() if runs[r] == task]
                self.VoxelReg[session][task], self.SurfaceTxt[session][task] =\
                    lv.execute(self.subject, session, rs,
                               self.flex_dir,
                               self.BehavAligned[session], task)

    def Output(self, dir='SubjectLevel'):
        print('Output')
        output_dir = join(self.flex_dir, dir, self.subject)
        slu.mkdir_p(output_dir)
        for name, attribute in self.__iter__():
            if name in ['BehavFrame', 'BehavAligned', 'PupilFrame',
                        'CortRois', 'BrainstemRois', 'ChoiceEpochs']:
                for session in attribute.keys():
                    for run in attribute[session].keys():
                        print('Saving', name, session, run)
                        attribute[session][run].to_hdf(join(output_dir,
                                                            '{0}_{1}_{2}.hdf'.
                                                            format(name,
                                                                   self.subject,
                                                                   session)),
                                                       key=run)

            elif name == 'CleanEpochs':
                for session in attribute.keys():
                    print('Saving', name, session)
                    attribute[session].to_hdf(join(output_dir, '{0}_{1}.hdf'.
                                                   format(name, self.subject)),
                                              key=session)
            elif name in ['VoxelReg', 'SurfaceTxt']:
                for session in attribute.keys():
                    for task in attribute[session].keys():
                        for parameter, content in attribute[session][task].items():
                            print('Saving', name, session, task, parameter)
                            if name == 'VoxelReg':
                                content.to_filename(join(output_dir,
                                                         '{0}_{1}_{2}_{3}_{4}.nii.gz'.
                                                         format(name, self.subject,
                                                                session, parameter, task)))
                            elif name == 'SurfaceTxt':
                                for hemisphere, cont in content.items():
                                    cont.to_hdf(join(output_dir,
                                                     '{0}_{1}_{2}_{3}_{4}.hdf'.
                                                     format(name, self.subject,
                                                            session, parameter, hemisphere)),
                                                key=task)


def execute(sub, ses, environment):
    sl = SubjectLevel(sub, ses_runs={ses: spec_subs[sub][ses]}, environment=environment)
    '''
    sl.BehavAligned = defaultdict(dict)
    file = join(sl.flex_dir, 'BehavAligned_30-08-2018', sl.subject, 'BehavAligned_{0}_ses-{1}.hdf'.format(sl.subject, ses))
    with pd.HDFStore(file) as hdf:
        k = hdf.keys()
    for run in k:
        sl.BehavAligned['ses-{}'.format(ses)][run[run.find('in'):]] = pd.read_hdf(file, key=run)
    '''
    sl.BehavFrames()
    # sl.RoiExtract()
    sl.BehavAlign(fast=False)
    # sl.ChoiceEpochs()
    # sl.SwitchEpochs()
    # del sl.PupilFrame
    # sl.CleanEpochs()
    sl.LinregVoxel()
    sl.Output(dir='GLM_31-08-2018')


def par_execute(keys):
    with Pool(2) as p:
        p.starmap(execute, keys)


def submit(sub, env='Hummel'):
    if env == 'Hummel':
        def keys(sub):
            keys = []
            for ses in [2, 3]:
                keys.append((sub, ses, env))
            return keys

        slu.pmap(par_execute, keys(sub), walltime='2:00:00',
                 memory=40, nodes=1, tasks=2, name='SubjectLevel')
    elif env == 'Climag':
        for ses in [2, 3]:
            pbs.pmap(execute, [(sub, ses, env)], walltime='4:00:00',
                     memory=10, nodes=1, tasks=2, name='SubjectLevel')

    '''
    sl.PupilFrame = defaultdict(dict)
    file = glob(join(sl.flex_dir, 'pupil/linear_pupilframes', '*Frame_{0}_ses-{1}.hdf'.format(sl.sub, ses)))
    if len(file) != 1:
        print(len(file), ' pupil frames found...')
    with pd.HDFStore(file[0]) as hdf:
        k = hdf.keys()
    for run in k:
        sl.PupilFrame['ses-{}'.format(ses)][run[run.find('in'):]] = pd.read_hdf(file[0], key=run)
    '''
