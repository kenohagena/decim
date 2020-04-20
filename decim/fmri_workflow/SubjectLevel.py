import numpy as np
import pandas as pd
import datetime
from decim.fmri_workflow import BehavDataframe as bd
from decim.fmri_workflow import RoiExtract as re
from decim.fmri_workflow import ChoiceEpochs as ce
from decim.fmri_workflow import LinregVoxel as lv
from decim.fmri_workflow import SwitchEpochs as se
from decim.fmri_workflow import CortexEpochs as cort
from decim.fmri_workflow import SingleTrialGLM as st
from decim.fmri_workflow import Decoder as dec

from decim.adjuvant import slurm_submit as slu
from collections import defaultdict
from os.path import join
from glob import glob
from multiprocessing import Pool
from pymeg import parallel as pbs
try:
    from decim.fmri_workflow import PupilLinear as pf
except ImportError:
    pass

'''
I use this script to connect several subscripts into workflows.

The workflow is defined in "Execute".
"Execute" takes the following arguments:
    a) subject as int (e.g. 17)
    b) session as int (e.g. 2)
    c) environment ('Climag', 'Hummel', 'Volume')
Flexrule directory is then chosen for my accounts on these clusters. (This an be changed in the init function)

Within the execute function, the workflow can be adapted.

Example I (run all analyses from scratch for one subject):
    def execute(sub, ses, environment):
        sl = SubjectLevel(sub, ses_runs={ses: spec_subs[sub][ses]}, environment=environment)
        sl.BehavFrames()
        sl.RoiExtract()
        sl.PupilFrame = defaultdict(dict)
        file = glob(join(sl.flex_dir, 'pupil/linear_pupilframes', 'PupilFrame_{0}_ses-{1}.hdf'.format(sl.sub, ses)))
        if len(file) != 1:
            print(len(file), ' pupil frames found...')
        with pd.HDFStore(file[0]) as hdf:
            k = hdf.keys()
        for run in k:
            sl.PupilFrame['ses-{}'.format(ses)][run[run.find('in'):]] = pd.read_hdf(file[0], key=run)
        sl.ChoiceEpochs()
        sl.SwitchEpochs()
        del sl.PupilFrame
        sl.CleanEpochs(epoch='Switch')
        sl.LinregVoxel()
        sl.Output(dir='Sublevel_GLM_{1}_{0}'.format(datetime.datetime.now().strftime("%Y-%m-%d"), environment))

Example II (run only GLM):

    def execute(sub, ses, environment):
        sl = SubjectLevel(sub, ses_runs={ses: spec_subs[sub][ses]}, environment=environment)
        sl.BehavFrames()
        sl.LinregVoxel()
        sl.Output(dir='Sublevel_GLM_{1}_{0}'.format(datetime.datetime.now().strftime("%Y-%m-%d"), environment))

Comments:
1. In the "Output" method of "SubjectLevel" i specify an ouput directory.
2. I once preprocessed pupil data manually. Thus I do not include this step in workflows,
    but rather reimport the preprocessed pupil data into the workflow



"Submit" function takes subject as integer and environment (e.g. submit(17, 'CLimag'))
and runs execute for all sessions for the subject.
'''

# Nested dictionary with subjects, sessions and completed runs
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
             17: {2: [7, 8], 3: [7, 8]},
             18: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             19: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             20: {2: [4, 5, 6, 7], 3: [4, 5, 6, 7, 8]},
             21: {2: [4, 5, 6, 8], 3: [4, 5, 6, 7, 8]},
             22: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]}}


class SubjectLevel(object):

    def __init__(self, sub, ses_runs={2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
                 environment='Volume', out_dir='SubjectLevel'):
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
        self.out_dir = join(self.flex_dir, out_dir, self.subject)
        slu.mkdir_p(self.out_dir)

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

    def BehavFrames(self, belief_TR=False):
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
                                                self.summary, belief_TR=belief_TR)
                    self.BehavFrame[session][run] = behavior_frame
                except RuntimeError:
                    print('Runtimeerror for', self.subject, session, run)
                    self.BehavFrame[session][run] = None

    def RoiExtract(self, input_nifti='mni_retroicor'):
        '''
        '''
        self.CortRois = defaultdict(dict)
        self.BrainstemRois = defaultdict(dict)
        for session, runs in self.session_runs.items():
            for run in runs.keys():
                print('Do roi extract', self.subject, session, run)
                self.BrainstemRois[session][run], self.CortRois[session][run] =\
                    re.execute(self.subject, session, run,
                               self.flex_dir, input_nifti=input_nifti)

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

    def SwitchEpochs(self, mode):
        self.SwitchEpochs = defaultdict(dict)
        for session, runs in self.session_runs.items():
            for run, task in runs.items():
                print('Do switch epochs', self.subject, session, run)
                self.SwitchEpochs[session][run] =\
                    se.execute(self.subject, session,
                               run, task, self.flex_dir,
                               self.BehavFrame[session][run],
                               self.PupilFrame[session][run],
                               self.BrainstemRois[session][run], mode=mode)

    def CortexEpochs(self):
        self.CortexEpochs = defaultdict(dict)
        for session, runs in self.session_runs.items():
            for run, task in runs.items():
                print('Do cort epochs', self.subject, session, run)
                self.CortexEpochs[session][run] =\
                    cort.execute(self.subject, session,
                                 run, task, self.flex_dir,
                                 self.BehavFrame[session][run],
                                 self.CortRois[session][run])

    def CleanEpochs(self, epoch='Choice'):
        '''
        Concatenate runs within a Session per task.
        '''
        print('Clean epochs')
        self.CleanEpochs = defaultdict(dict)
        for session, runs in self.session_runs.items():
            per_session = []
            for run, task in runs.items():
                if epoch == 'Choice':
                    run_epochs = self.ChoiceEpochs[session][run]
                if epoch == 'Switch':
                    run_epochs = self.SwitchEpochs[session][run]
                run_epochs['run'] = run
                run_epochs['task'] = task
                per_session.append(run_epochs)
            per_session = pd.concat(per_session, ignore_index=True)
            if epoch == 'Choice':
                self.CleanEpochs[session] = ce.defit_clean(per_session)
            else:
                self.CleanEpochs[session] = per_session

    def LinregVoxel(self):
        print('Linreg voxel')
        self.VoxelReg = defaultdict(dict)
        self.SurfaceTxt = defaultdict(dict)
        self.DesignMatrix = defaultdict(dict)
        self.Residuals = defaultdict(dict)
        for session, runs in self.session_runs.items():
            for task in set(runs.values()):
                print(task, session)
                rs = [r for r in runs.keys() if runs[r] == task]
                self.VoxelReg[session][task], self.SurfaceTxt[session][task], self.DesignMatrix[session][task], self.Residuals[session][task] =\
                    lv.execute(self.subject, session, rs,
                               self.flex_dir,
                               self.BehavFrame[session], task)
                print(self.Residuals[session][task])

    def SingleTrialGLM(self):
        print('SingleTrialGLM')
        self.SingleTrial = defaultdict(dict)
        self.TrialRules = defaultdict(dict)
        for session, runs in self.session_runs.items():
            for task in set(runs.values()):
                print(task, session)
                rs = [r for r in runs.keys() if runs[r] == task]
                self.SingleTrial[session], self.TrialRules[session] = st.execute(self.subject, session, rs,
                                                                                 self.flex_dir,
                                                                                 self.BehavFrame[session], self.Residuals[session][task], self.out_dir)

    def Decode(self):
        print('Decoder')
        for session in self.session_runs.keys():
            dec.execute(self.subject, session, trialbetas=self.SingleTrial[session],
                        trialrules=pd.Series(self.TrialRules[session]))

    def Output(self):
        print('Output')
        for name, attribute in self.__iter__():
            if name in ['BehavFrame', 'BehavAligned', 'PupilFrame',
                        'CortRois', 'BrainstemRois', 'ChoiceEpochs', 'CortexEpochs', 'Residuals', 'Sin']:
                for session in attribute.keys():
                    for run in attribute[session].keys():
                        print('Saving', name, session, run)
                        if name == 'Residuals':
                            for task, nifti in attribute[session][run].items():
                                nifti.to_filename(join(self.out_dir,
                                                       '{0}_{1}_{2}_{3}_{4}.nii.gz'.
                                                       format(name,
                                                              self.subject,
                                                              session, run, task)))
                        else:
                            attribute[session][run].to_hdf(join(self.out_dir,
                                                                '{0}_{1}_{2}.hdf'.
                                                                format(name,
                                                                       self.subject,
                                                                       session)),
                                                           key=run)

            elif name == 'CleanEpochs':
                for session in attribute.keys():
                    print('Saving', name, session)
                    attribute[session].to_hdf(join(self.out_dir, '{0}_{1}_{2}.hdf'.
                                                   format(name, self.subject, session)),
                                              key=session)
            elif name in ['VoxelReg', 'SurfaceTxt']:
                for session in attribute.keys():
                    for task in attribute[session].keys():
                        for parameter, content in attribute[session][task].items():
                            print('Saving', name, session, task, parameter)
                            if name == 'VoxelReg':
                                content.to_filename(join(self.out_dir,
                                                         '{0}_{1}_{2}_{3}_{4}.nii.gz'.
                                                         format(name, self.subject,
                                                                session, parameter, task)))
                            elif name == 'SurfaceTxt':
                                for hemisphere, cont in content.items():
                                    cont.to_hdf(join(self.out_dir,
                                                     '{0}_{1}_{2}_{3}_{4}.hdf'.
                                                     format(name, self.subject,
                                                            session, parameter, hemisphere)),
                                                key=task)
            elif name == 'DesignMatrix':
                for session in attribute.keys():
                    for task in attribute[session].keys():
                        attribute[session][task].to_hdf(join(self.out_dir,
                                                             '{0}_{1}_{2}.hdf'.format(name, self.subject, session)), key=task)

            elif name in ['SingleTrial', 'TrialRules']:
                for session in attribute.keys():
                    if name == 'SingleTrial':
                        attribute[session].to_filename(join(self.out_dir,
                                                            '{0}_{1}_{2}.nii.gz'.
                                                            format(name,
                                                                   self.subject,
                                                                   session)))
                    elif name == 'TrialRules':
                        pd.Series(attribute[session]).to_hdf(join(self.out_dir,
                                                                  '{0}_{1}.hdf'.
                                                                  format(name,
                                                                         self.subject)), key=session)


def execute(sub, ses, environment):
    '''
    sl = SubjectLevel(sub, ses_runs={ses: spec_subs[sub][ses]}, environment=environment)
    sl.BehavFrames()
    sl.RoiExtract(input_nifti='T1w')
    sl.CortexEpochs()
    sl.Output(dir='Workflow/Sublevel_CortEpochs_{1}_{0}-b'.format(datetime.datetime.now().strftime("%Y-%m-%d"), environment))
    '''
    out_dir = 'Workflow/Sublevel_GLM_{1}_{0}'.format(datetime.datetime.now().strftime("%Y-%m-%d"), environment)
    sl = SubjectLevel(sub, ses_runs={ses: spec_subs[sub][ses]}, environment=environment, out_dir=out_dir)
    sl.BehavFrames()
    sl.LinregVoxel()
    sl.SingleTrialGLM()
    sl.Decode()
    sl.Output()

    '''
    sl = SubjectLevel(sub, ses_runs={ses: spec_subs[sub][ses]}, environment=environment)  # {ses: [4, 5, 6]} to only run inference
    sl.BehavFrames()
    sl.RoiExtract(input_nifti='mni_retroicor')
    sl.PupilFrame = defaultdict(dict)
    file = glob(join(sl.flex_dir, 'pupil/linear_pupilframes_manual', 'PupilFrame_sub-{0}_ses-{1}.hdf'.format(sl.sub, ses)))
    if len(file) != 1:
        print(len(file), ' pupil frames found...')
    with pd.HDFStore(file[0]) as hdf:
        k = hdf.keys()
    for run in k:
        sl.PupilFrame['ses-{}'.format(ses)][run[run.find('in'):]] = pd.read_hdf(file[0], key=run)
    # sl.ChoiceEpochs()
    sl.SwitchEpochs(mode='switch')
    del sl.PupilFrame
    # sl.CleanEpochs(epoch='Choice')
    sl.CleanEpochs(epoch='Switch')
    sl.Output(dir='Workflow/Sublevel_SwitchEpochs_{1}_{0}'.format(datetime.datetime.now().strftime("%Y-%m-%d"), environment))
    '''


def par_execute(keys):
    with Pool(2) as p:
        p.starmap(execute, keys)


def submit(sub, env='Climag'):
    if env == 'Hummel':
        def keys(sub):
            keys = []
            for ses in [2, 3]:
                keys.append((sub, ses, env))
            return keys

        slu.pmap(par_execute, keys(sub), walltime='2:00:00',
                 memory=40, nodes=1, tasks=2, name='SubjectLevel')
    elif env == 'Climag':
        for ses in [2]:
            pbs.pmap(execute, [(sub, ses, env)], walltime='4:00:00',
                     memory=40, nodes=1, tasks=2, name='subvert_sub-{}'.format(sub))


'working_version'
