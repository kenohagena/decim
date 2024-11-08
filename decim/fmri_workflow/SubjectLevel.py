import numpy as np
import pandas as pd
import datetime
from decim.fmri_workflow import BehavDataframe as bd
from decim.fmri_workflow import RoiExtract as re
from decim.fmri_workflow import ChoiceEpochs as ce
from decim.fmri_workflow import LinregVoxel as lv
from decim.fmri_workflow import SwitchEpochs as se
from decim.fmri_workflow import CortexEpochs as cort
from decim.fmri_workflow import SampleCortexEpochs as samplecort
from decim.fmri_workflow import SingleTrialGLM as st
from decim.fmri_workflow import Decoder as dec
from decim.fmri_workflow import KernelEpochs as ke
from decim.fmri_workflow import AlternativeModel as am

import nibabel as nib
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


Example III (Decoding):
    out_dir = 'Workflow/Sublevel_GLM_{1}_{0}-b'.format(datetime.datetime.now().strftime("%Y-%m-%d"), environment)
    sl = SubjectLevel(sub, ses, runs=[7, 8], environment=environment, out_dir=out_dir)
    sl.BehavFrames()
    sl.Residuals = defaultdict(dict)
    d = {}
    for run in ['instructed_run-7', 'instructed_run-8']:
        path = join(sl.flex_dir, 'fmri', 'completed_preprocessed',
                    sl.subject, 'fmriprep', sl.subject,
                    'ses-{}'.format(ses), 'func',
                    '{0}_{1}_task-{2}_*{3}*nii.gz'.
                    format(sl.subject, 'ses-{}'.format(ses), run,
                                 'space-MNI152NLin2009cAsym_preproc'))
        print(path)
        files = glob(path)
        print(files)
        d[run] = nib.load(files[0])
    sl.Residuals['instructed'] = d
    sl.SingleTrialGLM()
    sl.Decode()
    sl.Output()

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
             17: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             18: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             19: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]},
             20: {2: [4, 5, 6, 7], 3: [4, 5, 6, 7, 8]},
             21: {2: [4, 5, 6, 8], 3: [4, 5, 6, 7, 8]},
             22: {2: [4, 5, 6, 7, 8], 3: [4, 5, 6, 7, 8]}}

run_names = ['inference_run-4',
             'inference_run-5',
             'inference_run-6',
             'instructed_run-7',
             'instructed_run-8']


class SubjectLevel(object):

    def __init__(self, sub, ses, runs=[4, 5, 6, 7, 8],
                 environment='Volume', out_dir='SubjectLevel'):
        self.sub = sub
        self.subject = 'sub-{}'.format(sub)
        self.session = 'ses-{}'.format(ses)
        self.runs = [run_names[i - 4]for i in runs]
        self.tasks = set([r[:-6] for r in self.runs])
        if environment == 'Volume':
            self.flex_dir = '/Volumes/flxrl/FLEXRULE'
            self.summary = pd.read_csv('/Users/kenohagena/Flexrule/fmri/analyses/bids_stan_fits/summary_stan_fits.csv')
        elif environment == 'Climag':
            self.flex_dir = '/home/khagena/FLEXRULE'
            self.summary = pd.read_csv('/home/khagena/FLEXRULE/behavior/summary_stan_fits.csv')
            self.leaky_fits = pd.read_csv('/home/khagena/FLEXRULE/behavior/Stan_Fits_Leaky_2020-08-22/new/summary_stan_fits.csv')
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
        for run in self.runs:
            print('Do pupil ', self.subject, self.session, run)
            if run in self.PupilFrame[self.session].keys():
                pass
            else:
                pupil_frame = pf.execute(self.subject, self.session, run,
                                         self.flex_dir, manual)
                self.PupilFrame[self.session][run] = pupil_frame

    def BehavFrames(self, belief_TR=False):
        '''
        First Level
        OUTPUT: BehavFrames
        '''
        print('Do behavior ', self.subject)
        self.BehavFrame = defaultdict(dict)
        for run in self.runs:
            task = run[:-6]
            print('Do behav ', self.subject, self.session, run)
            try:
                behavior_frame = bd.execute(self.subject, self.session,
                                            run, task, self.flex_dir,
                                            self.summary, self.leaky_fits, belief_TR=belief_TR)
                self.BehavFrame[run] = behavior_frame
            except RuntimeError:
                print('Runtimeerror for', self.subject, self.session, run)
                self.BehavFrame[run] = None

    def RoiExtract(self, input_nifti='mni_retroicor'):
        '''
        '''
        self.CortRois = defaultdict(dict)
        self.BrainstemRois = defaultdict(dict)
        for run in self.runs:
            print('Do roi extract', self.subject, self.session, run)
            self.BrainstemRois[run], self.CortRois[run] =\
                re.execute(self.subject, self.session, run,
                           self.flex_dir, input_nifti=input_nifti)

    def ChoiceEpochs(self):
        self.ChoiceEpochs = defaultdict(dict)
        for run in self.runs:
            task = run[:-6]
            print('Do choice epochs', self.subject, self.session, run)
            self.ChoiceEpochs[run] =\
                ce.execute(self.subject, self.session,
                           run, task, self.flex_dir,
                           self.BehavFrame[run],
                           self.PupilFrame[run],
                           self.BrainstemRois[run])

    def SwitchEpochs(self, mode):
        self.SwitchEpochs = defaultdict(dict)
        for run in self.runs:
            task = run[:-6]
            print('Do switch epochs', self.subject, self.session, run)
            self.SwitchEpochs[run] =\
                se.execute(self.subject, self.session,
                           run, task, self.flex_dir,
                           self.BehavFrame[run],
                           self.PupilFrame[run],
                           self.BrainstemRois[run], mode=mode)

    def CortexEpochs(self):
        self.CortexEpochs = defaultdict(dict)
        for run in self.runs:
            task = run[:-6]
            print('Do cort epochs', self.subject, self.session, run)
            self.CortexEpochs[run] =\
                cort.execute(self.subject, self.session,
                             run, task, self.flex_dir,
                             self.BehavFrame[run],
                             self.CortRois[run])

    def SampleCortexEpochs(self):
        self.SampleCortexEpochs = defaultdict(dict)
        for run in self.runs:
            task = run[:-6]
            print('Do cort epochs', self.subject, self.session, run)
            self.SampleCortexEpochs[run] =\
                samplecort.execute(self.subject, self.session,
                                   run, task, self.flex_dir,
                                   self.BehavFrame[run],
                                   self.CortRois[run])

    def KernEpochs(self, n):
        self.KernelEpochs = defaultdict(dict)
        for run in self.runs:
            task = 'inference'
            print('Do kernel epochs', self.subject, self.session, run)
            self.KernelEpochs[run] =\
                ke.execute(self.subject, self.session,
                           run, task, self.flex_dir,
                           self.BehavFrame[run], n)

    def AltModel(self):
        self.AlternativeModel = defaultdict(dict)
        for run in self.runs:
            task = 'inference'
            print('Do alternative model', self.subject, self.session, run)
            self.AlternativeModel[run] =\
                am.execute(self.subject, self.session,
                           run, task, self.flex_dir,
                           self.BehavFrame[run], n=2)

    def CleanEpochs(self, epoch='Choice'):
        '''
        Concatenate runs within a Session per task.
        '''
        print('Clean epochs')
        per_session = []
        for run in self.runs:
            task = run[:-6]
            if epoch == 'Choice':
                run_epochs = self.ChoiceEpochs[run]
            if epoch == 'Switch':
                run_epochs = self.SwitchEpochs[run]
            run_epochs['run'] = run
            run_epochs['task'] = task
            per_session.append(run_epochs)
        per_session = pd.concat(per_session, ignore_index=True)
        if epoch == 'Choice':
            self.CleanEpochs = ce.defit_clean(per_session)
        else:
            self.CleanEpochs = per_session

    def LinregVoxel(self):
        print('Linreg voxel')
        self.VoxelReg = defaultdict(dict)
        self.SurfaceTxt = defaultdict(dict)
        self.DesignMatrix = defaultdict(dict)
        self.Residuals = defaultdict(dict)
        for task in self.tasks:
            rs = [r for r in self.runs if r[:-6] == task]
            self.VoxelReg[task], self.SurfaceTxt[task], self.DesignMatrix[task], self.Residuals[task] =\
                lv.execute(self.subject, self.session, rs,
                           self.flex_dir,
                           self.BehavFrame, task)

    def SingleTrialGLM(self):
        print('SingleTrialGLM')
        for task in self.tasks:
            rs = [r for r in self.runs if r[:-6] == task]
            self.SingleTrial, self.TrialRules = st.execute(self.subject, self.session, rs,
                                                           self.flex_dir,
                                                           self.BehavFrame, self.Residuals[task], self.out_dir)

    def Decode(self):
        print('Decoder')
        DecodeResult = dec.execute(self.subject, self.session, trialbetas=self.SingleTrial,
                                   trialrules=pd.Series(self.TrialRules))
        DecodeResult.to_hdf(join(self.out_dir, '{0}_{1}.hdf'.format('Decoding', self.subject)), key=self.session)

    def Output(self):
        print('Output')
        for name, attribute in self.__iter__():
            if name in ['BehavFrame', 'BehavAligned', 'PupilFrame',
                        'CortRois', 'BrainstemRois', 'ChoiceEpochs', 'CortexEpochs', 'SampleCortexEpochs', 'KernelEpochs', 'Residuals', 'Sin', 'AlternativeModel']:
                for run in attribute.keys():
                    print('Saving', name, run)
                    if name == 'Residuals':
                        for task, nifti in attribute[run].items():
                            nifti.to_filename(join(self.out_dir,
                                                   '{0}_{1}_{2}_{3}_{4}.nii.gz'.
                                                   format(name,
                                                          self.subject,
                                                          self.session, run, task)))
                    else:
                        attribute[run].to_hdf(join(self.out_dir,
                                                   '{0}_{1}_{2}.hdf'.
                                                   format(name,
                                                          self.subject,
                                                          self.session)),
                                              key=run)

            elif name == 'CleanEpochs':
                print('Saving', name)
                attribute.to_hdf(join(self.out_dir, '{0}_{1}_{2}.hdf'.
                                      format(name, self.subject, self.session)),
                                 key=self.session)
            elif name in ['VoxelReg', 'SurfaceTxt']:
                for task in attribute.keys():
                    for parameter, content in attribute[task].items():
                        print('Saving', name, task, parameter)
                        if name == 'VoxelReg':
                            content.to_filename(join(self.out_dir,
                                                     '{0}_{1}_{2}_{3}_{4}.nii.gz'.
                                                     format(name, self.subject,
                                                            self.session, parameter, task)))
                        elif name == 'SurfaceTxt':
                            for hemisphere, cont in content.items():
                                cont.to_hdf(join(self.out_dir,
                                                 '{0}_{1}_{2}_{3}_{4}.hdf'.
                                                 format(name, self.subject,
                                                        self.session, parameter, hemisphere)),
                                            key=task)
            elif name == 'DesignMatrix':
                for task in attribute.keys():
                    attribute[task].to_hdf(join(self.out_dir,
                                                '{0}_{1}_{2}.hdf'.format(name, self.subject, self.session)), key=task)

            elif name in ['SingleTrial', 'TrialRules']:
                if name == 'SingleTrial':
                    attribute.to_filename(join(self.out_dir,
                                               '{0}_{1}_{2}.nii.gz'.
                                               format(name,
                                                      self.subject,
                                                      self.session)))
                elif name == 'TrialRules':
                    pd.Series(attribute).to_hdf(join(self.out_dir,
                                                     '{0}_{1}.hdf'.
                                                     format(name,
                                                            self.subject)), key=self.session)


def execute(sub, ses, environment):
    out_dir = 'Workflow/Sublevel_Behavior_{1}_{0}'.format(datetime.datetime.now().strftime("%Y-%m-%d"), environment)
    sl = SubjectLevel(sub, ses, runs=[4, 5, 6], environment=environment, out_dir=out_dir)
    sl.BehavFrames()
    #sl.KernEpochs(n=12)
    sl.Output()


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
        for ses in [2, 3]:
            pbs.pmap(execute, [(sub, ses, env)], walltime='20:00:00',
                     memory=40, nodes=1, tasks=2, name='subvert_sub-{}'.format(sub))
