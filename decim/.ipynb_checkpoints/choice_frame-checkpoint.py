import pandas as pd
import numpy as np
from decim import glaze2 as gl
from decim import glaze_control as gc
from os.path import join
from decim import slurm_submit as slu
from decim import fmri_align as fa
from collections import defaultdict
from tqdm import tqdm

summary = pd.read_csv('/Users/kenohagena/Flexrule/fmri/analyses/bids_stan_fits/summary_stan_fits.csv')


def baseline(grating, choice, length=1000):
    baseline = np.matrix((grating.loc[:, 0:length].mean(axis=1))).T
    return pd.DataFrame(np.matrix(grating) - baseline), pd.DataFrame(np.matrix(choice) - baseline), baseline


class Choiceframe(object):

    def __init__(self, sub, ses, flex_dir, run_indices=[0, 1, 2], master=None):
        '''
        Initialize
        '''
        self.sub = sub
        self.ses = ses
        self.subject = 'sub-{}'.format(sub)
        self.session = 'ses-{}'.format(ses)
        self.flex_dir = flex_dir
        inf_runs = np.array(['inference_run-4',
                             'inference_run-5',
                             'inference_run-6'])
        self.runs = inf_runs[run_indices]
        self.hazard = summary.loc[(summary.subject == self.subject) & (summary.session == self.session)].hmode.values
        self.master = master
        if self.master is not None:
            self.choice_behavior = self.master.behavior.parameters.drop('run', axis=1).reset_index('run')
            self.points = self.master.behavior.points
            self.gratingpupil = self.master.pupil.gratinglock
            self.choicepupil = self.master.pupil.choicelock
            self.parameters = self.master.pupil.parameters

    def choice_behavior(self):
        logs = gc.load_logs_bids(self.sub, self.ses, join(self.flex_dir, 'raw', 'bids_mr_v1.1'))
        logs = pd.concat(logs[1], ignore_index=True)

        self.rawbehavior = logs.loc[logs.event.isin(['START_GLAZE', 'GL_TRIAL_START',
                                                     'GL_TRIAL_GENSIDE', 'GL_TRIAL_LOCATION', 'GL_TRIAL_TYPE',
                                                     'SAMPLE_ONSET', 'GL_TRIAL_STIM_ID', 'CHOICE_TRIAL_ONSET',
                                                     'CHOICE_TRIAL_STIMOFF', 'CHOICE_TRIAL_RESP',
                                                     'CHOICE_TRIAL_RT', 'CHOICE_TRIAL_RULE_RESP', 'GL_TRIAL_REWARD'])].reset_index()

        df = self.rawbehavior
        df = df.replace('n/a', np.nan)
        rule_response = df.loc[df.event == "CHOICE_TRIAL_RULE_RESP", 'value']
        rt = df.loc[df.event == 'CHOICE_TRIAL_RT']['value'].astype('float').values
        stimulus = df.loc[df.event == 'GL_TRIAL_STIM_ID']['value'].astype('float').values
        reward = df.loc[df.event == 'GL_TRIAL_REWARD']['value'].astype('float').values
        response = df.loc[df.event == 'CHOICE_TRIAL_RESP']['value'].astype('float').values
        onset = df.loc[df.event == 'CHOICE_TRIAL_ONSET']['onset'].astype('float').values
        block = df.loc[df.event == 'CHOICE_TRIAL_ONSET']['block'].astype('float').values
        trial_inds = df.loc[df.event == 'CHOICE_TRIAL_RESP'].index
        trial_inds = sum([list(range(i - 8, i))for i in trial_inds], [])
        trial_id = df.loc[(df.event == 'GL_TRIAL_START') & (df.index.isin(trial_inds))].value.astype('float').values
        belief_indices = df.loc[rule_response.index - 11].index.values
        belief = gl.belief(df, self.hazard, ident='event').loc[belief_indices].values
        choices = pd.DataFrame({'rule_response': rule_response.astype('float').values,
                                'rt': rt,
                                'stimulus': stimulus,
                                'response': response,
                                'reward': reward,
                                'onset': onset,
                                'run': block,
                                'trial_id': trial_id,
                                'accumulated_belief': belief})
        self.choice_behavior = choices

    def points(self, n=20):
        '''
        Add last n points before choice onset.
        '''
        p = []
        points = self.rawbehavior.loc[(self.rawbehavior.event == 'GL_TRIAL_LOCATION')]
        for i, row in self.choice_behavior.iterrows():
            trial_points = points.loc[points.onset.astype('float') < row.onset]
            if len(trial_points) < 20:
                trial_points = np.full(20, np.nan)
            else:
                trial_points = trial_points.value.values[len(trial_points) - 20:len(trial_points)]
            p.append(trial_points)
        points = pd.DataFrame(p)
        points['trial_id'] = self.choice_behavior.trial_id.values
        points['run'] = self.choice_behavior.run.values
        self.points = points

    def choice_pupil(self, artifact_threshhold=.2, choicelock=True, tw=4500):
        '''
        Takes existing pupilframe and makes choicepupil frame.
        If there is no existing pupilframe, a new one is created.
        '''
        frames = []
        for run in self.runs:
            pupil_frame = pd.read_csv(join(self.flex_dir, 'pupil', 'linear_pupilframes',
                                           'pupilframe_{0}_{1}_{2}.csv'.format(self.subject, self.session, run)))
            pupil_frame['run'] = run
            frames.append(pupil_frame)
        pupil_frame = pd.concat(frames, ignore_index=True)
        df = pupil_frame.loc[:, ['message', 'biz', 'message_value', 'blink', 'all_artifacts', 'run', 'trial_id', 'gaze_angle']]

        gratingpupil = []
        choicepupil = []
        parameters = []
        for choice_trial in df.loc[df.message == "CHOICE_TRIAL_ONSET"].index:
            '''
            Extract gratinglocked pupilresponse, choicelocked pupil response & choice parameters
            '''
            onset = choice_trial
            if len(df.iloc[onset: onset + 3500, :].loc[df.message == 'RT', 'message_value']) == 0:
                continue
            else:
                resp = df.iloc[onset - 1000: onset + 3500, :].loc[df.message == 'RT', 'message_value']
                gratinglock = df.loc[np.arange(onset - 1000, onset + tw - 1000).astype(int), 'biz'].values
                choice_parameters = [df.iloc[onset - 1000: onset + 3500, :].loc[df.message == 'RT', 'message_value'].values,
                                     df.loc[np.arange(onset, onset + resp + 1500), 'all_artifacts'].mean(),
                                     df.loc[np.arange(onset, onset + resp + 1500), 'blink'].mean(),
                                     df.loc[onset, 'run'],
                                     onset, df.loc[onset, 'trial_id']]
                choicelock = df.loc[np.arange(onset + resp - 1000, onset + resp + 1500).astype(int), 'biz'].values
                gratingpupil.append(gratinglock)
                choicepupil.append(choicelock)
                parameters.append(choice_parameters)
        gratingpupil = pd.DataFrame(gratingpupil)
        choicepupil = pd.DataFrame(choicepupil)
        baseline_correct = baseline(gratingpupil, choicepupil)
        self.gratingpupil = baseline_correct[0]
        self.choicepupil = baseline_correct[1]
        self.parameters = pd.DataFrame(parameters)
        self.parameters.columns = (['response', 'all_artifacts', 'blink', 'run', 'onset', 'trial_id'])
        self.parameters['TPR'] = self.choicepupil.mean(axis=1)
        clean_mean = self.parameters.loc[(self.parameters.blink == 0) & (self.parameters.all_artifacts < 0.5)].TPR.mean()
        self.parameters['TPR_JW'] = self.parameters.TPR * clean_mean / np.square(clean_mean)

    def fmri_epochs(self, basel=2000, te=6):
        evoked_session = defaultdict(list)
        for run in self.runs:
            roi = pd.read_csv(join(flex_dir, 'fmri', 'roi_extract', 'weighted', '{0}_{1}_{2}_weighted_rois.csv'.format(self.sub, self.session, run)), index_col=0)
            roi = roi.loc[:, ['AAN_DR', 'basal_forebrain_4',
                              'basal_forebrain_123', 'LC_standard', 'NAc', 'SNc',
                              'VTA']]
            dt = pd.to_timedelta(roi.index.values * 1900, unit='ms')
            roi = roi.set_index(dt)
            target = roi.resample('1ms').mean().index
            roi = pd.concat([fa.interp(dt, roi[c], target) for c in roi.columns], axis=1)
            behav = self.choice_behavior.loc[self.choice_behavior.run == float(run[-1])]
            onsets = behav.onset.values
            trial_ids = behav.trial_id.values
            evoked_run = defaultdict(list)
            for onset in onsets:
                cue = pd.Timedelta(onset, unit='s').round('ms')
                bl = pd.Timedelta(basel, unit='ms')
                baseline = roi.loc[cue - bl: cue + bl].mean()
                task_evoked = roi.loc[cue - bl: cue + bl * te] - baseline
                for col in task_evoked.columns:
                    evoked_run[col].append(task_evoked[col].values)
            for key, values in evoked_run.items():
                df = pd.DataFrame(values)
                df['trial_id'] = trial_ids
                df['onset'] = onsets
                df['run'] = run
                if run != 'inference_run-6':
                    evoked_session[key].append(df)
                else:
                    evoked_session[key].append(df)
                    evoked_session[key] = pd.concat(evoked_session[key], ignore_index=True)
            self.roi_task_evoked = evoked_session

    def merge(self):
        '''
        Merge everything. And Save.
        '''
        grating = self.gratingpupil
        choice = self.choicepupil
        paras = self.parameters
        points = self.points
        choices = self.choice_behavior
        grating.columns = pd.MultiIndex.from_product([['pupil'], ['gratinglock'], range(grating.shape[1])], names=['source', 'type', 'name'])
        choice.columns = pd.MultiIndex.from_product([['pupil'], ['choicelock'], range(choice.shape[1])], names=['source', 'type', 'name'])
        paras.columns = pd.MultiIndex.from_product([['pupil'], ['parameters'], paras.columns], names=['source', 'type', 'name'])
        choices.columns = pd.MultiIndex.from_product([['behavior'], ['parameters'], choices.columns], names=['source', 'type', 'name'])
        points.columns = pd.MultiIndex.from_product([['behavior'], ['points'], range(points.shape[1])], names=['source', 'type', 'name'])
        singles = [grating, choice, choices, points, paras]
        master = pd.concat(singles, axis=1)
        master = master.set_index([master.pupil.parameters.trial_id, master.pupil.parameters.run])
        singles = []
        for key, frame in self.roi_task_evoked.items():
            frame.columns = pd.MultiIndex.from_product([['fmri'], [key], frame.columns], names=['source', 'type', 'name'])
            singles.append(frame)
        fmri = pd.concat(singles, axis=1)
        fmri = fmri.set_index([fmri.fmri.AAN_DR.trial_id, fmri.fmri.AAN_DR.run])
        merge = pd.merge(fmri.reset_index(), master.reset_index()).set_index(['trial_id', 'run'])
        self.master = merge
        out_dir = join(self.flex_dir, 'pupil', 'choice_epochs')
        slu.mkdir_p(out_dir)
        self.master.to_csv(join(out_dir, 'choice_epochs_{0}_{1}.csv'.format(self.subject, self.session)))


if __name__ == '__main__':
    print('jessa')
    dfs = []
    for sub in tqdm(range(1, 23)):
        for ses in [2, 3]:
            try:
                flex_dir = '/Volumes/flxrl/FLEXRULE/'
                c = Choiceframe(sub, ses, flex_dir)
                c.choice_behavior()
                print('behavior')
                c.points()
                print('points')
                c.choice_pupil()
                print('pupil')
                c.fmri_epochs()
                print('fmri. ready for the big merge')
                c.merge()
                c.master['subject'] = c.subject
                c.master['session'] = c.session
                sd = c.master.set_index(['subject', 'session'], append=True)
                dfs.append(sd)
                

            except RuntimeError:
                pass
            except FileNotFoundError:
                pass
    df = pd.concat(dfs)
    df = df[dfs[0].columns] # https://github.com/pandas-dev/pandas/issues/4588
    df.to_csv('/Volumes/flxrl/FLEXRULE/pupil/choice_epochs/choice_epochs_concatenated.csv')

__version__ = '2.0'
'''
2.0
-Input linear pupilframes
-recquires BIDS
1.2
-triallocked period now 1000ms before offset and total of 4500ms
-if rt > 2000ms choicelocked is set to np.nan





    import subprocess
    subprocess.call(['osascript', '-e',
   'tell app "System Events" to shut down'])




                   singles=[]
                for key, frame in c.roi_task_evoked.items():
                    frame.columns = pd.MultiIndex.from_product([['fmri'], [key], frame.columns], names=['source', 'type', 'name'])
                    singles.append(frame)
                fmri = pd.concat(singles, axis=1)
                fmri = fmri.set_index([fmri.fmri.AAN_DR.trial_id, fmri.fmri.AAN_DR.run])
                merge = pd.merge(fmri.reset_index(), master.reset_index()).set_index(['trial_id', 'run'])
                merge.to_csv('/Volumes/flxrl/FLEXRULE/pupil/choice_epochs/choice_epochs_sub-{0}_ses-{1}.csv'.format(sub, ses))
'''
