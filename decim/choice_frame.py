import pandas as pd
import numpy as np
from decim import glaze2 as gl
from decim import glaze_control as gc
from os.path import join
from decim import slurm_submit as slu


summary = pd.read_csv('/Users/kenohagena/Flexrule/fmri/analyses/bids_stan_fits/summary_stan_fits.csv')


def baseline(grating, choice, length=1000):
    baseline = np.matrix((grating.loc[:, 0:length].mean(axis=1))).T
    return pd.DataFrame(np.matrix(grating) - baseline), pd.DataFrame(np.matrix(choice) - baseline), baseline


class Choiceframe(object):

    def __init__(self, sub, ses, flex_dir, run_indices=[0, 1, 2]):
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

    def choice_behavior(self):
        logs = gc.load_logs_bids(self.sub, self.ses, join(self.flex_dir, 'raw', 'bids_mr_v1.1'))
        logs = pd.concat(logs[1], ignore_index=True)

        self.rawbehavior = logs.loc[logs.event.isin(['START_GLAZE', 'GL_TRIAL_START',
                                                     'GL_TRIAL_GENSIDE', 'GL_TRIAL_LOCATION', 'GL_TRIAL_TYPE',
                                                     'SAMPLE_ONSET', 'GL_TRIAL_STIM_ID', 'CHOICE_TRIAL_ONSET',
                                                     'CHOICE_TRIAL_STIMOFF', 'CHOICE_TRIAL_RESP',
                                                     'CHOICE_TRIAL_RT', 'CHOICE_TRIAL_RULE_RESP', 'GL_TRIAL_REWARD'])].reset_index()

        df = self.rawbehavior
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
        gratingpupil = pd.DataFrame(choicepupil)
        choicepupil = pd.DataFrame(choicepupil)
        baseline_correct = baseline(gratingpupil, choicepupil)
        self.gratingpupil = baseline_correct[0]
        self.choicepupil = baseline_correct[1]
        self.parameters = pd.DataFrame(parameters)
        self.parameters.columns = (['response', 'all_artifacts', 'blink', 'run', 'onset', 'trial_id'])
        self.parameters['TPR'] = self.choicepupil.mean(axis=1)

    def merge(self):
        '''
        Merge everything. And Save.
        '''
        grating = self.gratingpupil
        choice = self.choicepupil
        paras = self.parameters
        points = self.points
        choices = self.choices
        grating.columns = pd.MultiIndex.from_product([['pupil'], ['gratinglock'], range(grating.shape[1])], names=['source', 'type', 'name'])
        choice.columns = pd.MultiIndex.from_product([['pupil'], ['choicelock'], range(choice.shape[1])], names=['source', 'type', 'name'])
        paras.columns = pd.MultiIndex.from_product([['pupil'], ['parameters'], paras.columns], names=['source', 'type', 'name'])
        choices.columns = pd.MultiIndex.from_product([['behavior'], ['parameters'], choices.columns], names=['source', 'type', 'name'])
        points.columns = pd.MultiIndex.from_product([['behavior'], ['points'], range(points.shape[1])], names=['source', 'type', 'name'])
        master = pd.concat([grating, choice, choices, points, paras], axis=1)
        self.master = master.set_index([master.pupil.parameters.trial_id, master.pupil.parameters.run])
        out_dir = join(self.flex_dir, 'pupil', 'choice_epochs')
        slu.mkdir_p(out_dir)
        self.master.to_csv(join(out_dir, 'choice_epochs_{0}_{1}.csv'.format(self.subject, self.session)))


if __name__ == '__main__':
    for sub in range(1, 23):
        for ses in [2, 3]:
            try:
                flex_dir = '/Volumes/flxrl/FLEXRULE/'
                c = Choiceframe(sub, ses, flex_dir)
                c.choice_behavior()
                c.points()
                c.choice_pupil()
                c.merge()
            except RuntimeError:
                continue


__version__ = '2.0'
'''
2.0
-Input linear pupilframes
-recquires BIDS
1.2
-triallocked period now 1000ms before offset and total of 4500ms
-if rt > 2000ms choicelocked is set to np.nan
'''
