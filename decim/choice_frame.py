import pandas as pd
import numpy as np
import glaze2 as gl
import pupil_frame as pop


sessions = {1: 'A', 2: 'B', 3: 'C'}
phases = {'A': 1, 'B': 3, 'C': 5}


class Choiceframe(object):

    def __init__(self, subject, session, path, blocks=[1, 2, 3, 4, 5, 6, 7]):
        '''
        Initialize
        '''
        self.subject = 'VPIM0{}'.format(subject)
        self.sub = subject
        self.session = sessions[session]
        self.ses = session
        self.blocks = blocks
        self.path = path

    def choicetrials(self):
        blockdfs = []
        for block in self.blocks:
            df = gl.log2pd(gl.load_log(self.subject, self.session, phases[self.session], block, self.path), block)
            df = df.loc[df.message != 'BUTTON_PRESS']  # Because sometimes duplicated and not of importance
            df['block'] = float(block)
            blockdfs.append(df)
        self.rawbehavior = pd.concat(blockdfs, ignore_index=True)
        self.rawbehavior = self.rawbehavior.reset_index()
        df = self.rawbehavior
        rule_response = (df.loc[df.message == "CHOICE_TRIAL_RULE_RESP", 'value'].astype(float))
        rt = df.loc[df.message == 'CHOICE_TRIAL_RT']['value']
        stimulus = df.loc[df.message == 'GL_TRIAL_STIM_ID']['value']
        reward = df.loc[df.message == 'GL_TRIAL_REWARD']['value']
        response = df.loc[df.message == 'CHOICE_TRIAL_RESP']['value']
        onset_time = df.loc[df.message == 'CHOICE_TRIAL_ONSET']['time']
        block = df.loc[df.message == 'CHOICE_TRIAL_ONSET']['block']
        trial_inds = df.loc[df.message == 'CHOICE_TRIAL_RESP'].index
        trial_inds = sum([list(range(i - 8, i))for i in trial_inds], [])
        trial_id = df.loc[(df.message == 'GL_TRIAL_START') & (df.index.isin(trial_inds))].value

        assert len(rule_response) == len(stimulus) == len(reward) == len(response) == len(rt)
        tuples = [('behavior', 'parameter', 'rule_response'),
                  ('behavior', 'parameter', 'reaction_time'),
                  ('behavior', 'parameter', 'stimulus'),
                  ('behavior', 'parameter', 'response'),
                  ('behavior', 'parameter', 'reward'),
                  ('behavior', 'parameter', 'onset')]
        columns = pd.MultiIndex.from_tuples(tuples, names=('source', 'type', 'name'))
        self.choices = pd.DataFrame(np.full((len(rt), len(tuples)), np.nan), columns=columns)
        self.choices.loc[:, ('behavior', 'parameter', 'rule_response')] = rule_response.values
        self.choices.loc[:, ('behavior', 'parameter', 'reaction_time')] = rt.values
        self.choices.loc[:, ('behavior', 'parameter', 'stimulus')] = stimulus.values
        self.choices.loc[:, ('behavior', 'parameter', 'response')] = reward.values
        self.choices.loc[:, ('behavior', 'parameter', 'reward')] = rule_response.values
        self.choices.loc[:, ('behavior', 'parameter', 'onset')] = onset_time.values
        self.choices['block'] = block.values
        self.choices['trial_id'] = trial_id.values.astype(float)

    def points(self, n=20):
        '''
        Add last n points before choice onset.
        '''
        columns = pd.MultiIndex.from_product([['behavior'], ['points'], range(n)], names=('source', 'type', 'name'))
        df = pd.DataFrame(np.full((len(self.choices), n), np.nan), columns=columns)

        for i, row in self.choices.iterrows():
            points = self.rawbehavior
            points = points.loc[(points.message == 'GL_TRIAL_LOCATION') & (points.time < row.behavior.parameter.onset)]
            if len(points) < n:
                points = np.full(n, np.nan)
            else:
                points = points.value.values[len(points) - n:len(points)]
            df.iloc[i] = points

        df['trial_id'] = self.choices.trial_id.values
        df['block'] = self.choices.block.values
        self.points = df
        self.choices = self.choices.merge(df, how='left', on=['block', 'trial_id'])

    def glaze_belief(self, subjective_h, true_h=1 / 70):
        '''
        Computes models accumulated evidence at choice trials.
        '''
        df = self.rawbehavior
        choices = (df.loc[df.message == "CHOICE_TRIAL_RULE_RESP", 'value']
                   .astype(float))
        belief_indices = df.iloc[choices.index - 11].index
        glaze_belief_subjective = gl.belief(df, subjective_h).loc[belief_indices].values
        glaze_belief_true = gl.belief(df, true_h).loc[belief_indices].values
        self.choices.loc[:, ('behavior', 'parameter', 'subjective_evidence')] = glaze_belief_subjective
        self.choices.loc[:, ('behavior', 'parameter', 'true_evidence')] = glaze_belief_true

    def choice_pupil(self, pupilframe=None, artifact_threshhold=.2, choicelock=True, tw=4500):
        '''
        Takes existing pupilframe and makes choicepupil frame.
        If there is no existing pupilframe, a new one is created.
        '''
        frames = []
        if pupilframe != None:
            pass
        else:
            for block in self.blocks:
                p = pop.Pupilframe(self.sub, self.ses, block, self.path)
                p.basicframe()
                p.gaze_angle()
                p.all_artifacts()
                p.small_fragments()
                p.interpol()
                p.filter()
                p.z_score()
                p.pupil_frame['block'] = block
                frames.append(p.pupil_frame)
            pupilframe = pd.concat(frames, ignore_index=True)

        df = pupilframe.loc[:, ['message', 'biz', 'message_value', 'blink', 'all_artifacts', 'block', 'trial_id', 'gaze_angle']]
        choicepupil = []
        #print(len(df.loc[df.message == "CHOICE_TRIAL_ONSET"]))
        #print(len(df.loc[df.message == "RT"]))
        for onset in df.loc[df.message == "CHOICE_TRIAL_ONSET"].index:
            if len(df.iloc[onset: onset + 3500, :].loc[df.message == 'RT', 'message_value']) == 0:
                continue
            else:
                resp = df.iloc[onset - 1000: onset + 3500, :].loc[df.message == 'RT', 'message_value']
                trial = df.loc[np.arange(onset - 1000, onset + tw - 1000).astype(int), 'biz'].values
                trial = np.append(trial, df.loc[np.arange(onset, onset + resp + 1500), 'all_artifacts'].mean())
                trial = np.append(trial, df.loc[np.arange(onset, onset + resp + 1500), 'blink'].mean())
                trial = np.append(trial, df.loc[onset, 'block'])
                trial = np.append(trial, df.loc[onset + resp, 'gaze_angle'])
                trial = np.append(trial, onset)
                trial = np.append(trial, df.loc[onset, 'gaze_angle'])
                trial = np.append(trial, resp)
                trial = np.append(trial, df.loc[onset, 'trial_id'])

                choicepupil.append(trial)
        choicedf = pd.DataFrame(choicepupil)

        # subtract baseline per trial
        for i, row in choicedf.iterrows():
            choicedf.iloc[i, 0:tw] = (choicedf.iloc[i, 0:tw] - choicedf.iloc[i, 0:1000].mean())

        c1 = pd.MultiIndex.from_product([['pupil'], ['triallock'], range(tw)], names=['source', 'type', 'name'])
        c1 = pd.DataFrame(np.full((len(choicedf), tw), np.nan), columns=c1)
        c3 = pd.MultiIndex.from_product([['pupil'], ['parameter'], ['rt', 'onset', 'blink', 'all_artifacts', 'trial_id', 'block', 'onset_gaze', 'choice_gaze']], names=['source', 'type', 'name'])
        c3 = pd.DataFrame(np.full((len(choicedf), 8), np.nan), columns=c3)

        if choicelock == False:
            design = pd.concat([c1, c3], ignore_index=True)
            design = design.iloc[0:len(choicedf)]
            design.loc[:, ('pupil', 'triallock')] = choicedf.iloc[:, :tw].values
            design.loc[:, ('pupil', 'parameter')] = choicedf.iloc[:, tw:].values
            self.pupil = design

        else:
            c2 = pd.MultiIndex.from_product([['pupil'], ['choicelock'], range(2500)], names=['source', 'type', 'name'])
            c2 = pd.DataFrame(np.full((len(choicedf), 2500), np.nan), columns=c2)
            design = pd.concat([c1, c2, c3], ignore_index=True)
            design = design.iloc[0:len(choicedf)]

            design.loc[:, ('pupil', 'triallock')] = choicedf.iloc[:, :tw].values
            design.loc[:, ('pupil', 'parameter')] = choicedf.iloc[:, tw:].values

            design.pupil.parameter.tpr = np.nan

            for i, row in design.iterrows():
                reaction = int(row.pupil.parameter.rt) + 1000
                if reaction > 3000:
                    choicelock = np.full(2500, np.nan)
                else:
                    choicelock = row.pupil.triallock.iloc[reaction - 1000:reaction + 1500].values
                row.loc[('pupil', 'choicelock')] = choicelock

            design.loc[(design.pupil.parameter.blink == 0) & (design.pupil.parameter.all_artifacts < artifact_threshhold), ('pupil', 'parameter', 'tpr')] =\
                np.dot(design.loc[(design.pupil.parameter.blink == 0) & (design.pupil.parameter.all_artifacts < artifact_threshhold), ('pupil', 'choicelock')],
                       design.loc[(design.pupil.parameter.blink == 0) & (design.pupil.parameter.all_artifacts < artifact_threshhold), ('pupil', 'choicelock')].mean()) /\
                np.dot(design.loc[(design.pupil.parameter.blink == 0) & (design.pupil.parameter.all_artifacts < artifact_threshhold), ('pupil', 'choicelock')].mean(),
                       design.loc[(design.pupil.parameter.blink == 0) & (design.pupil.parameter.all_artifacts < artifact_threshhold), ('pupil', 'choicelock')].mean())

            self.pupil = design

    def merge(self):
        '''
        merge self.pupil and self.choices
        '''
        pupil = self.pupil.reset_index(drop=True)
        behavior = self.choices.reset_index(drop=True)
        self.choices = behavior.merge(right=pupil, how='left',
                                      left_on=['trial_id', 'block'],
                                      right_on=[('pupil', 'parameter', 'trial_id'), ('pupil', 'parameter', 'block')])


__version__ = '1.2'
'''
1.2
-triallocked period now 1000ms before offset and total of 4500ms
-if rt > 2000ms choicelocked is set to np.nan
'''
#c = Choiceframe(1, 2, '/Users/kenohagena/Documents/immuno/data/vaccine', blocks=[1])
# c.choicetrials()
#c.choice_pupil(choicelock=False, tw=4500)
#print(c.choicedf.iloc[:, 4500:])
# print(c.pupil.pupil.parameter)
