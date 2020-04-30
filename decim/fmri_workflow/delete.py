import pandas as pd
for sub in range(23):
    for ses in [2, 3]:
        for run in [7, 8]:
            try:
                behav = pd.read_hdf('/home/khagena/FLEXRULE/Workflow/Sublevel_GLM_Climag_2020-01-20/sub-{0}/BehavFrame_sub-{0}_ses-{1}.hdf'.format(sub, ses),
                                    key='instructed_run-{}'.format(run))
                behav.loc[behav.event == 'REWARDED_RULE_STIM'].loc[:, ['onset', 'rewarded_rule']].to_csv('/home/khagena/behav_csv/instructed_rule_switch_sub-{0}_ses-{1}_instructed_run-{2}'.format(sub, ses, run))

            except FileNotFoundError:
                continue
            except KeyError:
                continue
