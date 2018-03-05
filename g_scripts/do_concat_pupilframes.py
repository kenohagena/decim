import pandas as pd
count = 1

for sub in ['VPIM01', 'VPIM02', 'VPIM03', 'VPIM04', 'VPIM06', 'VPIM07', 'VPIM09']:
    for ses in ['A', 'B', 'C']:
        count = 0

        for blo in [1, 2, 3, 4, 5, 6, 7]:
            if (sub == 'VPIM03') & (ses == 'B') & (blo == 7):
                continue
            else:
                df = pd.read_csv('pupilframes/deblink/pupil_frame_{0}{1}{2}.csv'.format(sub, ses, blo))
                pupilside = df.columns[0][3:]
                df['pupil'] = df['pa_{}'.format(pupilside)]
                red = df.loc[:, ['pupil', 'message', 'message_value',
                                 'cor_lib', 'just_blinks', 'subject',
                                 'session', 'block', 'triple_clean']]
                if count == 0:
                    dfr = red
                else:
                    dfr = pd.concat([dfr, red], ignore_index=True)

                count += 1
                print(count)

        dfr['clear_pupil'] = dfr.loc[(dfr.just_blinks == True) & (dfr.cor_lib == True), 'pupil']
        dfr['z_pupil'] = (dfr['clear_pupil'] - dfr.clear_pupil.mean()) / dfr.clear_pupil.std()
        dfr.to_csv('pupil_{0}{1}'.format(sub, ses))
        print('d')
