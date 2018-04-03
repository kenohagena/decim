import pandas as pd


sdict = {'A': 1, 'B': 2, 'C': 3}


def sessions():
    for sub in ['VPIM01', 'VPIM02', 'VPIM03', 'VPIM04', 'VPIM06', 'VPIM07', 'VPIM09']:
        for ses in ['A', 'B', 'C']:
            yield(sub, ses)


count = 0
for sub, ses in sessions():
    df = pd.read_csv('pupilframes/ctriallocked/ctriallock_pupil_{0}{1}.csv'.format(sub, ses))
    df['subject'] = sub[5]
    df['session'] = sdict[ses]
    if count == 0:
        dfc = df
    else:
        dfc = pd.concat([dfc, df], ignore_index=True)
    print(sub)
    count += 1


dfc.to_csv('pupilframes/ctriallocked/allsubs_ctriallocked.csv', index=False)
