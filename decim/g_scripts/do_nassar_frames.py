import nassar as na
import pandas as pd

subjects = [1, 2, 3, 4, 5, 6, 7, 9]

l = []
for subject in subjects:
    for session in [1, 2, 3]:
        if (subject == 1) & (session == 1):
            n = na.Nassarframe(subject, session, '/Users/kenohagena/Documents/immuno/data/vaccine', blocks=[1, 2, 3])
        else:
            n = na.Nassarframe(subject, session, '/Users/kenohagena/Documents/immuno/data/vaccine')

        n.get_trials()
        n.concat()
        n.sessiontrials['session'] = session
        n.sessiontrials['subject'] = subject
        l.append(n.sessiontrials)


df = pd.concat(l, ignore_index=True)
df.to_csv('basic_nassarframe_all05038.csv', index=False)
