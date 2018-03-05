from glob import glob
from os.path import join
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import signal


subjects = {'VPIM01': 1, 'VPIM02': 2, 'VPIM03': 3, 'VPIM04': 4, 'VPIM06': 6, 'VPIM07': 7, 'VPIM09': 9}
sessions = {'A': 1, 'B': 2, 'C': 3}
blocks = [1, 2, 3, 4, 5, 6, 7]

# INPUT FILES: 'session-pupilframes'
pa = '/Users/kenohagena/Documents/immuno/da/decim/pupilframes/session_pupilframes_230218/'


sampling_freq = 1000
segment = 60 * sampling_freq
overlap = segment / 4

# PLOT per Subject

for subject in subjects:
    fig, ax = plt.subplots(figsize=(16, 9))
    for session in sessions:
        file = glob(join(pa, '*{0}{1}.csv'.format(subjects[subject], sessions[session])))
        df = pd.read_csv(file[0])
        x = df.biz  # interpolated, bandpassed, z-scored
        f, p = signal.welch(x, fs=sampling_freq, nperseg=segment, noverlap=overlap, detrend='constant')
        df = pd.DataFrame({'freq': f, 'power': p, 'lfreq': np.log10(f)})
        ax.loglog(df.freq, df.power, label=session)
        ax.set(title='{0} {1}'.format(subject, session), xlabel='Frequency (Hz)', ylabel='Power')
        ax.legend()
        ax.axvline(2, color='black', alpha=.3)
    sns.despine()
    fig.savefig('power_spectra_{}.png'.format(subject), dpi=160)

# PLOT for all


ses_power = {}
for session in sessions:
    powers = []
    for subject in subjects:
        file = glob(join(pa, '*{0}{1}.csv'.format(subjects[subject], sessions[session])))
        df = pd.read_csv(file[0])
        x = df.biz  # interpolated, bandpassed, z-scored
        f, p = signal.welch(x, fs=sampling_freq, nperseg=segment, noverlap=overlap, detrend='constant')
        powers.append(p)

    ses_power[session] = pd.DataFrame(powers)
freqs = f
colors = {'A': '#fdb147', 'B': '#8fb67b', 'C': '#3778bf'}

fig, ax = plt.subplots(figsize=(16, 9))
for session in sessions:
    ax.loglog(freqs, ses_power[session].mean().values, color=colors[session], label=session)
    for i, row in ses_power[session].iterrows():
        ax.loglog(freqs, row.values, color=colors[session], alpha=.1)


ax.axvline(2, color='black', alpha=.6)
ax.fill_between([1e-2, 6], 1e-11 * 4, 1e-10, color='g', alpha=.2)
ax.set(title='Power spectra all subjects', xlabel='Frequency (Hz)', ylabel='Power')
ax.legend()
sns.despine()
fig.savefig('power_spectra_all.png', dpi=160)
