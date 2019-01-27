from decim import localizer as loco
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


# DATA
localizer = []
for name in ['keno', 'niklas']:
    L = loco.Localizer("/Users/kenohagena/Flexrule/immuno/data/localizer/localizer_{}.edf".format(name))
    L.basicframe()
    L.gaze_angle()
    L.all_artifacts()
    L.small_fragments()
    L.interpol()
    L.filter()
    # L.demean()
    L.z_score()
    L.reframe(tw=8000, which='biz')
    L.reframe['localizer'] = name
    localizer.append(L.reframe)
df = pd.concat(localizer, ignore_index=True)
drop = df.dropna(how='any')
clean = drop.loc[(drop.blink == 0) & (drop.all_artifacts < .1)]
clean.to_csv('/Users/kenohagena/Desktop/clean_local.csv')

# PLOT
plt.rcParams['pdf.fonttype'] = 3
plt.rcParams['ps.fonttype'] = 3
sns.set(style='ticks', font_scale=1, rc={
    'axes.labelsize': 20,
    'axes.titlesize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 12,
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'ytick.major.width': 2,
    'ytick.major.width': 2,
    'ytick.major.pad': 2.0,
    'ytick.minor.pad': 2.0,
    'xtick.major.pad': 2.0,
    'xtick.minor.pad': 2.0,
    'axes.labelpad': 4.0,
})

palette = sns.color_palette("Blues")
f, ax = plt.subplots(figsize=[9, 6])
plt.plot(clean.loc[clean.localizer == 'keno'].iloc[:, 0:8000].mean().values, lw=5, color=palette[1])
plt.plot(clean.loc[clean.localizer == 'niklas'].iloc[:, 0:8000].mean().values, lw=5, color=palette[3])
ax.annotate('Onset', xy=(1000, -.05), xytext=(1500, .3),
            arrowprops=dict(facecolor='black', shrink=0.05), size=15)
ax.annotate('Offset', xy=(3000, -.5), xytext=(3500, .3),
            arrowprops=dict(facecolor='black', shrink=0.05), size=15)

ax.set(xticks=[-1000, 1000, 3000, 5000, 7000], xticklabels=[-2, 0, 2, 4, 6], xlim=[-1500, 9000],
       yticks=[-1.5, 0], ylim=[-1.7, .8],
       xlabel='Time (s)', ylabel='Pupil diameter',
       title='Pupil response on grating')
sns.despine(trim=True)
f.savefig('/Users/kenohagena/Flexrule/fmri/plots/pupil/localizer_plot_zs.png', dpi=160)
