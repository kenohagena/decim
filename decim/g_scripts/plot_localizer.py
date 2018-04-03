import localizer as loco
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


# DATA
localizer = []
for name in ['keno', 'niklas']:
    L = loco.Localizer("/Users/kenohagena/Documents/immuno/data/localizer/localizer_{}.edf".format(name))
    L.basicframe()
    L.gaze_angle()
    L.all_artifacts()
    L.small_fragments()
    L.interpol()
    L.filter()
    #L.demean()
    L.z_score()
    L.reframe(tw=8000, which = 'biz')
    L.reframe['localizer'] = name
    localizer.append(L.reframe)
df = pd.concat(localizer, ignore_index=True)
drop = df.dropna(how='any')
clean = drop.loc[(drop.blink == 0) & (drop.all_artifacts < .1)]


# PLOT
f, ax = plt.subplots(figsize=(16, 9))
plt.plot(clean.loc[clean.localizer == 'keno'].iloc[:, 0:8000].mean().values)
plt.plot(clean.loc[clean.localizer == 'niklas'].iloc[:, 0:8000].mean().values)
#for i, row in clean.iterrows():
 #   plt.plot(row.values[0:12000], color='grey', alpha=.1)
ax.axvline(1000, color='black', alpha=.6)
ax.axvline(3000, color='black', alpha=.6)
f.savefig('localizer_plot_zs.png', dpi=160)
