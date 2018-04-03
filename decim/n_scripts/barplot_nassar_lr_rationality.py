import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
sns.set_style('white')

# INPUT
df = pd.read_csv('basic_nassarframe_all05038.csv')

# TRANSFORM
bins = [-1000, -0.0001, 1, 1000]
names = ['negative', 'rational', 'hyper']
df['lr_bins'] = pd.cut(df.learning_rate, bins=bins, labels=names).values

rational = df.loc[df.lr_bins == 'rational'].groupby('subject').count().X.values
hyper = df.loc[df.lr_bins == 'hyper'].groupby('subject').count().X.values
negative = df.loc[df.lr_bins == 'negative'].groupby('subject').count().X.values
precentage = rational / (rational + hyper + negative)

# PLOT
subjects = df.subject.unique()
inds = np.arange(len(subjects))
width = 0.5

f, ax = plt.subplots(figsize=(16, 9))
plt.bar(inds, negative, width, label='Negative LR', color=sns.color_palette("GnBu_d")[5], alpha=.9)
plt.bar(inds, rational, width, bottom=negative, label='Rational LR', color=sns.color_palette("GnBu_d")[3], alpha=.9)
plt.bar(inds, hyper, width, bottom=negative + rational, label='LR > 1', color=sns.color_palette("GnBu_d")[0], alpha=.9)
sns.despine(left=True, bottom=True)
ax.set_yticks([])
ax.set_xticklabels(labels=[0, 1, 2, 3, 4, 5, 6, 7, 9])
ax.set_xlabel('Subject')
ax.set_title('''Do the subjects learn "rationally"?''', fontdict={'fontsize': 18})
for ind in inds:
    ax.text(ind, 800, "%.2f" % precentage[ind], horizontalalignment='center', fontdict={'fontsize': 16})
ax.legend()
f.savefig('barplot_nassar_lr_rational?.png', dpi=160)
