# numbers with decimals

print("R=%.2f, p=%.3f" % (123.4235436, 132123.234234))


plt.rcParams['pdf.fonttype'] = 3
plt.rcParams['ps.fonttype'] = 3
sns.set(style='ticks', font_scale=1, rc={
    'axes.labelsize': 6,
    'axes.titlesize': 40,
    'xtick.labelsize': 40,
    'ytick.labelsize': 5,
    'legend.fontsize': 250,
    'axes.linewidth': 0.25,
    'xtick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.pad': 2.0,
    'ytick.minor.pad': 2.0,
    'xtick.major.pad': 2.0,
    'xtick.minor.pad': 2.0,
    'axes.labelpad': 4.0,
})
subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=None)


pd.set_option("display.max_rows", 101)


with pd.HDFStore(data_store) as hdf:
    print(hdf.keys())


def randomword(length):
    import random
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))
