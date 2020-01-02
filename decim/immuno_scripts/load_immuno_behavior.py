from os.path import join
from glob import glob
from io import loadmat
import pandas as pd


def load_log(sub_code, session, phase, block, base_path):
    """
    Concatenates path and file name and loads matlba file.

    Recquires subject code, session, phase and block.
    """
    directory = join(base_path,
                     "{}".format(sub_code),
                     "{}".format(session),
                     'PH_' + "{}".format(phase) +
                     'PH_' + "{}".format(block))
    files = glob(join(directory, '*.mat'))
    if len(files) > 1:
        raise RuntimeError(
            'More than one log file found for this block: %s' % files)
    elif len(files) == 0:
        raise RuntimeError(
            'No log file found for this block: %s, %s, %s, %s' %
            (sub_code, session, phase, block))
    return loadmat(files[0])


def row2dict(item):
    """
    Convert a single row of the log to a dictionary.
    """
    return {'time': item[0, 0][0, 0],
            'message': item[0, 1][0],
            'value': item[0, 2].ravel()[0],
            'phase': item[0, 3][0, 0],
            'block': item[0, 4][0, 0]}


def log2pd(log, block, key="p"):
    """
    Takes loaded matlab log and returns panda dataframe.

    Extracts only logs of the current block.
    """
    log = log["p"]['out'][0][0][0, 0][0]
    pl = [row2dict(log[i, 0]) for i in range(log.shape[0])]

    df = pd.DataFrame(pl)
    df.loc[:, 'message'] = df.message.astype('str')
    df.loc[:, 'value'] = df.value
    return df.query('block==%i' % block)
