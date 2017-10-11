# decim
python functions to analyse matlab data from the experiment





glaze is a python module that contains functions to \
a)extract data from the source matlab files and create feasable data structures from it (np array, pd dataframe)\
b)extract relevant data from these dataframes, arrays (e.g. reward, answers, et.)\
c)calculate models choices on the given data (model according to glaze et al 2015)\
d)calculate cross entropy errors for data and given hazard rate\
e)calculate optimal hazard rate, i.e. hazard rate with minimal cross entropy error\
