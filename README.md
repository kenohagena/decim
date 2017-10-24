# decim
contains python modules for analyzing data from immuno/decision study


glaze.py contains functions to

a)extract data from the source matlab files and create feasable data structures from it (np array, pd dataframe)\
b)calculate models choices on the given data (model according to glaze et al 2015)\
c)calculate cross entropy errors for data and given hazard rate\
d)calculate optimal hazard rate, i.e. hazard rate with minimal cross entropy error\

pointsimulation.py is a module to

a)simulate a data set similar to those presented to the subjects\
b)let the glaze model guess the distribution source (given a hazard rate H)\
c)evaluate how the model performs (percentage of correct answers)\
