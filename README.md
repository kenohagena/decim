# decim
Contains python modules for analyzing data from immuno/decision study


Module glaze2.py

functionality consists basically in 3 steps:

1. data is extracted from a matlab file using the function load_log()\
2. log2pd creaets a panda dataframe from the previously loaded log using log2pd()\
3. two computations of interest can be done:\
    3.1. cross_entropy_error() computes the cross entropy error given a
    dataframe and a Hazardrate\
    3.2 optimal_h() minimizes the cross entropy error and returns the
    optimal hazardrate



Module pointsimulation.py

basically 3 steps again:

1. data is simulated using fast_sim() function\
2. cross entropy error and optimal h are computed analog to glaze2.py
    using the functions cer() and opt_h()\
3. h_iter() repeats this process n-times and for a list of
    true generating hazardrates and returns a numpy matrix


