FLEXRULE CODE

GENERAL STRUCTURE (not very stringent so far...):
        ```
        DECIM
        │   README.md
        │   setup.py
        │
        └───decim
            │   general_functionality
            │
            └───immuno_scripts
            │       Immuno_related_analysis_steps
            │
            └───fmri_steps
                    fMRI_related_analysis_steps
        ```

Analysis steps and scripts:

1. Behavior:

    1.1 Fit stan model to behavioral data per session
    SCRIPT: decim/stan_bids.py
    INPUT: original bids_mr directory
    Dependencies: environment with pystan (currently best option hummel-cluster)
    OUTPUT: .csv file per subject-session
    OUTPUT_DIR: bids_stan_fits

    1.2 Concatenate stan fits and plot them
    SCRIPT: decim/fmri_steps/behav_stan_fits_plot_summary.py
    INPUT: .csv files with stan_fits per subject-session
    Dependencies: normal python environment
    OUTPUT: summary .csv file
    OUTPUT_DIR: bids_stan_fits

    1.3 Behavior dataframes
    SCRIPT: decim/fmri_steps/behavior_make_dfs.py
    INPUT: bids_mr and summary-file for stan fits
    Dependencies: normal python environment
    OUTPUT: behav_$sub_$ses_$run.csv file per subject-session-run

    1.3 Convolve and resample behavior
    SCRIPT: decim/fmri_align.py
    INPUT: behav_$sub_$ses_$run.csv files
    OUTPUT: pandas dataframe with EPI timedeltas as index and behavioral parameters as columns --> beh_regressors_$sub_$ses_$run.csv

2. fMRI:

    2.1 fmriprep
    SCRIPT: submit_fmriprep_man_x.py
    Dependencies: singularity (lisa-cluster) or manual dependencies (in 'beta' on hummel)
    INPUT: BIDS_mr
    OUTPUT: $sub w/ fmriprep results
    OUTPUT_DIR: completed_preprocessed

    2.2 Atlases MNI152 --> T1w Subject space
    SCRIPT: decim/fmri_steps/warp_masks_MNI_to_T1w_subject_space.sh
    INPUT: Atlases in MNI-space (atlases/selected/)
    Dependencies: antsRegistration
    OUTPUT: $atlas_T1w_$sub.nii per atlas & subject
    OUTPUT_DIR: atlases/$sub

    2.3 ROI-Extract
    SCRIPT: decim/fmri_steps/do_roi_extracts
    INPUT: preprocessed EPIs in T1w space, masks in T1w space
    OUTPUT: per subject:
                - single roi-extracted pd.DataFrame per run & atlas
                - single weights.csv - file per run & atlas
                - $sub_rois_indexed.csv with ROI-BOLD-Series for all atlases and all runs with pd.MultiIndex
                - $sub_$atlas_weights.csv-files per atlas & subject
    OUTPUT_DIR: roi_extract/$sub

3. Pupil

    3.1 Pupil dataframes
    SCRIPT: decim/fmri_steps/flexrule_pupil_dfs.py




