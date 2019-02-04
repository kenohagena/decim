#!/bin/bash
subject=$1
atlas_dir=$2
cd "$atlas_dir"
subdir=$atlas_dir/$subject
echo $subdir
if [ ! -d "$subdir" ]; then
    mkdir "$subdir"
    for maskfile in $atlas_dir/selected/*.nii.gz; do
        outfile=$atlas_dir/$subject/${maskfile:$((${#atlas_dir}+10)):$((${#maskfile}-${#atlas_dir}-17))}_T1w_$subject.nii.gz
        warpfile=/Volumes/flxrl/FLEXRULE/fmri/completed_preprocessed/"$subject"/fmriprep/"$subject"/anat/"$subject"_T1w_space-MNI152NLin2009cAsym_target-T1w_warp.h5
        reference_file=/Volumes/flxrl/FLEXRULE/fmri/completed_preprocessed/"$subject"/fmriprep/"$subject"/anat/"$subject"_T1w_preproc.nii.gz
        antsApplyTransforms -d 3 -e 3 -i $maskfile -r $reference_file -o $outfile -n NearestNeighbor -t $warpfile -t identity -v
    done
else
    echo "$subject already exists"
fi
find . -empty -type d -delete
