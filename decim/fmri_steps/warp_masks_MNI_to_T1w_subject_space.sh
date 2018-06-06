#!/bin/bash
atlas_dir=~/Flexrule/fmri/atlases
cd "$atlas_dir"
for subject in {1..22}; do
    subdir=$atlas_dir/sub-$subject
    echo $subdir
    if [ ! -d "$subdir" ]; then
        mkdir "$subdir"
        for maskfile in $atlas_dir/selected/*.nii.gz; do
            outfile=$atlas_dir/sub-$subject/${maskfile:$((${#atlas_dir}+10)):$((${#maskfile}-${#atlas_dir}-17))}_T1w_sub-$subject.nii.gz
            warpfile=/Volumes/flxrl/preprocessed2/sub-"$subject"/fmriprep/sub-"$subject"/anat/sub-"$subject"_T1w_space-MNI152NLin2009cAsym_target-T1w_warp.h5
            reference_file=/Volumes/flxrl/preprocessed2/sub-"$subject"/fmriprep/sub-"$subject"/anat/sub-"$subject"_T1w_preproc.nii.gz
            antsApplyTransforms -d 3 -e 3 -i $maskfile -r $reference_file -o $outfile -n NearestNeighbor -t $warpfile -t identity -v
        done
    else
        echo "Subject $subject already exists"
    fi
done
find . -empty -type d -delete
