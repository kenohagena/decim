module purge
module load site/hummel
module load env/2017Q2-gcc-openmpi
module load libpng12

export HOME=/work/faty014
source /home/faty014/.bashrc

fmriprep --participant_label sub-1 -w /work/faty014/fmriprep_work --fs-license-file /work/faty014/license.txt /work/faty014/bids_mr/ /work/faty014/preprocessed/ participant



