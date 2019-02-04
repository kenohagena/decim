import mne
import os


'''
Get HCP MMP Parcellation for a set of subjects

Functions from pymeg.atlas_glasser
'''


def get_hcp(subjects_dir):
    mne.datasets.fetch_hcp_mmp_parcellation(
        subjects_dir=subjects_dir, verbose=True)


def get_hcp_annotation_combined(subjects_dir, subject):
    for hemi in ['lh', 'rh']:
        # transform atlas to individual space:
        cmd = 'source $FREESURFER_HOME/sources.sh; mris_apply_reg --src-annot {} --trg {} --streg {} {}'.format(
            os.path.join(subjects_dir, 'fsaverage', 'label',
                         '{}.HCPMMP1_combined.annot'.format(hemi)),
            os.path.join(subjects_dir, subject, 'label',
                         '{}.HCPMMP1_combined.annot'.format(hemi)),
            os.path.join(subjects_dir, 'fsaverage', 'surf',
                         '{}.sphere.reg'.format(hemi)),
            os.path.join(subjects_dir, subject, 'surf', '{}.sphere.reg'.format(hemi)),)
        os.system(cmd)


def get_hcp_annotation(subjects_dir, subject):
    for hemi in ['lh', 'rh']:
        # transform atlas to individual space:
        cmd = 'source $FREESURFER_HOME/sources.sh; mris_apply_reg --src-annot {} --trg {} --streg {} {}'.format(
            os.path.join(subjects_dir, 'fsaverage', 'label',
                         '{}.HCPMMP1.annot'.format(hemi)),
            os.path.join(subjects_dir, subject, 'label',
                         '{}.HCPMMP1.annot'.format(hemi)),
            os.path.join(subjects_dir, 'fsaverage', 'surf',
                         '{}.sphere.reg'.format(hemi)),
            os.path.join(subjects_dir, subject, 'surf', '{}.sphere.reg'.format(hemi)),)
        os.system(cmd)


def execute(subject):
    subject_dir = '/Volumes/flxrl/FLEXRULE/fmri/completed_preprocessed/{}/freesurfer'.format(subject)
    # get_hcp(subject_dir)
    get_hcp(subject_dir)
    get_hcp_annotation(subject_dir, subject)
