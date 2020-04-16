import pandas as pd
import numpy as np
from os.path import join, expanduser
from decim.adjuvant import slurm_submit as slu
from glob import glob
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import nibabel as nib
from nilearn import image, masking


class Decoder(object):

    def __init__(self, subject, trialbetas, trialrules, session,
                 flex_dir='/home/khagena/FLEXRULE'):

        self.subject = subject
        self.session = session
        self.flex_dir = flex_dir
        self.trialbetas = trialbetas
        self.trialrules = trialrules
        self.trialbetas_data = self.trialbetas.get_fdata()
        assert self.trialbetas_data.shape[3] == len(self.trialrules)

    def get_masks(self):
        self.masks = nib.load(join(self.flex_dir, 'fmri/atlases/', 'Glasser_full.nii'))
        self.masks = image.resample_img(self.masks,
                                        self.trialbetas.affine,
                                        target_shape=self.trialbetas.get_fdata().shape[0:3])

    def decode(self, index):
        roi_voxels = []
        for hemi_index in [index, index + 181]:
            img = image.index_img(self.masks, hemi_index)
            thresh = image.new_img_like(img, img.get_fdata() > 0.01)
            betas_masked = masking.apply_mask(self.trialbetas, thresh)
            roi_voxels.append(pd.DataFrame(betas_masked))
        concat = pd.concat(roi_voxels, axis=1)

        X = concat.values
        y = self.trialrules.values

        steps = [('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='linear'))]
        pipeline = Pipeline(steps)
        aucs = []
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        parameteres = {'SVM__C': np.logspace(-5, 5, 11), 'SVM__tol': np.logspace(-5, 5, 11), 'SVM__max_iter': np.logspace(3, 4.5, 4)}
        for i, (train, test) in enumerate(cv.split(X, y)):
            grid = GridSearchCV(pipeline, param_grid=parameteres, cv=4)
            grid.fit(X[train], y[train])
            # print(grid.best_params_)
            aucs.append(roc_auc_score(y[test], grid.predict(X[test])))
        print(aucs)


def execute(subject, session, trialbetas, trialrules):
    d = Decoder(subject=subject, trialbetas=trialbetas, trialrules=trialrules, session=session,
                flex_dir='/home/khagena/FLEXRULE')
    d.get_masks()
    for index in range(181):
        d.decode(index)
