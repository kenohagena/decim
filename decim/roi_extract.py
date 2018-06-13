import pandas as pd
from nilearn import image, masking
import os
from glob import glob
from os.path import join, expanduser
import errno
import random
import string


def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


class EPI(object):

    def __init__(self, subject, out_dir=None):
        self.subject = 'sub-{}'.format(subject)
        if out_dir is not None:
            try:
                self.out_dir = join(expanduser(out_dir), self.subject)
                os.makedirs(self.out_dir)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(out_dir):
                    out_dir = join(expanduser(out_dir), '{}{}'.format(self.subject, randomword(4)))
                    os.makedirs(out_dir)
                    self.out_dir = out_dir
                else:
                    raise
        else:
            self.out_dir = os.getcwd()
        self.EPIs = {}
        self.masks = {}
        self.resampled_masks = {}
        self.epi_masked = {}
        self.weights = {}
        print('outdir is {}'.format(self.out_dir))

    def load_epi(self, dir, identifier=None):
        '''
        Find and load EPI-files.
        '''
        dir = expanduser(dir)
        if identifier is None:
            files = glob(join(dir, '*.nii*'))
        else:
            files = glob(join(dir, '*{}*.nii*'.format(identifier)))
        for file in files:
            name = file[len(dir):file.find('.nii')]
            self.EPIs[name] = image.load_img(file)

    def load_mask(self, mask_dir, identifier=None, mult_roi_atlases=None):
        '''
        Find and load ROI masks.

        Mult_roi_atlases should take the form of a dict of dicts.
        Outerkey: substring to identify atlas, innerkey: frame within that 4D Nifty, value: name of that ROI.
        '''
        mask_dir = expanduser(mask_dir)
        if identifier is None:
            files = glob(join(mask_dir, '*.nii*'))
        else:
            files = glob(join(mask_dir, '*{}*.nii*'.format(identifier)))
        if mult_roi_atlases is not None:
            files = [r for r in files if all(z not in r for z in mult_roi_atlases.keys())]
            for key, value in mult_roi_atlases.items():
                atlas = glob(join(mask_dir, '*{}*.nii*'.format(key)))
                for k, v in value.items():
                    self.masks[v] = image.index_img(atlas[0], k)
        for file in files:
            name = file[len(mask_dir) + 1:file.find('.nii')]
            self.masks[name] = image.load_img(file)

    def resample_masks(self):
        '''
        Resample masks to affine/shape of EPIs.
        '''
        epi_img = self.EPIs[list(self.EPIs.keys())[0]]
        for key, value in self.masks.items():
            self.resampled_masks['{}_resampled'.format(key)] = image.resample_img(value, epi_img.affine,
                                                                                  target_shape=epi_img.get_data().shape[0:3])

    def mask(self):
        '''
        Apply all masks to all EPIs.
        '''
        for mask, mimg in self.resampled_masks.items():
            for epi, eimg in self.EPIs.items():
                thresh = image.new_img_like(mimg, mimg.get_data() > 0.01)
                key = '{0}_{1}'.format(epi, mask)
                key = key[:key.find('_resampled')]
                self.epi_masked[key] = masking.apply_mask(eimg, thresh)
            self.weights['{0}_{1}'.format(self.subject, mask)] = masking.apply_mask(mimg, thresh)

    def save(self):
        '''
        Save results as .csv.
        '''
        for key, value in self.epi_masked.items():
            pd.DataFrame(value).to_csv(join(self.out_dir, key))
        for key, value in self.weights.items():
            pd.DataFrame(value).to_csv(join(self.out_dir, '{0}_weights'.format(key)))
