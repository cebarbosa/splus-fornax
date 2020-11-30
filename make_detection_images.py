""" Produces detection images for stamps of galaxies. """
from __future__ import print_function, division

import os
from subprocess import call

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

import context

def make_detection_img(filenames, outfile, bands=None):
    """ Use cutouts of stamps to produce a detection image. """
    bands = context.bands if bands is None else bands
    data = np.array([fits.getdata(fname) for fname in filenames if
                     os.path.exists(fname)])
    data = np.sum(data, axis=0)
    # Save detection image to a fits file preserving WCS
    idx = bands.index("R")
    h = fits.getheader(filenames[idx])
    h["EXTNAME"] = ("DETECTION", "Image produced by summing all bands")
    hdu = fits.PrimaryHDU(data, h)
    hdul = fits.HDUList([hdu])
    hdul.writeto(outfile, overwrite=True)
    return

def make_detection_images(survey, overwrite=False):
    survey_dir =  os.path.join(context.data_dir, survey)
    wdir = os.path.join(survey_dir, "scubes")
    outdir = os.path.join(survey_dir, "detection")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sample = os.listdir(wdir)
    for scube in tqdm(sample, desc="Processing {}".format(survey)):
        galid = "_".join(scube.split("_")[:-1])
        output = os.path.join(outdir, scube.replace(".fits", ".png"))
        if os.path.exists(output) and not overwrite:
            continue
        data = fits.getdata(os.path.join(wdir, scube), ext=0)
        error = fits.getdata(os.path.join(wdir, scube), ext=1)
        mask = np.where(data == 0)
        data[mask] = np.nan
        detection = np.nansum(data, axis=(0,))
        deterr = np.sqrt(np.nansum(error**2, axis=(0,)))
        sigma = detection / deterr
        plt.imshow(gaussian_filter(sigma, 2), origin="lower", vmax=1, vmin=-1,
                   cmap="Spectral")
        plt.colorbar()
        plt.title(galid.replace("_", "\_"))
        plt.tight_layout()
        plt.savefig(output, dpi=250)
        plt.clf()

if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    surveys = ["smudges2", "patricia", "FDS_dwarfs", "FDS_LSB", "FCC", "11HUGS"]
    surveys = ["jellyfish"]
    for survey in surveys:
        make_detection_images(survey, overwrite=False)