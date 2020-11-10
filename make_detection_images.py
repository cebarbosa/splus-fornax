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

def make_detection_images_FDS_LSB():
    wdir = os.path.join(context.data_dir, "FDS_LSB")
    outdir = os.path.join(context.data_dir, "FDS_LSB_detection")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sample = os.listdir(wdir)
    for galaxy in tqdm(sample, desc="Processing FDS LSB stamps"):
        galdir = os.path.join(wdir, galaxy)
        dims = set([_.split("_")[4] for _ in os.listdir(galdir)])
        tiles = set([_.split("_")[2] for _ in os.listdir(galdir)])
        for tile in tiles:
            for dim in dims:
                imgs = [os.path.join(galdir, "{}_{}_{}_{}_swp.fits".format(
                    galaxy, tile, band, dim)) for band in context.bands]
                if not all([os.path.exists(img) for img in imgs]):
                    continue
                outimg = os.path.join(outdir, "{}_{}_{}_swp.fits".format(
                                       galaxy, tile, dim))
                make_detection_img(imgs, outimg)
                pngfile = os.path.join(outimg.replace(".fits", ".png"))
                data = fits.getdata(outimg)
                vmin = np.percentile(data, 80)
                vmax = np.percentile(data, 95)
                plt.imshow(data, origin="bottom", vmin=vmin, vmax=vmax)
                plt.tight_layout()
                plt.savefig(pngfile)

if __name__ == "__main__":
    make_detection_images_FDS_LSB()