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

def make_detection_images(survey, overwrite=False, maxdim=1000):
    survey_dir =  os.path.join(context.data_dir, survey)
    wdir = os.path.join(survey_dir, "cutouts")
    outdir = os.path.join(survey_dir, "detection")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sample = os.listdir(wdir)
    for galaxy in tqdm(sample, desc="Processing {}".format(survey)):
        galdir = os.path.join(wdir, galaxy)
        obsfields = set([_.split("_")[-4] for _ in os.listdir(galdir)])
        dims = set([_.split("_")[-2] for _ in os.listdir(galdir)])
        for dim in dims:
            xymax = np.max([float(_) for _ in dim.split("x")])
            if xymax >= maxdim:
                continue
            for obsfield in obsfields:
                output = os.path.join(outdir, "{}_{}_{}_detect.fits".format(
                                      galaxy, obsfield, dim))
                outpng = output.replace(".fits", ".png")
                if os.path.exists(outpng) and not overwrite:
                    continue
                imgs = [os.path.join(galdir, _) for _ in os.listdir(galdir) if
                        dim in _ and obsfield in _ and _.endswith("swp.fits")]
                imgs = [_ for _ in imgs if os.path.exists(_) and
                        os.path.exists(_.replace("swp", "swpweight"))]
                headers = [fits.getheader(img, ext=1) for img in imgs]
                gain = np.array([h["GAIN"] for h in headers])
                wimgs = [_.replace("swp", "swpweight") for _ in imgs]
                data = np.array([fits.getdata(_, ext=0) for _ in imgs])
                weights = np.array([fits.getdata(os.path.join(galdir, _), ext=0)
                                 for _ in wimgs])
                error = 1 / weights + np.clip(data, 0, np.infty) / gain[:,
                                                                     None, None]
                mask = np.where(data == 0)
                detection = np.nansum(data, axis=(0,))
                deterr = np.sqrt(np.nansum(error ** 2, axis=(0,)))
                sigma = detection / deterr
                title = "galaxy: {}; field: {}".format(
                         galaxy.replace("_", " "), obsfield)
                ydim, xdim = sigma.shape
                extent = np.array([-0.5 * xdim, 0.5 * xdim, -0.5 * ydim,
                                  0.5 * ydim]) * context.ps.value
                plt.imshow(gaussian_filter(sigma, 1), origin="lower",
                           cmap="viridis", vmax=5, vmin=0, extent=extent)
                plt.colorbar()
                plt.title(title)
                plt.xlabel("$\Delta$ X (arcsec)")
                plt.ylabel("$\Delta$ Y (arcsec)")
                plt.savefig(outpng, dpi=250)
                plt.clf()


        # output = os.path.join(outdir, galaxy.replace(".fits", ".png"))
        # if os.path.exists(output) and not overwrite:
        #     continue
        # data = fits.getdata(os.path.join(wdir, galaxy), ext=0)
        # error = fits.getdata(os.path.join(wdir, galaxy), ext=1)
        #
        # data[mask] = np.nan
        # detection = np.nansum(data, axis=(0,))
        # deterr = np.sqrt(np.nansum(error**2, axis=(0,)))
        # sigma = detection / deterr
        # plt.imshow(gaussian_filter(sigma, 2), origin="lower", vmax=1, vmin=-1,
        #            cmap="Spectral")
        # plt.colorbar()
        # plt.title(galid.replace("_", "\_"))
        # plt.tight_layout()
        # plt.savefig(output, dpi=250)
        # plt.clf()

if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    surveys = ["FDS_LSB"]
    # surveys = ["smudges2", "patricia", "FDS_dwarfs", "FCC", \
    #           "11HUGS", "jellyfish"]
    for survey in surveys:
        make_detection_images(survey, overwrite=False)