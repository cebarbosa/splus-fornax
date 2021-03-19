# -*- coding: utf-8 -*-
"""
Author : Carlos Eduardo Barbosa

Methods to extract H-alpha in the SPLUS filters according to methods presented
in Villela-Rojo et al. 2015 (VR+15).

"""
from __future__ import division, print_function

import os

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils import aperture_photometry, CircularAperture
from skimage import color
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import cv2

import context
from emlines_estimator import EmLine3Filters

def read_SPLUS_transmission_curves():
    filters_dir = os.path.join(os.path.dirname(__file__),
                               "filter_curves-master")
    filenames = sorted([os.path.join(filters_dir, _) for _ in os.listdir( \
                        filters_dir)])
    filternames = [os.path.split(_)[1].replace(".dat", "") for _ in filenames]
    filternames = [_.replace("F0", "F").replace("JAVA", "") for _ in
                   filternames]
    filternames = [_.replace("SDSS", "").upper() for _ in filternames]
    fcurves = [np.loadtxt(f) for f in filenames]
    wcurves = [curve[:,0] * u.AA for curve in fcurves]
    trans = [curve[:,1] for curve in fcurves]
    wcurves = dict(zip(filternames, wcurves))
    trans =  dict(zip(filternames, trans))
    return wcurves, trans

def make_halpha_images(sample, overwrite=False, data_dir=None):
    bands = ["R", "F660", "I"]
    wcurves, trans = read_SPLUS_transmission_curves()
    halpha_estimator = EmLine3Filters(6562.8 * u.AA, bands, wcurves, trans)
    bands = context.bands if bands is None else bands
    data_dir = context.data_dir if data_dir is None else data_dir
    survey_dir =  os.path.join(data_dir, sample)
    wdir = os.path.join(survey_dir, "scubes")
    outdir = os.path.join(survey_dir, "halpha_3F")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    idx = [context.bands.index(band) for band in bands]
    for scube in tqdm(os.listdir(wdir), desc="Processing {}".format(sample)):
        output = os.path.join(outdir, "halpha3F_{}".format(scube))
        outimg = output.replace(".fits", ".png")
        if os.path.exists(output) and not overwrite:
            continue
        cubefile = os.path.join(wdir, scube)
        bunit = u.Unit(fits.getval(cubefile, "BUNIT", ext=1))
        bscale = fits.getval(cubefile, "BSCALE", ext=1)
        flam = fits.getdata(cubefile, ext=0)[idx] * bunit
        flamerr = fits.getdata(cubefile, ext=1)[idx] * bunit
        halpha = halpha_estimator.flux_3F(flam)
        halpha_err = halpha_estimator.fluxerr_3F(flamerr)
        # Saving fits
        h = fits.getheader(cubefile)
        w = WCS(h)
        nw = WCS(naxis=2)
        nw.wcs.cdelt= w.wcs.cdelt[:2]
        nw.wcs.crval = w.wcs.crval[:2]
        nw.wcs.crpix = w.wcs.crpix[:2]
        nw.wcs.ctype[0] = w.wcs.ctype[0]
        nw.wcs.ctype[1] = w.wcs.ctype[1]
        try:
            nw.wcs.pc = w.wcs.pc[:2, :2]
        except:
            pass
        h.update(nw.to_header())
        hdu1 = fits.ImageHDU(halpha.value / bscale, h)
        hdu2 = fits.ImageHDU(halpha_err.value / bscale, h)
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2])
        hdulist.writeto(output, overwrite=True)
        # Making PNG image
        z = halpha.value
        extent = np.array([-z.shape[0] / 2, z.shape[0] / 2, -z.shape[1] / 2,
                           z.shape[1] / 2]) * 0.55
        vmin = np.nanpercentile(z, 1)
        vmax = np.nanpercentile(z, 99)
        fig = plt.figure(figsize=(5.,4))
        ax = plt.subplot(111)
        im = ax.imshow(z, origin="lower", vmax=vmax, vmin=vmin, extent=extent)
        cbar = plt.colorbar(im)
        cbar.set_label("H$\\alpha$ (erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)")
        plt.title(scube.replace("_", "\_"))
        plt.xlabel("$\Delta$X (arcsec)")
        plt.ylabel("$\Delta$Y (arcsec)")
        plt.tight_layout()
        plt.savefig(outimg)
        # plt.show()
        plt.clf()
        plt.close()

def make_overlay_RGB_halpha(sample, data_dir=None):
    """ Overlays an RGB image of the galaxies with H-alpha. """
    data_dir = context.data_dir if data_dir is None else data_dir
    survey_dir =  os.path.join(context.data_dir, sample)
    halpha_dir = os.path.join(survey_dir, "halpha_3F")
    cubes_dir = os.path.join(survey_dir, "scubes")
    out_dir = os.path.join(survey_dir, "RGB+halpha")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    idx = [context.bands.index(band) for band in ["I", "R", "G"]]
    bb = 5
    desc = "Making RGB+halpha images of {} sample".format(sample)
    for scube in tqdm(os.listdir(cubes_dir), desc=desc):
        cube = fits.getdata(os.path.join(cubes_dir, scube))
        r = cube[idx[0]]
        g = cube[idx[1]]
        b = cube[idx[2]]
        r[np.isnan(r)] = 0.
        g[np.isnan(g)] = 0.
        b[np.isnan(b)] - 0.
        I = (b + g + r) / 3.
        beta = np.nanmedian(I) * bb
        R = r * np.arcsinh(I / beta) / I
        G = g * np.arcsinh(I / beta) / I
        B = b * np.arcsinh(I / beta) / I
        maxRGB = np.percentile(np.stack([R, G, B]), 99.5)
        R = np.clip(255 * R / maxRGB, 0, 255).astype("uint8")
        G = np.clip(255 * G / maxRGB, 0., 255).astype("uint8")
        B = np.clip(255 * B / maxRGB, 0, 255).astype("uint8")
        RGB = np.stack([np.rot90(R, 3), np.rot90(G, 3),
                        np.rot90(B, 3)]).T

        # Make h-alpha image to be superposed
        halpha_file = os.path.join(halpha_dir, "halpha3F_{}".format(scube))
        halpha = fits.getdata(halpha_file)
        mean, median, stddev = sigma_clipped_stats(halpha)
        maxha = np.percentile(halpha, 99.5)
        halpha = np.clip(halpha, 1 * stddev, maxha)
        halpha -= halpha.min()
        halpha = (halpha / halpha.max() * 255).astype("uint8")

        hamask = np.zeros_like(RGB)
        hamask[:, :, 0] = np.rot90(halpha, 3).T
        # Overlay images
        out = Image.fromarray(cv2.addWeighted(RGB, 0.9, hamask, 0.4, 0))
        outimg = os.path.join(out_dir, "RGB+halpha_{}".format(
            scube.replace(".fits", ".png")))
        out.save(outimg)

if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    samples = ["FCC", "jellyfish", "FDS_dwarfs", "smudges2", "patricia",
               "11HUGS"]
    samples += ["interacting_galaxies"]
    samples += ["FDS_UDGs"]
    samples += ["sample_gc_galaxies"]
    # samples += ["FDS_LSB"]
    for sample in samples:
        data_dir = "/home/kadu/Dropbox/splus-halpha/data" if sample=="FCC" \
            else None
        make_halpha_images(sample, data_dir=data_dir, overwrite=True)
        make_overlay_RGB_halpha(sample, data_dir=data_dir)