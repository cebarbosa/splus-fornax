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
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from photutils import aperture_photometry, CircularAperture

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



def make_halpha_images(survey, overwrite=False):
    bands = ["R", "F660", "I"]
    wcurves, trans = read_SPLUS_transmission_curves()
    halpha_estimator = EmLine3Filters(6562.8 * u.AA, bands, wcurves, trans)
    bands = context.bands if bands is None else bands
    survey_dir =  os.path.join(context.data_dir, survey)
    wdir = os.path.join(survey_dir, "scubes")
    outdir = os.path.join(survey_dir, "halpha_3F")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sample = os.listdir(wdir)
    idx = [context.bands.index(band) for band in bands]
    for scube in tqdm(sample, desc="Processing {}".format(survey)):
        galid = "_".join(scube.split("_")[:-1])
        output = os.path.join(outdir, "halpha3F_{}.fits".format(galid))
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
        nw.wcs.pc = w.wcs.pc[:2, :2]
        h.update(nw.to_header())
        hdu1 = fits.ImageHDU(halpha.value / bscale, h)
        hdu2 = fits.ImageHDU(halpha_err.value / bscale, h)
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2])
        hdulist.writeto(output, overwrite=True)
        # Making PNG image
        z = halpha.value
        extent = np.array([-z.shape[0]/2, z.shape[0]/2, -z.shape[1] / 2,
                           z.shape[1]/2]) * 0.55
        vmin = np.nanpercentile(z, 1)
        vmax = np.nanpercentile(z, 99)
        fig = plt.figure(figsize=(5.,4))
        ax = plt.subplot(111)
        im = ax.imshow(z, origin="lower", vmax=vmax, vmin=vmin, extent=extent)
        cbar = plt.colorbar(im)
        cbar.set_label("H$\\alpha$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
        plt.title(galid.replace("_", "\_"))
        plt.xlabel("$\Delta$X (arcsec)")
        plt.ylabel("$\Delta$Y (arcsec)")
        plt.tight_layout()
        plt.savefig(outimg)
        # plt.show()
        plt.clf()
        plt.close()

if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    surveys = ["jellyfish", "FCC", "FDS_dwarfs", "smudges2", "patricia",
               "FDS_LSB", "11HUGS"]
    for survey in surveys:
        make_halpha_images(survey)