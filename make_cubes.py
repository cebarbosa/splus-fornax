# -*- coding: utf-8 -*-
"""

Created on 03/09/2020

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import itertools
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import astropy.constants as const
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
from tqdm import tqdm

import context

def make_cubes(indir, outdir, redo=False, bands=None, bscale=1e-19):
    """ Get results from cutouts and join them in a cube. """
    filenames = os.listdir(indir)
    galaxy = os.path.split(indir)[1]
    fields = set([_.split("_")[-4] for _ in filenames])
    sizes = set([_.split("_")[-2] for _ in filenames])
    bands = context.bands if bands is None else bands
    wave = np.array([context.wave_eff[band] for band in bands]) * u.Angstrom
    flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
    fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
    ext = {"swp": "DATA", "swpweight": "WEIGHTS"}
    hfields = ["GAIN", "PSFFWHM", "DATE-OBS"]
    for field, size in itertools.product(fields, sizes):
        cubename = os.path.join(outdir, "{}_{}_{}.fits".format(galaxy, field,
                                                               size))
        if os.path.exists(cubename) and not redo:
            continue
        # Loading and checking images
        imgs = [os.path.join(indir, "{}_{}_{}_{}_swp.fits".format(galaxy,
                field,  band, size)) for band in bands]
        if not all([os.path.exists(_) for _ in imgs]):
            continue
        # Checking if images have calibration available
        headers = [fits.getheader(img, ext=1) for img in imgs]
        if not all(["MAGZP" in h for h in headers]):
            continue
        # Checking if weight images are available
        wimgs = [os.path.join(indir, "{}_{}_{}_{}_swpweight.fits".format(
                 galaxy, field,  band, size)) for band in bands]
        has_errs = all([os.path.exists(_) for _ in wimgs])
        # Making new header with WCS
        h = headers[0].copy()
        del h["FILTER"]
        del h["MAGZP"]
        w = WCS(h)
        nw = WCS(naxis=3)
        nw.wcs.cdelt[:2] = w.wcs.cdelt
        nw.wcs.crval[:2] = w.wcs.crval
        nw.wcs.crpix[:2] = w.wcs.crpix
        nw.wcs.ctype[0] = w.wcs.ctype[0]
        nw.wcs.ctype[1] = w.wcs.ctype[1]
        try:
            nw.wcs.pc[:2, :2] = w.wcs.pc
        except:
            pass
        h.update(nw.to_header())
        # Performin calibration
        m0 = np.array([h["MAGZP"] for h in headers])
        gain = np.array([h["GAIN"] for h in headers])
        f0 = np.power(10, -0.4 * (48.6 + m0))
        data = np.array([fits.getdata(img, 1) for img in imgs])
        fnu = data * f0[:, None, None] * fnu_unit
        flam = fnu * const.c / wave[:, None, None]**2
        flam = flam.to(flam_unit).value / bscale
        if has_errs:
            weights = np.array([fits.getdata(img, 1) for img in wimgs])
            dataerr = 1 / weights + np.clip(data, 0, np.infty) / gain[:, None, None]
            fnuerr= dataerr * f0[:, None, None] * fnu_unit
            flamerr = fnuerr * const.c / wave[:, None, None] ** 2
            flamerr = flamerr.to(flam_unit).value / bscale
        # Making table with metadata
        tab = []
        tab.append(bands)
        tab.append([context.wave_eff[band] for band in bands])
        tab.append([context.exptimes[band] for band in bands])
        names = ["FILTER", "WAVE_EFF", "EXPTIME"]
        for f in hfields:
            if not all([f in h for h in headers]):
                continue
            tab.append([h[f] for h in headers])
            names.append(f)
        tab = Table(tab, names=names)
        # Producing data cubes HDUs.
        hdus = [fits.PrimaryHDU()]
        hdu1 = fits.ImageHDU(flam, h)
        hdu1.header["EXTNAME"] = ("DATA", "Name of the extension")
        hdus.append(hdu1)
        if has_errs:
            hdu2 = fits.ImageHDU(flamerr, h)
            hdu2.header["EXTNAME"] = ("ERRORS", "Name of the extension")
            hdus.append(hdu2)
        for hdu in hdus:
            hdu.header["BSCALE"] = (bscale, "Linear factor in scaling equation")
            hdu.header["BZERO"] = (0, "Zero point in scaling equation")
            hdu.header["BUNIT"] = ("{}".format(flam_unit),
                                   "Physical units of the array values")
        thdu = fits.BinTableHDU(tab)
        hdus.append(thdu)
        thdu.header["EXTNAME"] = "METADATA"
        hdulist = fits.HDUList(hdus)
        hdulist.writeto(cubename, overwrite=True)

if __name__ == "__main__":
    warnings.simplefilter('ignore', category=AstropyWarning)
    np.seterr(divide='ignore', invalid='ignore')
    surveys = ["smudges2", "FDS_dwarfs", "FDS_LSB", "patricia", "11HUGS",
               "FCC", "jellyfish", "FDS_UDGs"]
    surveys = ["interacting_galaxies"]
    for survey in surveys:
        cutouts_dir = os.path.join(context.data_dir, survey, "cutouts")
        cubes_dir = os.path.join(context.data_dir, survey, "scubes")
        if not os.path.exists(cubes_dir):
            os.mkdir(cubes_dir)
        galaxies = sorted(os.listdir(cutouts_dir))
        desc = "Producing data cubes for {}".format(survey)
        for galaxy in tqdm(galaxies, desc=desc):
            wdir = os.path.join(cutouts_dir, galaxy)
            make_cubes(wdir, cubes_dir, redo=True)