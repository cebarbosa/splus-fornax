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
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
from tqdm import tqdm

import context

def make_cubes(indir, outdir, redo=False):
    """ Get results from cutouts and join them in a cube. """
    filenames = os.listdir(indir)
    galaxy = filenames[0].split("_")[0]
    fields = set([_.split("_")[1] for _ in filenames])
    sizes = set([_.split("_")[3] for _ in filenames])
    bands = context.bands
    dtypes = ["swp", "swpweight"]
    ext = {"swp": "DATA", "swpweight": "WEIGHTS"}
    hfields = ["GAIN", "PSFFWHM", "DATE-OBS"]
    for field, size in itertools.product(fields, sizes):
        cubename = os.path.join(outdir, "{}_{}_{}.fits".format(galaxy, field,
                                                               size))
        if os.path.exists(cubename) and not redo:
            continue
        # Producing data cubes HDUs.
        hdulist = []
        for i, dtype in enumerate(dtypes):
            imgs = [os.path.join(indir,
                    "{}_{}_{}_{}_{}.fits".format(galaxy, field, band, size,
                                                   dtype)) for band in bands]
            if not all([os.path.exists(_) for _ in imgs]):
                continue
            data = np.array([fits.getdata(img, 1) for img in imgs])
            # Making WCS
            h = fits.getheader(imgs[0], 1)
            w = WCS(h)
            nw = WCS(naxis=3)
            nw.wcs.cdelt[:2] = w.wcs.cdelt
            nw.wcs.crval[:2] = w.wcs.crval
            nw.wcs.crpix[:2] = w.wcs.crpix
            nw.wcs.ctype[0] = w.wcs.ctype[0]
            nw.wcs.ctype[1] = w.wcs.ctype[1]
            nw.wcs.pc[:2,:2] = w.wcs.pc
            h.update(nw.to_header())
            hdu = fits.ImageHDU(data, nw.to_header())
            hdu.header["EXTNAME"] = ext[dtype]
            # Producing metadata table
            if i == 0:
                hs = [fits.getheader(img, 1) for img in imgs]
                # Adding header of R band in first extension
                hr = hs[bands.index("R")]
                hdulist.append(fits.PrimaryHDU(header=hr))
                # Making table with metadata
                tab = []
                tab.append(bands)
                tab.append([context.wave_eff[band] for band in bands])
                tab.append([context.exptimes[band] for band in bands])
                for f in hfields:
                    tab.append([h[f] for h in hs])
                tab = Table(tab, names=["FILTER", "WAVE_EFF", "EXPTIME"] +
                                       hfields)
                thdu = fits.BinTableHDU(tab)
                thdu.header["EXTNAME"] = "METADATA"
            hdulist.append(hdu)
        if len(hdulist) == 3:
            hdulist.append(thdu)
            # Saving the cube
            hdulist = fits.HDUList(hdulist)
            hdulist.writeto(cubename, overwrite=True)

if __name__ == "__main__":
    warnings.simplefilter('ignore', category=AstropyWarning)
    cutouts_dir = os.path.join(context.data_dir, "cutouts")
    cubes_dir = os.path.join(context.data_dir, "scubes")
    outdir = os.path.join(cubes_dir, "sky-subtracted-uncalibrated")
    for _dir in [cubes_dir, outdir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)
    galaxies = sorted(os.listdir(cutouts_dir))
    for galaxy in tqdm(galaxies, desc="Producing data cubes"):
        wdir = os.path.join(cutouts_dir, galaxy)
        make_cubes(wdir, outdir, redo=True)