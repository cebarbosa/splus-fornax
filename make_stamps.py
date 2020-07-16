# -*- coding: utf-8 -*-
"""

Created on 06/07/2020

Author : Carlos Eduardo Barbosa

Produces stamps of galaxies.

"""

from __future__ import print_function, division

import os
import warnings

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from tqdm import tqdm
from astropy.wcs import FITSFixedWarning

import context

warnings.simplefilter('ignore', category=FITSFixedWarning)

def stamps_fcc(redo=False, size0=64):
    """ Produces stamps for sample of galaxies in the FCC (Ferguson+ 1990). """
    header_keys = ["OBJECT", "FILTER", "EXPTIME", "GAIN", "TELESCOP",
                   "INSTRUME", "AIRMASS"]
    outdir = os.path.join(context.data_dir, "cutouts")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    img_types = ["swp", "swpweight"]
    tiles_dir = "/mnt/jype/MainSurvey/tiles/T01"
    catalog = Table.read(os.path.join(context.tables_dir, "FCC_sizes.fits"))
    fcc_coords = SkyCoord(catalog["RA"].data, catalog["DEC"].data,
                          unit=(u.degree, u.degree))
    ############################################################################
    # Selecting tiles for Fornax cluster
    fields = Table.read(os.path.join(context.tables_dir,
                                                "fornax_splus_tiles.fits"))
    ############################################################################
    # Producing stamps
    for field in tqdm(fields, desc="Fields"):
        field_coords = SkyCoord(field["RA"], field["DEC"], unit=(u.hourangle,
                                                                 u.degree))
        field_name = field["NAME"].split("-")[1]
        d2d = fcc_coords.separation(field_coords)
        idx = np.where(d2d < 2 * u.degree)[0]
        cat = catalog[idx]
        catcoords = fcc_coords[idx]
        for img_type in tqdm(img_types, desc="Data types", leave=False,
                             position=1):
            for band in tqdm(context.bands, desc="Bands", leave=False,
                             position=2):
                tile_dir = os.path.join(tiles_dir, field["NAME"], band)
                fitsfile = os.path.join(tile_dir, "{}_{}_{}.fits".format(
                                         field["NAME"], band, img_type))
                if not os.path.exists(fitsfile):
                    continue
                header = fits.getheader(fitsfile)
                wcs = WCS(header)
                data = fits.getdata(fitsfile)
                for i, c in enumerate(tqdm(cat, desc="Galaxies", leave=False,
                                 position=3)):
                    sizes = [size0]
                    size_pix = int(1.2 * c["Major Axis"] / context.ps.value) + 1
                    if size_pix > size0:
                        sizes.append(size_pix)
                    for size in sizes:
                        galdir = os.path.join(outdir, c["Object"])
                        output = os.path.join(galdir,
                                 "{0}_{1}_{2}_{3}x{3}_{4}.fits".format(
                            c["Object"], field_name, band, size, img_type))
                        if os.path.exists(output) and not redo:
                            continue
                        try:
                            cutout = Cutout2D(data, position=catcoords[i],
                                      size=size * u.pixel, wcs=wcs)
                        except ValueError:
                            continue
                        if np.all(cutout.data == 0):
                            continue
                        hdu = fits.ImageHDU(cutout.data)
                        for key in header_keys:
                            if key in header:
                                hdu.header[key] = header[key]
                        hdu.header["TILE"] = hdu.header["OBJECT"]
                        hdu.header["OBJECT"] = c["Object"]
                        if "HIERARCH OAJ PRO FWHMMEAN" in header:
                            hdu.header["PSFFWHM"] = header["HIERARCH OAJ PRO FWHMMEAN"]
                        hdu.header.update(cutout.wcs.to_header())
                        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
                        if not os.path.exists(galdir):
                            os.mkdir(galdir)
                        hdulist.writeto(output, overwrite=True)

if __name__ == "__main__":
    stamps_fcc()