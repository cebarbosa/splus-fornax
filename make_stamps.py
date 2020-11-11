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

def make_stamps_jype(names, coords, sizes, outdir=None, redo=False,
                     img_types=None, bands=None):
    """  Produces stamps of objects in S-PLUS from a table of names,
    coordinates.

    Parameters
    ----------
    names: np.array
        Array containing the name/id of the objects.

    coords: astropy.coordinates.SkyCoord
        Coordinates of the objects.

    size: np.array
        Size of the stamps (in pixels)

    outdir: str
        Path to the output directory. If not given, stamps are saved in the
        current directory.

    redo: bool
        Option to rewrite stamp in case it already exists.

    img_types: list
        List containing the image types to be used in stamps. Default is [
        "swp', "swpweight"] to save both the images and the weight images
        with uncertainties.

    bands: list
        List of bands for the stamps. Defaults produces stamps for all
        filters in S-PLUS. Options are 'U', 'F378', 'F395', 'F410', 'F430', 'G',
        'F515', 'R', 'F660', 'I', 'F861', and 'Z'.


    """
    names = np.atleast_1d(names)
    sizes = np.atleast_1d(sizes)
    if len(sizes) == 1:
        sizes = np.full(len(names), sizes[0])
    sizes = sizes.astype(np.int)
    img_types = ["swp", "swpweight"] if img_types is None else img_types
    outdir = os.getcwd() if outdir is None else outdir
    tiles_dir = "/mnt/jype/MainSurvey/tiles/T01"
    header_keys = ["OBJECT", "FILTER", "EXPTIME", "GAIN", "TELESCOP",
                   "INSTRUME", "AIRMASS"]
    bands = context.bands if bands is None else bands
    ############################################################################
    # Selecting tiles from S-PLUS footprint
    fields = Table.read(os.path.join(context._path, "data",
                                     "all_tiles_final.csv"))
    field_coords =  SkyCoord(fields["RA"], fields["DEC"],
                                unit=(u.hourangle, u.degree))
    idx, d2d, d3d = field_coords.match_to_catalog_sky(coords)
    idx = np.where(d2d < 2 * u.degree)
    fields = fields[idx]
    ############################################################################
    # Producing stamps
    for field in tqdm(fields, desc="Fields"):
        field_coords = SkyCoord(field["RA"], field["DEC"],
                                unit=(u.hourangle, u.degree))
        field_name = field["NAME"]
        d2d = coords.separation(field_coords)
        idx = np.where(d2d < 2 * u.degree)[0]
        fnames = names[idx]
        fcoords = coords[idx]
        fsizes = sizes[idx]
        for img_type in tqdm(img_types, desc="Data types", leave=False,
                             position=1):
            for band in tqdm(bands, desc="Bands", leave=False, position=2):
                tile_dir = os.path.join(tiles_dir, field["NAME"], band)
                fitsfile = os.path.join(tile_dir, "{}_{}_{}.fits".format(
                                         field["NAME"], band, img_type))
                if not os.path.exists(fitsfile):
                    continue
                header = fits.getheader(fitsfile)
                wcs = WCS(header)
                xys = wcs.all_world2pix(fcoords.ra, fcoords.dec, 1)
                data = fits.getdata(fitsfile)
                for i, (name, size) in enumerate(tqdm(zip(fnames, fsizes),
                                     desc="Galaxies", leave=False, position=3)):
                    galdir = os.path.join(outdir, name)
                    output = os.path.join(galdir,
                             "{0}_{1}_{2}_{3}x{3}_{4}.fits".format(
                              name, field_name, band, size, img_type))
                    if os.path.exists(output) and not redo:
                        continue
                    try:
                        cutout = Cutout2D(data, position=fcoords[i],
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
                    hdu.header["OBJECT"] = name
                    if "HIERARCH OAJ PRO FWHMMEAN" in header:
                        hdu.header["PSFFWHM"] = header["HIERARCH OAJ PRO FWHMMEAN"]
                    hdu.header["X0TILE"] = (xys[0][i], "Location in tile")
                    hdu.header["Y0TILE"] = (xys[1][i], "Location in tile")
                    hdu.header.update(cutout.wcs.to_header())
                    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
                    if not os.path.exists(galdir):
                        os.mkdir(galdir)
                    hdulist.writeto(output, overwrite=True)

def make_stamps_fcc():
    """Produces stamps for sample of galaxies in the FCC (Ferguson+ 1990)."""
    fcc_cat = Table.read(os.path.join(context.tables_dir, "FCC_sizes.fits"))
    fcc_names = np.array([ _ for _ in fcc_cat["Object"]])
    ras =  np.array([ _ for _ in fcc_cat["RA"]])
    decs = np.array([ _ for _ in fcc_cat["DEC"]])
    wdir = os.path.join(context.data_dir, "FCC")
    outdir = os.path.join(wdir, "cutouts")
    for _dir in [wdir, outdir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)
    sizes = np.array(1 + 1.2 * fcc_cat["Major Axis"].data /
                     context.ps.value).astype(np.int)
    sizes = np.clip(sizes.astype(np.int), 256, np.infty)
    # Duplicating large galaxies
    idx = np.where(sizes > 256)
    fcc_names = np.append(fcc_names, fcc_names[idx])
    ras = np.append(ras, ras[idx])
    decs = np.append(decs, decs[idx])
    sizes = np.append(np.full(len(fcc_cat), 256), sizes[idx])
    fcc_coords = SkyCoord(ras, decs, unit=(u.degree, u.degree))
    make_stamps_jype(fcc_names, fcc_coords, sizes, outdir=outdir)

def make_stamps_FDS_lsb():
    catname = os.path.join(context.tables_dir, "Venhola2017_FDS_LSB.dat")
    with open(catname) as f:
        data = f.readlines()
    data = [_ for _ in data if not _.startswith("#")]
    names = np.array([_[7:19].strip() for _ in data])
    ras = np.array([float(_[22:30]) for _ in data]) * u.degree
    decs = np.array([float(_[31:38])  for _ in data]) * u.degree
    res = np.array([float(_[74:80])  for _ in data])
    sizes = (np.floor(1 + 2 * 6 * res / context.ps.value)).astype(np.int)
    coords = SkyCoord(ras, decs)
    wdir = os.path.join(context.data_dir, "FDS_LSB")
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    outdir = os.path.join(wdir, "cutouts")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    make_stamps_jype(names, coords, sizes, outdir=outdir, redo=True)

def make_stamps_11HUGS():
    catname = os.path.join(context.tables_dir, "kennicutt2008.fits")
    outdir = os.path.join(context.data_dir, "stamps_11HUGS")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sample = Table.read(catname)
    names = [_ for _ in sample["Name"]]
    coords = SkyCoord(sample["RAJ2000"], sample["DEJ2000"], unit=u.degree)
    fov = 14. * u.arcmin
    sizes = fov.to(u.arcsec).value / context.ps.value
    make_stamps_jype(names, coords, sizes, outdir=outdir)

def make_stamps_FDS_dwarfs_idr3():
    catname = os.path.join(context.tables_dir, "Venhola2018_FDS_dwarfs.dat")
    with open(catname) as f:
        data = f.readlines()
    data = data[152:] # Removing header
    names = np.array([_[0:14].strip() for _ in data])
    ras = np.array([float(_[16:23]) for _ in data]) * u.degree
    decs = np.array([float(_[24:32])  for _ in data]) * u.degree
    res = np.array([float(_[126:135])  for _ in data]) * u.arcsec
    sizes = (np.floor(1 * u.pix + 2 * 3 * res / context.ps)).astype(
        np.int).value
    coords = SkyCoord(ras, decs)
    wdir = os.path.join(context.data_dir, "FDS_dwarfs")
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    outdir = os.path.join(wdir, "cutouts")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    make_stamps_jype(names, coords, sizes, outdir=outdir, redo=True)

if __name__ == "__main__":
    make_stamps_fcc()
    make_stamps_FDS_lsb()
    # make_stamps_11HUGS()
    # make_stamps_FDS_dwarfs_idr3()