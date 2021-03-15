"""
Make stamps using t80share machine.
"""
from __future__ import print_function, division

import os
import warnings
from getpass import getpass

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

def make_stamps_t80share(names, coords, sizes, outdir=None, redo=False,
                     img_types=None, bands=None, tiles_dir=None):
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
    tiles_dir = "/storage/share/all_coadded/" if tiles_dir is None else \
        tiles_dir
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
        tile_dir = os.path.join(tiles_dir, field["NAME"])
        d2d = coords.separation(field_coords)
        idx = np.where(d2d < 2 * u.degree)[0]
        fnames = names[idx]
        fcoords = coords[idx]
        fsizes = sizes[idx]
        for img_type in tqdm(img_types, desc="Data types", leave=False,
                             position=1):
            for band in tqdm(bands, desc="Bands", leave=False, position=2):

                fitsfile = os.path.join(tile_dir, "{}_{}_{}.fits".format(
                                         field["NAME"], band, img_type))
                fzfile = fitsfile.replace(".fits", ".fz")
                if os.path.exists(fitsfile):
                    header = fits.getheader(fitsfile)
                    data = fits.getdata(fitsfile)
                elif os.path.exists(fzfile):
                    f = fits.open(fzfile)[1]
                    header = f.header
                    data = f.data
                else:
                    continue
                wcs = WCS(header)
                xys = wcs.all_world2pix(fcoords.ra, fcoords.dec, 1)
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
                        hdu.header["PSFFWHM"] = header["HIERARCH OAJ " \
                                                       "PRO FWHMMEAN"]
                    hdu.header["X0TILE"] = (xys[0][i], "Location in tile")
                    hdu.header["Y0TILE"] = (xys[1][i], "Location in tile")
                    hdu.header.update(cutout.wcs.to_header())
                    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
                    if not os.path.exists(galdir):
                        os.mkdir(galdir)
                    hdulist.writeto(output, overwrite=True)

def make_stamps_GCs():
    """ Sample of local galaxies with GC samples. """
    catalog = Table.read(os.path.join(context.tables_dir, "sample_GCs.fits"))
    names = np.array(["NGC" + "{}".format(_.split()[1]).zfill(6) for _
                      in catalog["galaxy"]])
    ras =  np.array([ _ for _ in catalog["ra"]])
    decs = np.array([ _ for _ in catalog["dec"]])
    wdir = os.path.join(context.data_dir, "GCs")
    outdir = os.path.join(context.data_dir, wdir, "cutouts")
    for _dir in [wdir, outdir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)
    sizes = np.array(1 + 6 * catalog["size"].data /
                     context.ps.value).astype(np.int)
    coords = SkyCoord(ras, decs, unit=(u.degree, u.degree))
    make_stamps_t80share(names, coords, sizes, outdir=outdir)

def make_stamps_fornax():


if __name__ == "__main__":
    make_stamps_GCs()
    # make_stamps_fornax()