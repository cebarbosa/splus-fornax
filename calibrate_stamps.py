""" Use iDR3 zero points and field corrections to calibrate stamps. """
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

import context

def get_zps():
    """ Load all tables with zero points for iDR3. """
    _dir = os.path.join(context._path, "data/zps_idr3")
    tables = []
    for fname in os.listdir(_dir):
        filename = os.path.join(_dir, fname)
        data = np.genfromtxt(filename, dtype=None)
        with open(filename) as f:
            h = f.readline().replace("#", "").replace("SPLUS_", "").split()
        table = Table(data, names=h)
        tables.append(table)
    zptable = vstack(tables)
    return zptable

def get_zp_correction():
    """ Get corrections of zero points for location in the field. """
    x0, x1, nbins = 0, 9200, 16
    xgrid = np.linspace(x0, x1, nbins+1)
    zpcorr = {}
    for band in context.bands:
        corrfile = os.path.join(context._path, "data/zpcorr_idr3/SPLUS_{}"
                                       "_offsets_grid.npy".format(band))
        corr = np.load(corrfile)
        zpcorr[band] = RectBivariateSpline(xgrid, xgrid, corr)
    return zpcorr

def calibrate_samples(samples, tile_key="TILE", xycorr=True):
    comment = "Magnitude zero point"
    zps = get_zps()
    zpcorr = get_zp_correction()
    for sample in samples:
        wdir = os.path.join(context.data_dir, sample, "cutouts")
        if not os.path.exists(wdir):
            continue
        galaxies = sorted(os.listdir(wdir))
        desc = "Calibrating galaxies of sample {}".format(sample)
        for galaxy in tqdm(galaxies, desc=desc):
            galdir = os.path.join(wdir, galaxy)
            stamps = sorted([_ for _ in os.listdir(galdir) if _.endswith(".fits")])
            for stamp in stamps:
                filename = os.path.join(galdir, stamp)
                h = fits.getheader(filename, ext=1)
                tile = h[tile_key]
                filtername = h["FILTER"]
                idx = np.where((zps["FIELD"]==tile))[0]
                if len(idx) == 0:
                    print(tile)
                    continue
                zp = float(zps[idx][filtername].data[0])
                if xycorr:
                    x0 = h["X0TILE"]
                    y0 = h["Y0TILE"]
                    zp += float(zpcorr[filtername](x0, y0)[0][0])
                fits.setval(filename, "MAGZP", value=round(zp, 7),
                            comment=comment, ext=1)

if __name__ == "__main__":
    samples = ["smudges2", "FCC", "FDS_dwarfs", "FDS_LSB", "11HUGS", "patricia"]
    samples += ["jellyfish"]
    samples += ["FDS_UDGs"]
    calibrate_samples(samples)
    samples = ["interacting_galaxies"]
    calibrate_samples(samples, tile_key="OBJECT", xycorr=False)
