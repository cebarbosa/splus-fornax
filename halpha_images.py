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
from photutils import aperture_photometry, CircularAperture

import context

def halpha_3F_method(f660, fr, fi):
    """Three bands method from VR+2015 (equation (3)). """
    # Constants used in the calculation using S-PLUS bands.
    a = 1.29075002747
    betas = {"I": 1.1294371504165924e-07 / u.AA,
             'F660': 0.006619605316773919 / u.AA,
             "R" : 0.0007578642946167448 / u.AA}
    numen = (fr - fi) - a * (f660 - fi)
    b = - betas["F660"] * a + betas["R"]
    return numen / b

def halpha_3F_error(f660err, frerr, fierr):
    a = 1.29075002747
    betas = {"I": 1.1294371504165924e-07 / u.AA,
             'F660': 0.006619605316773919 / u.AA,
             "R" : 0.0007578642946167448 / u.AA}
    b = - betas["F660"] * a + betas["R"]
    var = ((1 + a) / b * frerr)**2 + (fierr / b)**2 + (a / b * f660err)**2
    return np.sqrt(var)

def make_halpha_images(survey, overwrite=False, bands=None):
    flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
    fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
    bands = context.bands if bands is None else bands
    wave = np.array([context.wave_eff[band] for band in bands]) * u.Angstrom
    survey_dir =  os.path.join(context.data_dir, survey)
    wdir = os.path.join(survey_dir, "scubes")
    outdir = os.path.join(survey_dir, "halpha_3F")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sample = os.listdir(wdir)
    # sample = [_ for _ in sample if _.startswith("FCC00036")]
    idx = [context.bands.index(band) for band in ["F660", "R", "I"]]
    for scube in tqdm(sample, desc="Processing {}".format(survey)):
        galid = "_".join(scube.split("_")[:-1])
        output = os.path.join(outdir, "{}_detection.png".format(galid))
        if os.path.exists(output) and not overwrite:
            continue
        cubefile = os.path.join(wdir, scube)
        bunit = u.Unit(fits.getval(cubefile, "BUNIT", ext=1))
        bscale = fits.getval(cubefile, "BSCALE", ext=1)
        flam = fits.getdata(cubefile, ext=0) * bscale * bunit
        flamerr = fits.getdata(cubefile, ext=1) * bscale * bunit
        fnu = flam / const.c * wave[:, None, None]**2
        fnuerr = flamerr / const.c * wave[:, None, None]**2
        fnu = fnu.to(fnu_unit)
        fnuerr = fnuerr.to(fnu_unit)
        f660 = fnu[idx[0]]
        r = fnu[idx[1]]
        i = fnu[idx[2]]
        f660err = fnuerr[idx[0]]
        rerr = fnuerr[idx[1]]
        ierr = fnuerr[idx[2]]
        halpha = halpha_3F_method(f660, r, i)
        halpha_err = halpha_3F_error(f660err, rerr, ierr)
        plt.imshow(halpha.value, origin="lower", vmax=np.nanpercentile(
            halpha.value, 99.5), vmin=np.nanpercentile(halpha.value, 1))
        cbar = plt.colorbar()
        cbar.set_label("H$\\alpha$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
        plt.title(galid.replace("_", "\_"))
        plt.tight_layout()
        # outdir = "/home/kadu/Dropbox/talks/2020-11-30-splus-meeting/figs"
        # plt.savefig("{}.png".format(os.path.join(outdir, galid)), dpi=250)
        plt.show()
        # ydim, xdim = halpha.shape
        # x0 = xdim / 2
        # y0 = ydim / 2
        # rs = np.logspace(0, np.log10(xdim/2), 10)
        # apertures = [CircularAperture((x0,y0), r=r) for r in rs]
        # phot_table = aperture_photometry(halpha, apertures, error=halpha_err)
        # for i in range(10):
        #     plt.plot(rs[i], phot_table["aperture_sum_{}".format(i)], "o")
        # plt.show()
        # print(phot_table)
        # input()

if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    surveys = ["jellyfish", "FCC", "FDS_dwarfs", "smudges2", "patricia",
               "FDS_LSB", "11HUGS"]
    for survey in surveys:
        make_halpha_images(survey)