# -*- coding: utf-8 -*-
"""

Created on 03/04/19

Author : Carlos Eduardo Barbosa

Produces colored images for the stamps of the galaxies

"""
from __future__ import print_function, division

import os

import numpy as np
from PIL import Image
from astropy.io import fits
from tqdm import tqdm

import context

def make_stamp(rimg, gimg, bimg, outimg, bb=2.):
    r = fits.getdata(rimg, hdu=1)
    g = fits.getdata(gimg, hdu=1)
    b = fits.getdata(bimg, hdu=1)
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
    img = Image.fromarray(RGB)
    img.save(outimg)
    return

def make_stamps_FCC():
    wdir = os.path.join(context.data_dir, "cutouts")
    outdir = os.path.join(context.data_dir, "RGB")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for i in tqdm(range(350), desc="Processing FCC catalog"):
        galaxy = "FCC{:05d}".format(i)
        galdir = os.path.join(wdir, galaxy)
        if not os.path.exists(galdir):
            continue
        dims = set([_.split("_")[3] for _ in os.listdir(galdir)])
        tiles = set([_.split("_")[1] for _ in os.listdir(galdir)])
        for tile in tiles:
            for dim in dims:
                rimg = os.path.join(galdir, "{}_{}_I_{}_swp.fits".format(
                    galaxy, tile, dim))
                gimg = os.path.join(galdir, "{}_{}_R_{}_swp.fits".format(
                    galaxy, tile, dim))
                bimg = os.path.join(galdir, "{}_{}_G_{}_swp.fits".format(
                    galaxy, tile, dim))
                if not all(os.path.exists(img) for img in [rimg, gimg, bimg]):
                    continue
                outimg = os.path.join(outdir, "{}_{}_{}.png".format(galaxy,
                                                                   tile, dim))
                make_stamp(rimg, gimg, bimg, outimg)

def make_stamps_FDS_LSB():
    wdir = os.path.join(context.data_dir, "FDS_LSB")
    outdir = os.path.join(context.data_dir, "FDS_LSB_RGB")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sample = os.listdir(wdir)
    for galaxy in tqdm(sample, desc="Processing FDS LSB stamps"):
        galdir = os.path.join(wdir, galaxy)
        dims = set([_.split("_")[4] for _ in os.listdir(galdir)])
        tiles = set([_.split("_")[2] for _ in os.listdir(galdir)])
        for tile in tiles:
            for dim in dims:
                rimg = os.path.join(galdir, "{}_{}_I_{}_swp.fits".format(
                    galaxy, tile, dim))
                gimg = os.path.join(galdir, "{}_{}_R_{}_swp.fits".format(
                    galaxy, tile, dim))
                bimg = os.path.join(galdir, "{}_{}_G_{}_swp.fits".format(
                    galaxy, tile, dim))
                if not all(os.path.exists(img) for img in [rimg, gimg, bimg]):
                    continue
                outimg = os.path.join(outdir, "{}_{}_{}.png".format(galaxy,
                                                                   tile, dim))
                make_stamp(rimg, gimg, bimg, outimg)

if __name__ == "__main__":
    make_stamps_FCC()
    make_stamps_FDS_LSB()
