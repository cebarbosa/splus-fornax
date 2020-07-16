# -*- coding: utf-8 -*-
"""

Created on 15/07/2020

Author : Carlos Eduardo Barbosa

Uses Maddox catalog of the Fornax cluster to determine S-PLUS fields

"""
import os

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

import context

if __name__ == "__main__":
    maddox_table = os.path.join(context.tables_dir, "Table2_Maddox_Fornax.txt")
    maddox = Table.read(maddox_table, format="ascii", delimiter=";",
                        names=["ID", "RA", "Dec", "cz", "czerr",  "source",
                               "comment"], data_start=0)
    tiles_table = os.path.join(context.tables_dir, "all_tiles_final.csv")
    tiles = Table.read(tiles_table)
    tcoords = SkyCoord(tiles["RA"], tiles["DEC"],
                       unit=(u.hourangle, u.degree))
    ramin, ramax = 47, 60
    decmin, decmax = -40, -31
    galcoords = SkyCoord(maddox["RA"], maddox["Dec"], unit=u.degree)
    idx = np.where((ramin <= tcoords.ra.value) & ( tcoords.ra.value<= ramax) &
                   (decmin <= tcoords.dec.value) &
                   ( tcoords.dec.value<= decmax))[0]
    tiles = tiles[idx]
    tiles.write(os.path.join(context.tables_dir, "fornax_splus_tiles.fits"),
                overwrite=True)
    tcoords = tcoords[idx]
    fig = plt.figure(figsize=(5.5, 3.3))
    ax = plt.subplot(111, aspect="equal")
    plt.plot(maddox["RA"], maddox["Dec"], "x", ms=0.8,)
    for i, tile in enumerate(tcoords):
        ra, dec = tile.ra.value, tile.dec.value
        if ra > 180:
            ra -= 360
        size = np.sqrt(2) / 2
        x0, x1 = ra - size, ra + size
        y0, y1 = dec - size, dec + size
        ax.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], "-", lw=0.3,
                c="y", zorder=100, alpha=1)
        ax.text(x1 - 0.15, .5 * (y0 + y1) -.1, tiles["NAME"][i].split("-")[1])
    ax.invert_xaxis()
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.12, top=0.98)
    plt.savefig(os.path.join(context.home_dir, "plots/fornax-splus-tiles.png"),
                dpi=300)