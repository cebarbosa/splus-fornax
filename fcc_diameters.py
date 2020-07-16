# -*- coding: utf-8 -*-
"""

Created on 07/07/2020

Author : Carlos Eduardo Barbosa

Determines the sizee of the galaxies in the FCC to be used in the stamps

"""

from __future__ import print_function, division

import os

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack, hstack
from astroquery.ned import Ned

import context

def fcc_diameters():
    catalog = Table.read(os.path.join(context.tables_dir, "FCC_likely.fits"))
    results = []
    for gal in catalog:
        galtab = Table()
        name = "FCC {}".format(gal["FCC"])
        size_cat = gal["Reff"]
        result_table = Ned.query_object(name)
        galtab["Object"] = ["FCC{:05d}".format(gal["FCC"])]
        galtab["RA"] = result_table["RA"]
        galtab["DEC"] = result_table["DEC"]
        ndiameters = result_table["Diameter Points"]
        if ndiameters > 0:
            diam = Ned.get_table(name, table='diameters' )
            units = [_.decode() for _ in diam["Major Axis Unit"].data]
            idx = [i for i, un in enumerate(units) if un in
                   ["arcmin", "arcsec", "degree"]]
            if len(idx) > 0:
                diam = diam[idx]
                sizes = [d["Major Axis"] * u.Unit(d["Major Axis Unit"]) for d in
                         diam]
                sizes = np.array([s.to("arcsec").value for s in sizes])
                size_ned = np.nanmax(sizes)
            else:
                size_ned = 0
        else:
            size_ned = 0
        size = np.max([size_cat, size_ned])
        galtab["Major Axis"] = [size] * u.arcsec
        results.append(galtab)
    results = vstack(results)
    results.write(os.path.join(context.tables_dir, "FCC_sizes.fits"),
                  overwrite=True)

if __name__ == "__main__":
    fcc_diameters()