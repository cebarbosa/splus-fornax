""" Combines catalogs of Analia and Maddox to get IDs. """
import os

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

import context



if __name__ == "__main__":
    fname = os.path.join(context.tables_dir,
        "All_Fornax_galaxies_literature_16_catalogs_not_duplicated_no_bckg.dat")
    table = Table.read(fname, format="ascii.commented_header")
    print(len(set(table["Ref"].data)))

