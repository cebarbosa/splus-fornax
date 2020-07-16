# -*- coding: utf-8 -*-
"""

Created on 06/07/2020

@Author: Carlos Eduardo Barbosa

"""

import os
import platform

import astropy.units as u
import matplotlib.pyplot as plt

if platform.node() in ["t80s-jype2", "t80s-jype"]:
    home_dir = "/mnt/public/kadu/fornax"
elif platform.node() in ["kadu-Inspiron-5557"]:
    home_dir = "/home/kadu/Dropbox/SPLUS/fornax"

data_dir = os.path.join(home_dir, "data")
tables_dir = os.path.join(home_dir, "tables")
plots_dir= os.path.join(home_dir, "plots")

ps = 0.55 * u.arcsec / u.pixel


bands = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I',
         'F861', 'Z']

narrow_bands = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']

broad_band = ['U', 'G', 'R', 'I', 'Z']

bands_names = {'U' : "$u$", 'F378': "$J378$", 'F395' : "$J395$",
               'F410' : "$J410$", 'F430' : "$J430$", 'G' : "$g$",
               'F515' : "$J515$", 'R' : "$r$", 'F660' : "$J660$",
               'I' : "$i$", 'F861' : "$J861$", 'Z' : "$z$"}

wave_eff = {"F378": 3773.4, "F395": 3940.8, "F410": 4095.4,
            "F430": 4292.5, "F515": 5133.5, "F660": 6614.0, "F861": 8608.1,
            "G": 4647.8, "I": 7683.8, "R": 6266.6, "U": 3536.5,
            "Z": 8679.5}

tiles = ["s29s31", "s28s32", "s27s33", "s29s31", "s29s31", "s27s34", "s26s35",
         "s29s31", "s27s35"]

# Matplotlib settings
plt.style.context("seaborn-paper")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True