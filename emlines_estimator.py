# -*- coding: utf-8 -*-
""" 

Created on 08/04/18

Author : Carlos Eduardo Barbosa

Methods to extract H-alpha in the SPLUS filters according to methods presented
in Villela-Rojo et al. 2015 (VR+15).

"""
from __future__ import division, print_function

import os

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

class EmLine3Filters():
    def __init__(self, wave, bands, wtrans, trans):
        self.wave = wave
        self.bands = bands
        self.wtrans = wtrans
        self.trans = trans
        self.wpiv = self.calc_wpiv()
        self.alphax = self.calc_alphax()
        self.deltax = self.calc_deltax()
        self.betax = self.calc_betax()
        self.a = (self.alphax[0] - self.alphax[2]) / \
                 (self.alphax[1] - self.alphax[2])

    def calc_wpiv(self):
        """ Numerical integration of equation (4) in VR+2015 for SPLUS
        system. """
        wpiv = []
        for band in self.bands:
            wave = self.wtrans[band]
            trans = self.trans[band]
            curve = interp1d(wave, trans, kind="linear", fill_value=0.,
                             bounds_error=False)
            term1 = np.trapz(curve(wave) * wave, wave)
            term2 = np.trapz(curve(wave) / wave, wave)
            wpiv.append(np.sqrt(term1 / term2))
        return u.Quantity(wpiv)

    def calc_alphax(self):
        """ Numerical integration of equation (4) in VR+2015 for SPLUS
        system. """
        alphax = []
        for band in self.bands:
            wave = self.wtrans[band]
            trans = self.trans[band]
            curve = interp1d(wave, trans, kind="linear", fill_value=0.,
                             bounds_error=False)
            term1 = np.trapz(trans * wave * wave, wave) / curve(self.wave) / \
                    self.wave
            term2 = np.trapz(trans * wave, wave) / curve(self.wave) / self.wave
            alphax.append(term1 / term2)
        return u.Quantity(alphax)

    def calc_deltax(self):
        """ Numerical integration of equation (2) in VR+2015. """
        deltax = []
        for band in self.bands:
            wave = self.wtrans[band]
            trans = self.trans[band]
            curve = interp1d(wave, trans, kind="linear", fill_value=0.,
                             bounds_error=False)
            val = np.trapz(trans * wave, wave) / curve(self.wave) / self.wave
            deltax.append(val)
        return u.Quantity(deltax)

    def calc_betax(self):
        """ Determination of second term of equation (4) using deltax. """
        betax = []
        for band in self.bands:
            wave = self.wtrans[band]
            trans = self.trans[band]
            curve = interp1d(wave, trans, kind="linear", fill_value=0.,
                             bounds_error=False)
            val = curve(self.wave) * self.wave / np.trapz(trans * wave, wave)
            betax.append(val)
        return u.Quantity(betax)

    def flux_3F(self, flux):
        """Three filter method from VR+2015 (equation (3)). """
        numen = (flux[0] - flux[2]) - self.a * (flux[1] - flux[2])
        denom = self.betax[0] - self.a * self.betax[1]
        return numen / denom

    def fluxerr_3F(self, fluxerr):
        numen = np.sqrt(fluxerr[0]**2 + (self.a * fluxerr[1])**2 +
                        ((self.a - 1) * fluxerr[2])**2)
        denom = np.abs(self.betax[0] - self.a * self.betax[1])
        return numen / denom

    def EW(self, mag):
        """ Calculates the equivalent width using equation (13) of VR+2015. """
        Q = np.power(10, 0.4 * (mag[0] - mag[1]))
        eps = self.deltax[1] / self.deltax[0]
        return self.deltax[1] * (Q - 1)**2 / (1 - Q * eps)
