#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:25:30 2026

@author: acanamero-ext
"""

import numpy as np
from astropy.modeling.core import Fittable2DModel
from astropy.modeling.parameters import Parameter
from astropy.units import UnitsError

class Moffat2Dell(Fittable2DModel):
    """Two dimensional Moffat elliptical model."""
    flux = Parameter(default=1, description="Scaling factor (peak value)")
    x_0 = Parameter(default=0, description="X position of max")
    y_0 = Parameter(default=0, description="Y position of max")
    gammax = Parameter(default=1, description="Core width (x-axis)")
    gammay = Parameter(default=1, description="Core width (y-axis)")
    phi = Parameter(default=0, description="Azimuthal angle of largest FWHM")
    alpha = Parameter(default=1, description="Power index")

    @property
    def fwhmx(self):
        return 2.0 * np.abs(self.gammax) * np.sqrt(2.0 ** (1.0 / self.alpha) - 1.0)

    @property
    def fwhmy(self):
        return 2.0 * np.abs(self.gammay) * np.sqrt(2.0 ** (1.0 / self.alpha) - 1.0)

    @staticmethod
    def evaluate(x, y, flux, x_0, y_0, gammax, gammay, phi, alpha):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        A = (cos_phi/gammax)**2 + (sin_phi/gammay)**2
        B = (sin_phi/gammax)**2 + (cos_phi/gammay)**2
        C = 2*cos_phi*sin_phi*(1/gammax**2 - 1/gammay**2)
        rr_gg = A*(x-x_0)**2 + B*(y-y_0)**2 + C*(x-x_0)*(y-y_0)
        return flux * (1 + rr_gg) ** (-alpha)

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, gammax, gammay, phi, alpha):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        A = (cos_phi/gammax)**2 + (sin_phi/gammay)**2
        B = (sin_phi/gammax)**2 + (cos_phi/gammay)**2
        C = 2*cos_phi*sin_phi*(1/gammax**2 - 1/gammay**2)
        rr_gg = A*(x-x_0)**2 + B*(y-y_0)**2 + C*(x-x_0)*(y-y_0)
        d_amp = (1 + rr_gg) ** (-alpha)
        
        d_flux = d_amp
        d_alpha = -flux * d_amp * np.log(1 + rr_gg)
        d_x_0 =  flux * alpha * d_amp / (1 + rr_gg) * (2*A*(x - x_0) + C*(y - y_0))
        d_y_0 =  flux * alpha * d_amp / (1 + rr_gg) * (2*B*(y - y_0) + C*(x - x_0))
        d_gammax = flux * alpha * d_amp / (1 + rr_gg) * 2 / gammax**3 * \
            ((x-x_0)**2*cos_phi**2 + (y-y_0)**2*sin_phi**2 + 2*(x-x_0)*(y-y_0)*sin_phi*cos_phi)
        d_gammay = flux * alpha * d_amp / (1 + rr_gg) * 2 / gammay**3 * \
            ((x-x_0)**2*sin_phi**2 + (y-y_0)**2*cos_phi**2 - 2*(x-x_0)*(y-y_0)*sin_phi*cos_phi)
        d_phi = -flux * alpha * d_amp / (1 + rr_gg) * 2 * \
            ((x-x_0)**2*cos_phi*sin_phi*(-1/gammax**2+1/gammay**2) + \
            (y-y_0)**2*cos_phi*sin_phi*(1/gammax**2-1/gammay**2) +\
            (x-x_0)*(y-y_0)*(cos_phi**2-sin_phi**2)*(1/gammax**2-1/gammay**2))
        return [d_flux, d_x_0, d_y_0, d_gammax, d_gammay, d_phi, d_alpha]