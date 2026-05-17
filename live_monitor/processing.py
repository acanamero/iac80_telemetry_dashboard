#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:26:56 2026

@author: acanamero-ext
"""

import os
import time
import warnings
import numpy as np
from datetime import datetime
from scipy.spatial import cKDTree

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyUserWarning
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry

import state
from config import LOCATION, ARCSEC_PIXEL, FILTER_COLORS, get_camera_settings, get_fits_index
from models import Moffat2Dell


def process_single_fits(fits_file):
    init = time.perf_counter()
    fwhm_min = 2.0
    index_csv = get_fits_index(fits_file)
    
    try:
        file_data = fits.open(fits_file)
        image_data = file_data[0].data.astype(float)
        header = file_data[0].header
        
        fits_type = str(header.get('IMAGETYP', 'UNKNOWN')).strip().upper()
        if fits_type != 'OBJECT':
            print(f"\n--- SKIPPED: {os.path.basename(fits_file)} (IMAGETYP is '{fits_type}') ---")
            return
        
        obs_time = header.get('DATE-OBS', 'Unknown_Time')
        filter_name = header.get('INSFILTE', 'Unknown_Filter')
        settings = get_camera_settings(header)
        state.latest_observer = header.get('OBSERVER', 'Astronomer')
        
        print(f"\n--- NEW FILE DETECTED: {os.path.basename(fits_file)} ---")
        saturation_limit = 0.9 * settings['saturation']
        
        ny, nx = image_data.shape
        y, x = np.indices((ny, nx))
        center_x, center_y = nx // 2, ny // 2
        vignette_mask = np.hypot(x - center_x, y - center_y) > 1450
    
        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0, mask=vignette_mask)
        data = image_data - median
        
        daofind = DAOStarFinder(fwhm=3, threshold=5.0 * std, peakmax=saturation_limit) 
        psf_m = Moffat2Dell(flux=1.0, x_0=0, y_0=0, gammax=2.0, gammay=2.5, phi=0.1, alpha=1.5)
        for param in ['gammax', 'gammay', 'phi', 'alpha']:
            getattr(psf_m, param).fixed = False
        psf_m.gammax.bounds, psf_m.gammay.bounds, psf_m.alpha.bounds = (0.5, 10.0), (0.5, 10.0), (1.5, 5.0)
        
        psfphot = PSFPhotometry(psf_m, (25, 25), finder=daofind, aperture_radius=4)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            phot = psfphot(data, mask=vignette_mask)

        alpha_fit = phot['alpha_fit']
        fwhm_x = 2.0 * np.abs(phot['gammax_fit']) * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
        fwhm_y = 2.0 * np.abs(phot['gammay_fit']) * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
        phot['fwhm_x'], phot['fwhm_y'], phot['fwhm'] = fwhm_x, fwhm_y, np.sqrt(fwhm_x * fwhm_y)
            
        mask_valid = ((phot['flags'] == 0) & (phot['flux_fit'] > 0) & (phot["fwhm_x"] > fwhm_min) & (phot["fwhm_y"] > fwhm_min))
        phot = phot[mask_valid]

        if len(phot) > 0:
            print(f">>> Photometry Done. Sources: {len(phot)}")
            filtered_fwhm = np.percentile(phot["fwhm"], [5, 95])
            phot = phot[(phot["fwhm"] >= filtered_fwhm[0]) & (phot["fwhm"] <= filtered_fwhm[1])]
            
            if len(phot) == 0: return
            
            if len(phot) >= 2:
                perc_x = np.percentile(phot["fwhm_x"], 75) - np.percentile(phot["fwhm_x"], 25)
                perc_y = np.percentile(phot["fwhm_y"], 75) - np.percentile(phot["fwhm_y"], 25)
            else:
                perc_x, perc_y = 0.0, 0.0
            
            try: obs_dt = datetime.fromisoformat(obs_time)
            except ValueError: obs_dt = datetime.now()
                
            header_ra = header.get('RA', header.get('OBJRA', 'UNKNOWN'))
            header_dec = header.get('DEC', header.get('OBJDEC', 'UNKNOWN'))
            
            alt_val, az_val = None, None
            if header_ra != 'UNKNOWN' and header_dec != 'UNKNOWN':
                try:
                    coord = SkyCoord(header_ra, header_dec, unit=(u.hourangle, u.deg))
                    altaz = coord.transform_to(AltAz(obstime=Time(obs_dt), location=LOCATION))
                    alt_val, az_val = altaz.alt.degree, altaz.az.degree
                except Exception as e: print(f">>> Coord error: {e}")
            
            # --- Auto-Guiding Logic ---
            state.guiding_error_x = None
            state.guiding_error_y = None
            is_new_base = False
            
            if header_ra != 'UNKNOWN' and header_dec != 'UNKNOWN':
                current_coord = SkyCoord(header_ra, header_dec, unit=(u.hourangle, u.deg))

                if state.base_coord is None:
                    state.base_coord, state.base_phot = current_coord, phot.copy()
                    is_new_base = True
                    state.guiding_error_x, state.guiding_error_y = 0.0, 0.0
                else:
                    if current_coord.separation(state.base_coord) < 3 * u.arcmin:
                        base_xy = np.vstack([state.base_phot['x_fit'], state.base_phot['y_fit']]).T
                        curr_xy = np.vstack([phot['x_fit'], phot['y_fit']]).T
                        dist, idx = cKDTree(base_xy).query(curr_xy, k=1)
                        good = dist < 5

                        if np.sum(good) >= 5:
                            dx = curr_xy[good, 0] - base_xy[idx[good], 0]
                            dy = curr_xy[good, 1] - base_xy[idx[good], 1]
                            mask = (np.abs(dx - np.median(dx)) < 2) & (np.abs(dy - np.median(dy)) < 2)
                            dx, dy = dx[mask], dy[mask]
                            
                            if len(dx) > 0 and len(dy) > 0:
                                state.guiding_error_x, state.guiding_error_y = np.median(dx) * ARCSEC_PIXEL, np.median(dy) * ARCSEC_PIXEL
                    else:
                        state.base_coord, state.base_phot = current_coord, phot.copy()
                        is_new_base = True
                        state.guiding_error_x, state.guiding_error_y = 0.0, 0.0
                
            with state.data_lock:
                state.history_alt.append(alt_val)
                state.history_az.append(az_val)
                state.history_times.append(obs_dt.isoformat())
                state.history_indexes.append(index_csv)
                state.history_fwhm_x.append(np.median(phot['fwhm_x']))
                state.history_err_x.append(perc_x)
                state.history_fwhm_y.append(np.median(phot['fwhm_y']))
                state.history_err_y.append(perc_y)
                state.history_colors.append(FILTER_COLORS.get(filter_name.upper(), 'black'))
                state.history_airmass.append(header.get('AIRMASS', None))
                state.history_telfocus.append(header.get('TELFOCUS', None))
                state.history_humidity.append(header.get('HUMIDITY', None))
                state.history_temperature.append(header.get('TEMP', None))
                state.history_sources.append(len(phot))
                state.history_filters.append(filter_name)
                state.history_ra.append(header_ra)
                state.history_dec.append(header_dec)
                state.history_gamma_x.append(np.median(phot['gammax_fit']))
                state.history_gamma_y.append(np.median(phot['gammay_fit']))
                state.history_alpha.append(np.median(phot['alpha_fit']))
                state.history_phi.append(np.median(phot['phi_fit']))
                state.history_dx.append(state.guiding_error_x)
                state.history_dy.append(state.guiding_error_y)
                if is_new_base: state.history_base_changes.append(obs_dt.isoformat())

        print(f">>> Time taken: {(time.perf_counter() - init):.2f} seconds\n")
    except Exception as e: print(f"Error processing {fits_file}: {e}")
    finally: file_data.close()