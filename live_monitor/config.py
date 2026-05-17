#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:23:31 2026

@author: acanamero-ext
"""

import astropy.units as u
from astropy.coordinates import EarthLocation
import os

# --- TELESCOPE CONFIGURATION ---
# Location of the IAC80 (Teide Observatory, Tenerife)
LOCATION = EarthLocation(lat=28.299667 * u.deg, lon=-16.511027 * u.deg, height=2381.25 * u.m)

# Pixel scale for guiding and error calculations
ARCSEC_PIXEL = 0.336

# Base directory for raw FITS files
BASE_DIR = "/hefestoe/data_raw"

# --- FILE & DIRECTORY NAMING CONVENTIONS ---
FITS_GLOB_PATTERN = 'O*.fits'

def get_night_dir_name(obs_date):
    """Formats the directory name for a given observation date. (Default: yyMmmdd)"""
    months = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    return f"{obs_date.strftime('%y')}{months[obs_date.month]}{obs_date.strftime('%d')}"

def get_fits_index(filepath):
    """Extracts the unique identifier from the filename to use as the CSV index."""
    # Default behavior: O20260220_1217.fits -> 20260220_1217
    return os.path.basename(filepath)[-18:-5]

def get_fits_sort_number(filepath):
    """Extracts the numeric part of the filename for chronological sorting."""
    # Default behavior: grabs the '1217' from 'O20260220_1217.fits'
    filename = os.path.basename(filepath)
    try:
        return int(filename.split('_')[-1].replace('.fits', ''))
    except (IndexError, ValueError):
        return 0

# --- CAMERA CONFIGURATION ---
def get_camera_settings(header):
    """Parses the FITS header to determine camera read speed and saturation limits."""
    mode_str = (header.get('READOUTM', '')).upper()
    settings = {"mode": "Unknown", "speed": None, "saturation": 40000}
    
    if "100KHZ" in mode_str: 
        settings.update({"mode": "Mode 3", "speed": 100, "saturation": 60000})
    elif "855KHZ" in mode_str: 
        settings.update({"mode": "Mode 4", "speed": 855, "saturation": 12000})
    elif "709KHZ" in mode_str: 
        settings.update({"mode": "Mode 2", "speed": 709, "saturation": 22000})
    elif "344KHZ" in mode_str:
        if "CCD ATTN0" in mode_str: 
            settings.update({"mode": "Mode 0", "speed": 344, "saturation": 56000})
        else: 
            settings.update({"mode": "Mode 1", "speed": 344, "saturation": 40000})
            
    return settings

# --- PLOT CONFIGURATION ---
FILTER_COLORS = {
    # Johnson-Cousins
    'U': 'purple', 'B': 'blue', 'V': 'green', 'R': 'red', 'I': 'darkred',
    # SDSS
    'SDSSu': '#9c27b0', 'SDSSg': '#4caf50', 'SDSSr': '#f44336', 'SDSSi': '#b71c1c', 'SDSSz': '#4a148c',
    # Failsafe
    'UNKNOWN_FILTER': 'black'
}