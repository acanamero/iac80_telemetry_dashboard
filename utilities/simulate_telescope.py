#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:09:15 2026

@author: acanamero-ext
"""

import os 
import time
import shutil
from glob import glob

# Where your data currently lives
SOURCE_DIR = '/home/acanamero-ext/practicas/data/26Feb20_Roi/'
# The empty folder where the "telescope" will drop new files
INCOMING_DIR = '/home/acanamero-ext/practicas/data/incoming_fits/'

os.makedirs(INCOMING_DIR, exist_ok=True)

fits_files = sorted(glob(os.path.join(SOURCE_DIR, '*.fits')))

print(f"Starting telescope simulation. Dropping files into {INCOMING_DIR}")

for fits_file in fits_files:
    filename = os.path.basename(fits_file)
    destination = os.path.join(INCOMING_DIR, filename)
    
    print(f"Telescope: Saving new image -> {filename}")
    shutil.copy(fits_file, destination)
    
    # Wait 10 seconds before generating the next "observation"
    time.sleep(15) 
    # Remove the file from the incoming directory when the wait finishes
    try:
        if os.path.exists(destination):
            os.remove(destination)
            print(f"Telescope: Cleaned up -> {filename}")
    except OSError as e:
        print(f"Telescope: Warning - could not delete {filename}: {e}")

print("Observation run complete.")