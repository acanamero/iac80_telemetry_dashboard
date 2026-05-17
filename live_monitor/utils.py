#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:26:06 2026

@author: acanamero-ext
"""

import os
import csv
import numpy as np
from datetime import datetime, timedelta
from ginga.util import grc

import state
from config import FILTER_COLORS, get_night_dir_name

def get_observing_night_dir(base_dir):
    """Calculates the directory path based on the rollover threshold (8:00 AM)."""
    now = datetime.now()
    obs_date = now - timedelta(days=1) if now.hour < 8 else now
    return os.path.join(base_dir, get_night_dir_name(obs_date))

def save_csv_to_disk(target_dir):
    if len(state.history_times) == 0:
        print("No telemetry data collected. Skipping CSV generation.")
        return

    # Use os.path.basename to dynamically get the folder name 
    folder_name = os.path.basename(get_observing_night_dir(target_dir))
    filename = os.path.join(target_dir, f"{folder_name}.csv")
    print(f"\nSaving final telemetry data to {filename}...")

    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'INDEX', 'UTC', 'AIRMASS', 'FILTER', 'TELFOCUS', 'HUMIDITY', 'TEMPERATURE', 
                'RA', 'DEC', 'ALT', 'AZ', 'GAMMA_X', 'FWHM_X', 'ERR_X', 'GAMMA_Y', 'FWHM_Y', 
                'ERR_Y', 'ALPHA', 'PHI', 'NSOURCES', 'DX', 'DY'
            ])
            
            dt_objects = [datetime.fromisoformat(t) for t in state.history_times]
            sorted_indices = np.argsort(dt_objects)
            
            for i in sorted_indices:
                writer.writerow([
                    state.history_indexes[i], state.history_times[i], state.history_airmass[i],
                    state.history_filters[i], state.history_telfocus[i], state.history_humidity[i],
                    state.history_temperature[i], state.history_ra[i], state.history_dec[i],
                    state.history_alt[i], state.history_az[i], state.history_gamma_x[i], 
                    state.history_fwhm_x[i], state.history_err_x[i], state.history_gamma_y[i], 
                    state.history_fwhm_y[i], state.history_err_y[i], state.history_alpha[i], 
                    state.history_phi[i], state.history_sources[i], state.history_dx[i], state.history_dy[i]
                ])
    except Exception as e:
        print(f"Error saving CSV to disk: {e}")

def load_csv_to_history(target_dir):
    folder_name = os.path.basename(get_observing_night_dir(target_dir))
    csv_file = os.path.join(target_dir, f"{folder_name}.csv")
    if not os.path.exists(csv_file):
        return set()

    processed_indices = set()
    print(f"\n>>> Found existing session in {csv_file}. Restoring history...")

    def safe_float(val):
        return float(val) if val not in ('', 'None', None) else 0

    try:
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row['INDEX']
                processed_indices.add(idx)

                state.history_indexes.append(idx)
                state.history_times.append(row['UTC'])
                state.history_airmass.append(safe_float(row['AIRMASS']))
                state.history_filters.append(row['FILTER'])
                state.history_colors.append(FILTER_COLORS.get(row['FILTER'].upper(), 'black'))
                state.history_telfocus.append(safe_float(row['TELFOCUS']))
                state.history_humidity.append(safe_float(row['HUMIDITY']))
                state.history_temperature.append(safe_float(row['TEMPERATURE']))
                state.history_ra.append(row['RA'])
                state.history_dec.append(row['DEC'])
                state.history_alt.append(safe_float(row['ALT']))
                state.history_az.append(safe_float(row['AZ']))
                state.history_gamma_x.append(safe_float(row['GAMMA_X']))
                state.history_fwhm_x.append(safe_float(row['FWHM_X']))
                state.history_err_x.append(safe_float(row['ERR_X']))
                state.history_gamma_y.append(safe_float(row['GAMMA_Y']))
                state.history_fwhm_y.append(safe_float(row['FWHM_Y']))
                state.history_err_y.append(safe_float(row['ERR_Y']))
                state.history_alpha.append(safe_float(row['ALPHA']))
                state.history_phi.append(safe_float(row['PHI']))
                state.history_sources.append(safe_float(row['NSOURCES']))
                state.history_dx.append(safe_float(row['DX']))
                state.history_dy.append(safe_float(row['DY']))
    except Exception as e:
        print(f"Error loading CSV: {e}")
    return processed_indices

def update_ginga_viewer(fits_file_path, channel_name='Image'):
    try:
        viewer = grc.RemoteClient('127.0.0.1', 11771)
        viewer.shell().load_file(fits_file_path, channel_name)
    except ConnectionRefusedError:
        print("Warning: Ginga is not open or the RC plugin is not running.")
    except Exception as e:
        print(f"Error sending image to Ginga: {e}")