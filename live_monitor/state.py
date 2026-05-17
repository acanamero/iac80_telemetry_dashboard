#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:25:01 2026

@author: acanamero-ext
"""

import queue
import threading

# --- THREAD SAFETY ---
data_lock = threading.Lock()
fits_queue = queue.Queue()

# --- GUIDING STATE ---
base_coord = None
base_phot = None
guiding_error_x = None
guiding_error_y = None
latest_observer = "Observer"

# --- TELEMETRY HISTORY ---
history_times = []
history_indexes = []
history_fwhm_x = []
history_err_x = []
history_err_y = []
history_fwhm_y = []
history_colors = []
history_alt = []
history_az = []
history_airmass = []
history_telfocus = []
history_humidity = []
history_temperature = []
history_sources = []
history_filters = []
history_ra = []
history_dec = []
history_gamma_x = []
history_gamma_y = []
history_alpha = []
history_phi = []
history_dx = []
history_dy = []
history_base_changes = []