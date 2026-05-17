#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:18:42 2026

@author: acanamero-ext
"""

import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg') # Needed to use this so that no warning appears on the terminal
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyUserWarning
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry
from astropy.modeling.functional_models import Moffat2Dell
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Your original helper function ---
def get_camera_settings(header):
    mode_str = (header.get('READOUTM', '')).upper()
    settings = {"mode": "Unknown", "speed": None, "saturation": 40000, "recommended_flat": None}
    if "100KHZ" in mode_str:
        settings.update({"mode": "Mode 3", "speed": 100, "saturation": 60000, "recommended_flat": 42000})
    elif "855KHZ" in mode_str:
        settings.update({"mode": "Mode 4", "speed": 855, "saturation": 12000, "recommended_flat": 8000})
    elif "709KHZ" in mode_str:
        settings.update({"mode": "Mode 2", "speed": 709, "saturation": 22000, "recommended_flat": 16000})
    elif "344KHZ" in mode_str:
        if "CCD ATTN0" in mode_str:
            settings.update({"mode": "Mode 0", "speed": 344, "saturation": 56000, "recommended_flat": 39000})
        else:
            settings.update({"mode": "Mode 1", "speed": 344, "saturation": 40000, "recommended_flat": 28000})
    return settings


# --- The core processing function for a SINGLE image ---
def process_single_fits(fits_file):
    init = time.perf_counter()
    fwhm_min = 2.0
    base_results_dir = "results/outcome_results"
    os.makedirs(base_results_dir, exist_ok=True)
    
    try:
        file_data = fits.open(fits_file)
        image_data = file_data[0].data.astype(float)
        header = file_data[0].header
        
        obs_time = header.get('DATE-OBS', 'Unknown_Time')
        filter_name = header.get('INSFILTE', 'Unknown_Filter')
        settings = get_camera_settings(header)
        name = header.get("OBSERVER", "UNK")[:3]
        
        print(f"\n--- NEW FILE DETECTED: {os.path.basename(fits_file)} ---")
        print(f"Time: {obs_time} | Filter: {filter_name} | {settings['mode']}")
        
        saturation_limit = 0.9 * settings['saturation']
        safe_time = str(obs_time).replace(":", "-").replace("T", "_")
        save_dir = os.path.join(base_results_dir, f"{name}{safe_time}_{filter_name}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Vignetting mask
        ny, nx = image_data.shape
        y, x = np.indices((ny, nx))
        center_x, center_y = nx // 2, ny // 2
        good_radius = 1450  
        r = np.hypot(x - center_x, y - center_y)
        vignette_mask = r > good_radius
    
        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0, mask=vignette_mask)
        data = image_data - median
        
        # Source Detection & PSF Modeling
        threshold = 5.0 * std
        daofind = DAOStarFinder(fwhm=3, threshold=threshold, peakmax=saturation_limit) 
        
        psf_m = Moffat2Dell(flux=1.0, x_0=0, y_0=0, gammax=2.0, gammay=2.5, phi=0.1, alpha=1.5)
        psf_m.gammax.fixed = False
        psf_m.gammay.fixed = False
        psf_m.phi.fixed = False
        psf_m.alpha.fixed = False
        psf_m.gammax.bounds = (0.5, 10.0) 
        psf_m.gammay.bounds = (0.5, 10.0)
        psf_m.alpha.bounds = (1.5, 5.0)   
        
        psfphot = PSFPhotometry(psf_m, (25, 25), finder=daofind, aperture_radius=4)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            phot = psfphot(data, mask=vignette_mask)

        # Math / FWHM Extraction
        alpha_fit = phot['alpha_fit']
        fwhm_x = 2.0 * np.abs(phot['gammax_fit']) * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
        fwhm_y = 2.0 * np.abs(phot['gammay_fit']) * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
        a = np.maximum(fwhm_x, fwhm_y)
        b = np.minimum(fwhm_x, fwhm_y)
        
        phot['fwhm_x'] = fwhm_x
        phot['fwhm_y'] = fwhm_y
        phot['fwhm'] = np.sqrt(fwhm_x * fwhm_y)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            phot["ellipticity"] = 1 - (b/a)
            
        # Filtering
        mask_valid = ((phot['flags'] == 0) & (phot['flux_fit'] > 0) &  
                      (phot["fwhm_x"] > fwhm_min) & (phot["fwhm_y"] > fwhm_min))
        phot = phot[mask_valid]

        filtered_fwhm = np.percentile(phot["fwhm"], [5, 95])
        mask_outliers = (phot["fwhm"] >= filtered_fwhm[0]) & (phot["fwhm"] <= filtered_fwhm[1])
        phot = phot[mask_outliers]
        
        if len(phot) > 0:
            mean_fwhm = np.mean(phot['fwhm'])
            median_fwhm = np.median(phot['fwhm'])
            std_fwhm = np.std(phot['fwhm'])
            
            # THE TERMINAL OUTPUT
            print(f">>> Photometry Done. Sources: {len(phot)}")
            print(f">>> FWHM Mean: {mean_fwhm:.2f} px | FWHM Median: {median_fwhm:.2f} px | FWHM std: {std_fwhm}")
            
            # Save plots 
            theta_deg = np.degrees(phot['phi_fit'])
            theta_bounded = (theta_deg + 90) % 180 - 90
            
            fig3, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15, 5))
            fig3.suptitle(f"Fitted Parameters - {obs_time}", fontsize=14, fontweight='bold')
            
            ax3.set_title("Overall FWHM"); ax3.set_xlabel("FWHM [px]")
            ax3.hist(phot['fwhm'], bins="auto", color='skyblue', edgecolor='black')
            
            ax4.set_title("Phi Angle"); ax4.set_xlabel("Phi [deg]")
            ax4.hist(theta_bounded, bins="auto", color='lightgreen', edgecolor='black')
            
            ax5.set_title("Ellipticity"); ax5.set_xlabel("Ellipticity e")
            ax5.hist(phot['ellipticity'], bins="auto", color='salmon', edgecolor='black')
    
            fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            plot_path = os.path.join(save_dir, "fwhm_stats.png")
            plt.savefig(plot_path)
            plt.close(fig3) # IMPORTANT: Close the figure to free memory!
            print(f">>> Plots saved to {plot_path}")
        else:
            print(">>> Error: No valid sources found.")

        print(f">>> Time taken: {(time.perf_counter() - init):.2f} seconds\n")
        
    except Exception as e:
        print(f"Error processing {fits_file}: {e}")
    finally:
        file_data.close()


# --- Watchdog Event Handler ---
class FitsHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Trigger only when a new .fits file is fully created
        if not event.is_directory and event.src_path.endswith('.fits'):
            # Sleep briefly to ensure the OS has finished writing the file
            time.sleep(0.5) 
            process_single_fits(event.src_path)

if __name__ == "__main__":
    # Directory to monitor
    INCOMING_DIR = '/home/acanamero-ext/practicas/data/incoming_fits/'
    os.makedirs(INCOMING_DIR, exist_ok=True)
    
    event_handler = FitsHandler()
    observer = Observer()
    observer.schedule(event_handler, path=INCOMING_DIR, recursive=False)
    
    print(f"Listening for new FITS files in {INCOMING_DIR}...\nPress Ctrl+C to stop.")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()