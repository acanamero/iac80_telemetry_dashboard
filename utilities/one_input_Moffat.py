# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:33:01 2026

@author: aleja
"""



from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.io import fits
from astropy.modeling.functional_models import Moffat2Dell
import os                            
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry
from photutils.datasets import make_model_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import SymLogNorm

# ---------------- Functions ---------------------
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

# ---------------- Data Retrieval ---------------------
#ruta_dataset = f'C:{os.sep}Users{os.sep}aleja{os.sep}Practicas{os.sep}data{os.sep}Comparacion{os.sep}O20260222_1209.fits'  
ruta_dataset = f'{os.sep}home{os.sep}acanamero-ext{os.sep}practicas{os.sep}data{os.sep}Comparacion{os.sep}O20260222_1209.fits'
fits_files = [ruta_dataset]                                            
    
fits_data = []
timestamps = []
filters = [] 
airmass = []
el = []    
fwhm_min = 2.0

for fits_file in fits_files:
        
        file_data = fits.open(fits_file)
        image_data = file_data[0].data.astype(float)
        fits_data.append(image_data)
        
        header = file_data[0].header
        obs_time = header.get('DATE-OBS', 'Unknown_Time')
        timestamps.append(obs_time)
        
        filter_name = header.get('INSFILTE', 'Unknown_Filter')
        filters.append(filter_name)
        
        print(f"Loaded {fits_file} - Filter: {filter_name}")
        
        settings = get_camera_settings(header)        
        print(f"Detected: {settings['mode']} ({settings['speed']} kHz)")
        
        name = header.get("OBSERVER", "UNK")[:3]
        airmass.append(header.get("AIRMASS", None))
        
saturation_limit = 0.9 * settings['saturation']

base_results_dir = "results"
os.makedirs(base_results_dir, exist_ok=True)
        
for i, data in enumerate(fits_data):
        
        current_time = timestamps[i]
        current_filter = filters[i]
        
        safe_time = str(current_time).replace(":", "-").replace("T", "_")
        
        save_dir = os.path.join(base_results_dir, f"{name}{safe_time}_{current_filter}")
        os.makedirs(save_dir, exist_ok=True)
        
        meta_title = f"Date/Time: {current_time} | Filter: {current_filter}"

        # ----------------- Define circular mask for vignetting -----------
        ny, nx = fits_data[i].shape
        y, x = np.indices((ny, nx))
    
        center_x = nx // 2
        center_y = ny // 2
        good_radius = 1450  
    
        r = np.hypot(x - center_x, y - center_y)
        vignette_mask = r > good_radius
    
        mean, median, std = sigma_clipped_stats(fits_data[i], sigma=3.0, mask=vignette_mask)
        print(f"\nProcessing image {i+1}/{len(fits_data)}: {current_time}")
        print(f"Global Background - Median: {median:.2f}, Std: {std:.2f}")
    
        data = fits_data[i] - median
    
        # --------------- Initial Source Detection ----------------------
        threshold = 5.0 * std
        daofind = DAOStarFinder(fwhm=3, threshold=threshold, peakmax=saturation_limit) 
    
        # ------------------------ PSF Modeling ----------------------
        # Initialize custom Moffat model
        psf_m = Moffat2Dell(flux=1.0, x_0=0, y_0=0, gammax=2.0, gammay=2.5, phi=0.1, alpha=1.5)
        
        # Unfix all structural parameters so they can be fitted
        psf_m.gammax.fixed = False
        psf_m.gammay.fixed = False
        psf_m.phi.fixed = False
        psf_m.alpha.fixed = False
        # ADD BOUNDS HERE to keep the fitter from going crazy
        psf_m.gammax.bounds = (0.5, 10.0) # Prevent negative or massive widths
        psf_m.gammay.bounds = (0.5, 10.0)
        psf_m.alpha.bounds = (1.5, 5.0)   # Force alpha to be strictly > 1 for FWHM math
        
        box_size = 19
        fit_shape = (box_size, box_size)
        
        psfphot = PSFPhotometry(psf_m, fit_shape, finder=daofind, aperture_radius=4)
        phot = psfphot(data, mask=vignette_mask)

        # ---------------- FWHM & Ellipticity Extraction --------------
        # Since photutils fits our parameters, the table outputs <param>_fit
        # We manually calculate FWHM from gammax, gammay, and alpha using the Moffat formula
        alpha_fit = phot['alpha_fit']
        
        
        # Initialize with zeros to safely handle the math
        fwhm_x = np.zeros(len(phot))
        fwhm_y = np.zeros(len(phot))
        
        # Compute where valid
        fwhm_x = 2.0 * np.abs(phot['gammax_fit']) * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
        fwhm_y = 2.0 * np.abs(phot['gammay_fit']) * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
        
        a = np.maximum(fwhm_x, fwhm_y)
        b = np.minimum(fwhm_x, fwhm_y)
        
        phot['fwhm_x'] = fwhm_x
        phot['fwhm_y'] = fwhm_y
        phot['fwhm'] = np.sqrt(fwhm_x * fwhm_y)
        
        # Suppress divide-by-zero warnings for failing sources during ellipticity calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            phot["ellipticity"] = 1 - (b/a)
        
        # -------------------------- Filtering process ----------------------
        # Filter out failures, invalid alpha values, and bad FWHMs
        mask_valid = ((phot['flags'] == 0) & (phot['flux_fit'] > 0 ) &  
                      (phot["fwhm_x"] > fwhm_min) & (phot["fwhm_y"] > fwhm_min))
        phot = phot[mask_valid]

        # Filter out outliers based on the calculated FWHM
        filtered_fwhm = np.percentile(phot["fwhm"], [5, 95])
        mask_outliers = (phot["fwhm"] >= filtered_fwhm[0]) & (phot["fwhm"] <= filtered_fwhm[1])
        phot = phot[mask_outliers]

        print(f"Found {len(phot)} sources to perform PSF photometry with")
    
        if phot is None or len(phot) == 0:
            print("Error: No PSF photometry data to plot for this image.")
        else:
            
            # ----------------- Plot 1 Setup -----------------
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.suptitle(f"2D Map - {meta_title}", fontsize=14, fontweight='bold')
            
            scatter = ax.scatter(phot['x_fit'], phot['y_fit'], c=phot['fwhm'], cmap='viridis', 
                                 s=20, edgecolor='none', alpha=0.8)
    
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Overall FWHM (pixels)')
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            ax.set_title('Object Location vs. Overall FWHM')
            ax.set_aspect('equal') 
            plt.xlim(0, nx)
            plt.ylim(0, ny)
            ax.grid(True, linestyle=':', color='gray', alpha=0.5)
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            fig.savefig(os.path.join(save_dir, "01_2D_Map_Moffat.png"))
    
            # ----------------- Plot 2 Setup -----------------
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig2.suptitle(f"Data vs Model - {meta_title}", fontsize=14, fontweight='bold')
            
            norm = SymLogNorm(linthresh=1.0, vmin=-5, vmax=10000)
            
            # Subplot 1: Real Data
            im1 = ax1.imshow(data, origin='lower', cmap='gray', norm=norm)
            ax1.set_title('Real Data (Background Subtracted)')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Counts')
            
            data_limit_circle = Circle((center_x, center_y), good_radius, edgecolor='red', 
                                       facecolor='none', linestyle='--', linewidth=2, 
                                       label=f'Data Limit (R={good_radius}px)')
            ax1.add_patch(data_limit_circle)
            ax1.legend(loc='upper right', frameon=True, shadow=True)
            
            # Subplot 2: Fitted PSF Model
            # Prepare data to generate models
            clean_phot = phot.copy()

            # Rename columns back to exactly match our Moffat parameters
            clean_phot.rename_column('x_fit', 'x_0')
            clean_phot.rename_column('y_fit', 'y_0')
            clean_phot.rename_column('flux_fit', 'flux')
            clean_phot.rename_column('gammax_fit', 'gammax')
            clean_phot.rename_column('gammay_fit', 'gammay')
            clean_phot.rename_column('phi_fit', 'phi')
            clean_phot.rename_column('alpha_fit', 'alpha')
            
            model_image = make_model_image(data.shape, psf_m, clean_phot, model_shape=fit_shape)
            
            im2 = ax2.imshow(model_image, origin='lower', cmap='gray', norm=norm)
            ax2.set_title('Fitted Moffat PSF Model')
            ax2.set_xlabel('X (pixels)')
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Counts')
            
            fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig2.savefig(os.path.join(save_dir, "02_Model_Comparison_Moffat.png"))
            
            # ----------------- Plot 3 Setup -----------------
            # Extract rotational parameter from our custom model
            theta_deg = np.degrees(phot['phi_fit'])
            theta_bounded = (theta_deg + 90) % 180 - 90
            
            fig3, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15, 5))
            fig3.suptitle(f"Fitted Parameters - {meta_title}", fontsize=14, fontweight='bold')
            
            ax3.set_title("Overall FWHM")
            ax3.set_ylabel("Frequency")
            ax3.set_xlabel("FWHM [px]")
            counts, fwhm_bins, patches = ax3.hist(phot['fwhm'], bins="auto", color='skyblue', edgecolor='black')
            
            num_bins = len(fwhm_bins) - 1
            
            ax4.set_title("Phi Angle")
            ax4.set_ylabel("Frequency")
            ax4.set_xlabel("Phi [deg]")
            ax4.hist(theta_bounded, bins=num_bins, color='lightgreen', edgecolor='black')
            
            ax5.set_title("Ellipticity")
            ax5.set_ylabel("Frequency")
            ax5.set_xlabel("Ellipticity e")
            ax5.hist(phot['ellipticity'], bins=num_bins, color='salmon', edgecolor='black')
    
            fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig3.savefig(os.path.join(save_dir, "03_Statistics_Moffat.png"))
            
            # ----------------- Plot 4 Setup ------------------
            fig4, (ax6, ax7) = plt.subplots(1, 2, figsize=(12, 5))
            fig4.suptitle(f"FWHM Trends - {meta_title}", fontsize=14, fontweight='bold')
            
            ax6.set_title("FWHM along x axis")
            ax6.set_ylabel("FWHM")
            ax6.set_xlabel("Detector x axis")
            ax6.plot(phot["x_fit"], phot["fwhm"], ".r")
        
            ax7.set_title("FWHM along y axis")
            ax7.set_ylabel("FWHM")
            ax7.set_xlabel("Detector y axis")
            ax7.plot(phot["y_fit"], phot["fwhm"], ".b")
            
            fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig4.savefig(os.path.join(save_dir, "04_Ellipticity_Trends_Moffat.png"))
            
            print("Overall Statistics:")
            print(f"  Mean FWHM: {np.mean(phot['fwhm']):.2f} ")
            print(f"  Median FWHM: {np.median(phot['fwhm']):.2f} ")
            print(f"  FWHM Std Dev: {np.std(phot['fwhm']):.2f}")
            
            el.append(np.mean(phot["fwhm"]))
            
            #---------------- Plot 5 Residual --------------------
            residual_image = data - model_image.astype(float)
            
            fig5, (ax_data, ax_model, ax_res) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            fig5.suptitle(f"Residual Analysis - {meta_title}", fontsize=14, fontweight='bold')
            
            norm_res = ImageNormalize(residual_image, stretch=SqrtStretch(), vmin=-10, vmax=500) 

            # Plot Original Data
            ax_data.imshow(data, origin='lower', cmap='grey', norm=norm_res)
            ax_data.set_title("Original (BG Subtracted)")
            ax_data.set_ylim(2160, 2360)
            ax_data.set_xlim(2100, 2500)
        
            # Plot Model
            ax_model.imshow(model_image, origin='lower', cmap='grey', norm=norm_res)
            ax_model.set_title("Moffat PSF Model")
            ax_model.set_ylim(2160, 2360)
            ax_model.set_xlim(2100, 2500)

            # Plot Residuals
            im_res = ax_res.imshow(residual_image, origin='lower', cmap='grey', norm=norm_res)
            ax_res.set_title("Residuals (Data - Model)")
            ax_res.set_ylim(2160, 2360)
            ax_res.set_xlim(2100, 2500)

            cbar_res = plt.colorbar(im_res, ax=ax_res, fraction=0.046, pad=0.04)
            cbar_res.set_label('Difference (Counts)')

            fig5.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig5.savefig(os.path.join(save_dir, "05_Residual_Analysis_Moffat.png"))
            
        
"""
Dos líneas, una para fwhm_x y otra fwhm_y, en un mismo plot, un punto por cada imagen
que además te ponga la hora en la que lo tomaste. 
"""
            
            

print("\nProcessing complete! All plots saved in the 'results' folder.")