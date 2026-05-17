# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:08:43 2026

@author: aleja
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:15:03 2026

@author: aleja
"""


from astropy.table import QTable  
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling.functional_models import Moffat2Dell
from glob import glob                
import os                            
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
from photutils.utils import circular_footprint
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from photutils.psf import PSFPhotometry
from photutils.psf import GaussianPSF, MoffatPSF
from photutils.datasets import make_model_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
from matplotlib.patches import Circle
from scipy.ndimage import binary_dilation
from scipy.spatial import KDTree 
from matplotlib.colors import SymLogNorm


def get_camera_settings(header):
    mode_str = (header.get('READOUTM', '')).upper()

    # Default fallback (safe values)
    settings = {
        "mode": "Unknown",
        "speed": None,
        "saturation": 40000,
        "recommended_flat": None
    }

    # --- Detect speed ---
    if "100KHZ" in mode_str:
        settings.update({
            "mode": "Mode 3",
            "speed": 100,
            "saturation": 60000,
            "recommended_flat": 42000
        })

    elif "855KHZ" in mode_str:
        settings.update({
            "mode": "Mode 4",
            "speed": 855,
            "saturation": 12000,
            "recommended_flat": 8000
        })

    elif "709KHZ" in mode_str:
        settings.update({
            "mode": "Mode 2",
            "speed": 709,
            "saturation": 22000,
            "recommended_flat": 16000
        })

    elif "344KHZ" in mode_str:
        # Need attenuation to distinguish
        if "CCD ATTN0" in mode_str:
            settings.update({
                "mode": "Mode 0",
                "speed": 344,
                "saturation": 56000,
                "recommended_flat": 39000
            })
        else:
            settings.update({
                "mode": "Mode 1",
                "speed": 344,
                "saturation": 40000,
                "recommended_flat": 28000
            })

    return settings

# ---------------- Data Retrieval ---------------------

#ruta_dataset = f'C:{os.sep}Users{os.sep}aleja{os.sep}Practicas{os.sep}data{os.sep}Comparacion{os.sep}O20260222_1209.fits'  
ruta_dataset = f'{os.sep}home{os.sep}acanamero-ext{os.sep}practicas{os.sep}data{os.sep}Comparacion{os.sep}O20260222_1209.fits'


fits_files = [ruta_dataset]                                              
    
fits_data = []
timestamps = []
    
filters = [] # Added list to store filters
airmass = []
el = []    
fwhm_min = 2.0

# --- Outside the loop (Initialize) ---
target_coma = (2170.6, 707.3) # Clear example of coma for manolo bright field
target_round = (2252, 2588)   # Clear example of nice and round for manolo bright field
search_radius = 10

for fits_file in fits_files:

    # Read image data
    file_data = fits.open(fits_file)
    image_data = file_data[0].data.astype(float)
    fits_data.append(image_data)

    # Read observation time from the header
    header = file_data[0].header
    obs_time = header.get('DATE-OBS', 'Not found')
    timestamps.append(obs_time)
    # Check the filter keyword (often 'FILTER', 'FLT', or 'INSFLT')
    filter_name = header.get('INSFILTE', 'Unknown_Filter')
    filters.append(filter_name)

    print(f"Loaded {fits_file} - Filter: {filter_name}")

    # Obtain settings for satuation related issues
    settings = get_camera_settings(header)
    print(f"Detected: {settings['mode']} ({settings['speed']} kHz)")

    # Obtain name
    name = header.get("OBSERVER")[:3]

    # Airmass for dispertion related computations
    airmass.append(header.get("AIRMASS", None))

saturation_limit = 0.9 * settings['saturation']  # safe margin

# --- Setup base results directory ---
base_results_dir = "results"
os.makedirs(base_results_dir, exist_ok=True)

for i, data in enumerate(fits_data):
        
        current_time = timestamps[i]
        current_filter = filters[i]
        
        # Format time to be safe for folder/file names
        safe_time = str(current_time).replace(":", "-").replace("T", "_")
        
        # Create a specific directory for this timestamp
        save_dir = os.path.join(base_results_dir, f"{name}{safe_time}_{current_filter}")
        os.makedirs(save_dir, exist_ok=True)
        
        # String to use in all plot titles
        meta_title = f"Date/Time: {current_time} | Filter: {current_filter}"

        # ----------------- Define circular mask for vignetting --------------------
        ny, nx = fits_data[i].shape
        y, x = np.indices((ny, nx))
    
        # Set the center and usable radius of the image
        center_x = nx // 2
        center_y = ny // 2
        good_radius = 1450  # Pixels from the center to keep
    
        # Calculate pixel distances from the center
        r = np.hypot(x - center_x, y - center_y)
    
        # Mask pixels outside our 'good' radius (True means ignore that data)
        vignette_mask = r > good_radius
    
        mean, median, std = sigma_clipped_stats(fits_data[i], sigma=3.0, mask=vignette_mask)
        print(f"\nProcessing image {i+1}/{len(fits_data)}: {current_time}")
        print(f"Global Background - Median: {median:.2f}, Std: {std:.2f}")
    
        data = fits_data[i] - median
    
        # --------- Initial Source Detection ---------
        threshold = 5.0 * std
        daofind = DAOStarFinder(fwhm=3, threshold=threshold, peakmax=saturation_limit) 
    
        # --------- PSF Modeling (Gaussian) ---------
        psf_g = GaussianPSF(x_fwhm=4, y_fwhm=4.5, theta=0.1)
        psf_g.x_fwhm.fixed = False
        psf_g.y_fwhm.fixed = False
        psf_g.theta.fixed = False
        
        box_size = 19
        fit_shape = (box_size, box_size)
        
        
        psfphot = PSFPhotometry(psf_g, fit_shape, finder=daofind, aperture_radius=4)
        phot = psfphot(data, mask=vignette_mask)
        print(f"Total detections before filtering: {len(phot)}")
        
        #  Calculate the fwhm and ellipticity
        fwhm_x = phot['x_fwhm_fit']
        fwhm_y = phot['y_fwhm_fit']
        
        a = np.maximum(fwhm_x, fwhm_y)
        b = np.minimum(fwhm_x, fwhm_y)
        
        phot['fwhm'] = np.sqrt(fwhm_x * fwhm_y)
        phot["ellipticity"] = 1 - (b/a)
        
        # ------------------------ PSF Modeling (Moffat) ----------------------
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
        
        
        psfphot_m = PSFPhotometry(psf_m, fit_shape, finder=daofind, aperture_radius=4)
        phot_m = psfphot_m(data, mask=vignette_mask)
        
        # Unfix all structural parameters so they can be fitted
        psf_m.gammax.fixed = False
        psf_m.gammay.fixed = False
        psf_m.phi.fixed = False
        psf_m.alpha.fixed = False
        # ADD BOUNDS HERE to keep the fitter from going crazy
        psf_m.gammax.bounds = (0.5, 10.0) # Prevent negative or massive widths
        psf_m.gammay.bounds = (0.5, 10.0)
        psf_m.alpha.bounds = (1.5, 5.0)   # Force alpha to be strictly > 1 for FWHM math
        
        
        # ---------------- FWHM & Ellipticity Extraction --------------
        # Since photutils fits our parameters, the table outputs <param>_fit
        # We manually calculate FWHM from gammax, gammay, and alpha using the Moffat formula
        alpha_fit = phot_m['alpha_fit']
        
        
        # Initialize with zeros to safely handle the math
        fwhm_x = np.zeros(len(phot_m))
        fwhm_y = np.zeros(len(phot_m))
        
        # Compute fwhm
        fwhm_x = 2.0 * np.abs(phot_m['gammax_fit']) * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
        fwhm_y = 2.0 * np.abs(phot_m['gammay_fit']) * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
        
        a = np.maximum(fwhm_x, fwhm_y)
        b = np.minimum(fwhm_x, fwhm_y)
        
        phot_m['fwhm_x'] = fwhm_x
        phot_m['fwhm_y'] = fwhm_y
        phot_m['fwhm'] = np.sqrt(fwhm_x * fwhm_y)
        
        # Suppress divide-by-zero warnings for failing sources during ellipticity calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            phot["ellipticity"] = 1 - (b/a)
        
        # -------------------------- Filtering process (G)----------------------
        # Filter out obvious failures first
        mask_valid = ((phot['flags'] == 0) & (phot['flux_fit'] > 0 ) &
        (phot["x_fwhm_fit"] > fwhm_min) & (phot["y_fwhm_fit"] > fwhm_min))
        phot = phot[mask_valid]


        
        # Filter out outliers
        filtered_fwhm = np.percentile(phot["fwhm"], [5, 90])
        mask_outliers = (phot["fwhm"] >= filtered_fwhm[0]) & (phot["fwhm"] <= filtered_fwhm[1])
        phot = phot[mask_outliers]
        
        # -------------------------- Filtering process (M) ----------------------
        # Filter out failures, invalid alpha values, and bad FWHMs
        mask_valid = ((phot_m['flags'] == 0) & (phot_m['flux_fit'] > 0 ) &  
                      (phot_m["fwhm_x"] > fwhm_min) & (phot_m["fwhm_y"] > fwhm_min))
        phot_m = phot_m[mask_valid]

        # Filter out outliers based on the calculated FWHM
        filtered_fwhm = np.percentile(phot_m["fwhm"], [5, 95])
        mask_outliers = (phot_m["fwhm"] >= filtered_fwhm[0]) & (phot_m["fwhm"] <= filtered_fwhm[1])
        phot_m = phot_m[mask_outliers]
        
        
        
        
        print(f"Found {len(phot)} sources to perform PSF photometry with (Gaussian)")
        print(f"Found {len(phot_m)} sources to perform PSF photometry with (Moffat)")
    
        if phot is None or len(phot) == 0:
            print("Error: No PSF photometry data to plot for this image.")
        else:
            # --- Source Tracking Logic ---
            # Calculate distance from every detected source to our reference point
            def dist(target):
                
                x = np.sqrt((phot["x_fit"] - target[0])**2 + 
                        (phot["y_fit"] - target[1])**2)
            
                return x
            
            dist_coma = dist(target_coma)
            
            dist_round = dist(target_round)
        
            # Find the index of the closest source
            closest_idx_coma = np.argmin(dist_coma)
            closest_idx_round = np.argmin(dist_round)
        
            # Only proceed if the closest star is within a reasonable distance (e.g., 10px)
            # This prevents "jumping" to a different star if the target is missing
            if (dist_coma[closest_idx_coma] < 
                search_radius) and (dist_round[closest_idx_round] < 
                                    search_radius):
                coma_x = phot['x_fit'][closest_idx_coma]
                coma_y = phot['y_fit'][closest_idx_coma]
                
                round_x = phot["x_fit"][closest_idx_round]
                round_y = phot["y_fit"][closest_idx_round]
                print(f"Tracking source at: ({coma_x:.2f}, {coma_y:.2f})")
                print(f"Tracking source at: ({round_x:.2f}, {round_y:.2f})")
            else:
                print(f"Warning: Target source not found within {search_radius}px. Skipping plots.")
                continue            
            
            # We need to prepare the model data before plotting it
            
            # Make a copy of the filtered table so we don't mess up  actual data
            clean_phot = phot.copy()

            # Rename the columns to match what the standard model expects
            clean_phot.rename_column('x_fit', 'x_0')
            clean_phot.rename_column('y_fit', 'y_0')
            clean_phot.rename_column('flux_fit', 'flux')
            clean_phot.rename_column('x_fwhm_fit', 'x_fwhm')
            clean_phot.rename_column('y_fwhm_fit', 'y_fwhm')
            clean_phot.rename_column('theta_fit', 'theta')
            
            # Exactly the same for the Moffat
            cln_phot = phot_m.copy()
            # Rename the columns to match what the standard model expects
            # Rename columns back to exactly match our Moffat parameters
            cln_phot.rename_column('x_fit', 'x_0')
            cln_phot.rename_column('y_fit', 'y_0')
            cln_phot.rename_column('flux_fit', 'flux')
            cln_phot.rename_column('gammax_fit', 'gammax')
            cln_phot.rename_column('gammay_fit', 'gammay')
            cln_phot.rename_column('phi_fit', 'phi')
            cln_phot.rename_column('alpha_fit', 'alpha')
            
            # Generate the image  (Gaussian)
            model_image = make_model_image(
                data.shape, 
                psf_g, 
                clean_phot, 
                model_shape=fit_shape
                )
            
            # Generate the image (Moffat)
            model_image_m = make_model_image(
                data.shape,
                psf_m,
                cln_phot,
                model_shape=fit_shape)
            
            
            
            # ----------------- Residual Analysis Plots -----------------
            residual_image = data - model_image.astype(float)
            residual_image_m = data - model_image_m.astype(float)
            # Define a zoom-in window 
            zoom_size = 30
            
            # Create a list of the sources you want to plot
            targets_to_plot = [
                ("Coma_Source", coma_x, coma_y),
                ("Round_Source", round_x, round_y)
            ]

            # Create a list of the models you want to evaluate
            models_to_plot = [
                ("Gaussian", model_image, residual_image),
                ("Moffat", model_image_m, residual_image_m)
            ]
            
            # Loop through each model, and then through each target
            for model_name, current_model, current_residual in models_to_plot:
                for source_name, target_x, target_y in targets_to_plot:
                    
                    # Define a zoom-in window based on the CURRENT target
                    x_min, x_max = int(target_x - zoom_size), int(target_x + zoom_size)
                    y_min, y_max = int(target_y - zoom_size), int(target_y + zoom_size)

                    # Setup Figure with GridSpec
                    fig = plt.figure(figsize=(18, 10))
                    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.6])
                    
                    ax_data = fig.add_subplot(gs[0, 0])
                    ax_model = fig.add_subplot(gs[0, 1])
                    ax_res = fig.add_subplot(gs[0, 2])
                    ax_plot_x = fig.add_subplot(gs[1, 0:2]) 
                    ax_plot_y = fig.add_subplot(gs[1, 2])

                    # Update the title to include the source name AND model name
                    fig.suptitle(f"{source_name.replace('_', ' ')} Residual Analysis ({model_name}) - {meta_title}\nSource at ({target_x:.1f}, {target_y:.1f})", 
                                 fontsize=14, fontweight='bold')

                    norm_res = ImageNormalize(current_residual, stretch=SqrtStretch(), vmin=-10, vmax=500)

                    # 2D Zoomed Plots
                    for ax, img, title in zip([ax_data, ax_model, ax_res], 
                                              [data, current_model, current_residual], 
                                              ["Original", f"Model ({model_name})", "Residual"]):
                        im = ax.imshow(img, origin='lower', cmap='grey', norm=norm_res)
                        ax.set_title(title)
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
        
                    plt.colorbar(im, ax=ax_res, label='Counts')

                    # 1D Residual Profiles
                    row_idx = int(round(target_y))
                    col_idx = int(round(target_x))
                    x_range = np.arange(x_min, x_max)
                    y_range = np.arange(y_min, y_max)
                        
                    # Horizontal Profile
                    ax_plot_x.plot(x_range, data[row_idx, x_min:x_max], 'k.', label='Data', alpha=0.4)
                    ax_plot_x.plot(x_range, current_model[row_idx, x_min:x_max], 'r-', label=f'{model_name} Model', linewidth=2)
                    ax_plot_x.step(x_range, current_residual[row_idx, x_min:x_max], where='mid', color='blue', label='Residual')
                    ax_plot_x.axhline(0, color='black', linestyle='--', alpha=0.5)
                    ax_plot_x.set_title(f"X-Profile (Row {row_idx})")
                    ax_plot_x.legend()
                    
                    # Vertical Profile
                    ax_plot_y.plot(data[y_min:y_max, col_idx], y_range, 'k.', alpha=0.4)
                    ax_plot_y.plot(current_model[y_min:y_max, col_idx], y_range, 'r-', linewidth=2)
                    ax_plot_y.step(current_residual[y_min:y_max, col_idx], y_range, where='mid', color='blue')
                    ax_plot_y.axvline(0, color='black', linestyle='--', alpha=0.5)
                    ax_plot_y.set_title(f"Y-Profile (Col {col_idx})")

                    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
                    
                    # Update the save filename to include the model name so they don't overwrite each other
                    fig.savefig(os.path.join(save_dir, f"{source_name}_Residual_{model_name}_{safe_time}.png"))
                    
                    
                    plt.show()
                    plt.close('all')
                    # ----------------- Elegant Model Comparison Plot -----------------
                    # Loop through both sources (Coma and Round) to compare models
                    for source_name, target_x, target_y in targets_to_plot:
                
                        zoom_size = 30  # Keep it tight to focus on the core and immediate wings
                
                        x_min, x_max = int(target_x - zoom_size), int(target_x + zoom_size)
                        y_min, y_max = int(target_y - zoom_size), int(target_y + zoom_size)
                
                        row_idx = int(round(target_y))
                        col_idx = int(round(target_x))
                        x_range = np.arange(x_min, x_max)
                        y_range = np.arange(y_min, y_max)
                
                        # Setup a clean 1x2 side-by-side figure
                        fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(14, 6))
                
                        clean_name = source_name.replace('_', ' ')
                        fig.suptitle(f"PSF Profile Comparison ({clean_name}): Gaussian vs. Moffat\n{meta_title}", 
                             fontsize=16, fontweight='bold', y=1.05)
                
                        # --- Left Panel: X-Profile ---
                        ax_x.plot(x_range, data[row_idx, x_min:x_max], 'ko', label='Raw Data', alpha=0.4, markersize=6)
                        ax_x.plot(x_range, model_image[row_idx, x_min:x_max], color='dodgerblue', linestyle='--', linewidth=2.5, label='Gaussian Model')
                        ax_x.plot(x_range, model_image_m[row_idx, x_min:x_max], color='crimson', linestyle='-', linewidth=2.5, label='Moffat Model')
                
                        ax_x.set_title("Horizontal Cut (X-Axis)", fontsize=14)
                        ax_x.set_xlabel("X Pixel Coordinate", fontsize=12)
                        ax_x.set_ylabel("Counts (Symlog Scale)", fontsize=12)
                
                        # --- Right Panel: Y-Profile ---
                        ax_y.plot(y_range, data[y_min:y_max, col_idx], 'ko', label='Raw Data', alpha=0.4, markersize=6)
                        ax_y.plot(y_range, model_image[y_min:y_max, col_idx], color='dodgerblue', linestyle='--', linewidth=2.5, label='Gaussian Model')
                        ax_y.plot(y_range, model_image_m[y_min:y_max, col_idx], color='crimson', linestyle='-', linewidth=2.5, label='Moffat Model')
                
                        ax_y.set_title("Vertical Cut (Y-Axis)", fontsize=14)
                        ax_y.set_xlabel("Y Pixel Coordinate", fontsize=12)
                
                    # --- Elegant Formatting for both axes ---
                        for ax in [ax_x, ax_y]:
                            # Symlog allows logarithmic scaling while safely handling negative background noise values
                            ax.set_yscale('symlog', linthresh=20) 
                            ax.grid(True, alpha=0.3, linestyle=':')
                            ax.legend(fontsize=11, frameon=True, shadow=False, edgecolor='black')
                    
                            # Remove top and right spines for a modern, minimal aesthetic
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                
                        fig.tight_layout()
                
                        # Save at a high DPI for crisp, presentation-ready quality (filename includes source_name)
                        fig.savefig(os.path.join(save_dir, f"Comparison_Gaussian_vs_Moffat_{source_name}_{safe_time}.png"), dpi=300, bbox_inches='tight')
                
                    
                    
    
    
    
