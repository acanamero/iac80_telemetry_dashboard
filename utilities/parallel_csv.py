import os
import glob
import time
import csv
import warnings
import multiprocessing
import contextlib
import joblib
import sys  # Essential for reading Slurm arguments
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np

# Astropy & Photutils imports
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry

from scipy.spatial import cKDTree
from joblib import Parallel, delayed

# --- IMPORTS FROM alpha_html ---
try:
    from alpha_html import IAC_LOCATION, FILTER_COLORS, get_camera_settings, Moffat2Dell
except ImportError as e:
    print(f"Import Error: {e}")
    IAC_LOCATION = None 
    FILTER_COLORS = {'U': 'blue', 'B': 'cyan', 'V': 'green', 'R': 'red', 'I': 'magenta'}
    def get_camera_settings(header):
        return {"mode": "Unknown", "speed": None, "saturation": 40000, "recommended_flat": None}

# --- PARALLEL WORKER FUNCTION ---
def process_single_fits(fits_file):
    """Processes a single FITS file and returns a dictionary of telemetry data."""
    fwhm_min = 2.0
    index_csv = os.path.basename(fits_file)[-18:-5]
    result_data = None
    
    try:
        with fits.open(fits_file) as file_data:
            image_data = file_data[0].data.astype(float)
            header = file_data[0].header
            fits_type = str(header.get('IMAGETYP', 'UNKNOWN')).strip().upper()
            
            if fits_type != 'OBJECT':
                return None
            
            obs_time = header.get('DATE-OBS', 'Unknown_Time')
            filter_name = header.get('INSFILTE', 'Unknown_Filter')
            settings = get_camera_settings(header)
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
            
            psf_m.gammax.bounds = (0.5, 10.0)
            psf_m.gammay.bounds = (0.5, 10.0)
            psf_m.alpha.bounds = (1.5, 5.0)   
            
            psfphot = PSFPhotometry(psf_m, (25, 25), finder=daofind, aperture_radius=4)
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                phot = psfphot(data, mask=vignette_mask)

            alpha_fit = phot['alpha_fit']
            fwhm_x = 2.0 * np.abs(phot['gammax_fit']) * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
            fwhm_y = 2.0 * np.abs(phot['gammay_fit']) * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
            
            phot['fwhm_x'] = fwhm_x
            phot['fwhm_y'] = fwhm_y
            phot['fwhm'] = np.sqrt(fwhm_x * fwhm_y)
                
            mask_valid = ((phot['flags'] == 0) & (phot['flux_fit'] > 0) &  
                          (phot["fwhm_x"] > fwhm_min) & (phot["fwhm_y"] > fwhm_min))
            phot = phot[mask_valid]

            if len(phot) > 0:
                filtered_fwhm = np.percentile(phot["fwhm"], [5, 95])
                phot = phot[(phot["fwhm"] >= filtered_fwhm[0]) & (phot["fwhm"] <= filtered_fwhm[1])]
                perc_x = np.percentile(phot["fwhm_x"], 75) - np.percentile(phot["fwhm_x"], 25)
                perc_y = np.percentile(phot["fwhm_y"], 75) - np.percentile(phot["fwhm_y"], 25)
                
                try:
                    obs_dt = datetime.fromisoformat(obs_time)
                except ValueError:
                    obs_dt = datetime.now()
                    
                header_ra = header.get('RA', header.get('OBJRA', 'UNKNOWN'))
                header_dec = header.get('DEC', header.get('OBJDEC', 'UNKNOWN'))
                alt_val, az_val = None, None
                
                if header_ra != 'UNKNOWN' and header_dec != 'UNKNOWN' and IAC_LOCATION is not None:
                    try:
                        coord = SkyCoord(header_ra, header_dec, unit=(u.hourangle, u.deg))
                        altaz = coord.transform_to(AltAz(obstime=Time(obs_dt), location=IAC_LOCATION))
                        alt_val = altaz.alt.degree
                        az_val = altaz.az.degree
                    except:
                        pass 

                result_data = {
                    'INDEX': index_csv, 'UTC': obs_dt.isoformat(), 'AIRMASS': header.get('AIRMASS', None),
                    'FILTER': filter_name, 'TELFOCUS': header.get('TELFOCUS', None),
                    'HUMIDITY': header.get('HUMIDITY', None), 'TEMPERATURE': header.get('TEMP', None),
                    'RA': header_ra, 'DEC': header_dec, 'ALT': alt_val, 'AZ': az_val,
                    'GAMMA_X': np.median(phot['gammax_fit']), 'FWHM_X': np.median(phot['fwhm_x']),
                    'ERR_X': perc_x, 'GAMMA_Y': np.median(phot['gammay_fit']),
                    'FWHM_Y': np.median(phot['fwhm_y']), 'ERR_Y': perc_y,
                    'ALPHA': np.median(phot['alpha_fit']), 'PHI': np.median(phot['phi_fit']),
                    'NSOURCES': len(phot), 'DX': 0.0, 'DY': 0.0,
                    'X_FIT': phot['x_fit'].value if hasattr(phot['x_fit'], 'value') else phot['x_fit'],
                    'Y_FIT': phot['y_fit'].value if hasattr(phot['y_fit'], 'value') else phot['y_fit']
                }
    except Exception as e:
        print(f"Error processing {fits_file}: {e}")
    return result_data

# --- SAVING LOGIC ---
def save_csv_to_disk(results_list, output_filename, arcsec_pixel):
    valid_results = [r for r in results_list if r is not None]
    if not valid_results: return

    valid_results.sort(key=lambda x: x['UTC'])
    base_coord, base_xy = None, None
    
    for row in valid_results:
        if row['RA'] != 'UNKNOWN' and row['DEC'] != 'UNKNOWN':
            current_coord = SkyCoord(row['RA'], row['DEC'], unit=(u.hourangle, u.deg))
            curr_xy = np.vstack([row['X_FIT'], row['Y_FIT']]).T
            if base_coord is None:
                base_coord, base_xy = current_coord, curr_xy
            else:
                separation = current_coord.separation(base_coord)
                if separation < 3 * u.arcmin:
                    tree = cKDTree(base_xy)
                    dist, idx = tree.query(curr_xy, k=1)
                    good = dist < 5
                    if np.sum(good) >= 5:
                        dx = curr_xy[good, 0] - base_xy[idx[good], 0]
                        dy = curr_xy[good, 1] - base_xy[idx[good], 1]
                        mask = (np.abs(dx - np.median(dx)) < 2) & (np.abs(dy - np.median(dy)) < 2)
                        row['DX'] = np.median(dx[mask]) * arcsec_pixel
                        row['DY'] = np.median(dy[mask]) * arcsec_pixel
                else:
                    base_coord, base_xy = current_coord, curr_xy
        del row['X_FIT'], row['Y_FIT']

    fieldnames = ['INDEX', 'UTC', 'AIRMASS', 'FILTER', 'TELFOCUS', 'HUMIDITY', 'TEMPERATURE', 
                  'RA', 'DEC', 'ALT', 'AZ', 'GAMMA_X', 'FWHM_X', 'ERR_X', 'GAMMA_Y', 'FWHM_Y', 
                  'ERR_Y', 'ALPHA', 'PHI', 'NSOURCES', 'DX', 'DY']
    with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_results)

# --- PROGRESS BAR HELPER ---
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try: yield tqdm_object
    finally: joblib.parallel.BatchCompletionCallBack = old_batch_callback

# --- MAIN BLOCK ---
if __name__ == "__main__":
    # Handle Slurm CPU Allocation
    if len(sys.argv) > 1:
        num_cores = int(sys.argv[1])
    else:
        num_cores = multiprocessing.cpu_count()
    
    print(f"Executing parallel job with {num_cores} cores.")

    base_search_directory = "/net/nas/proyectos/ttnn/camelot2/data_raw" 
    output_directory = "/scratch/acanamero/nights_csv/"
    os.makedirs(output_directory, exist_ok=True)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 4)
    threshold_date = datetime(2025, 7, 4)

    current_date = start_date
    while current_date <= end_date:
        loop_date = current_date
        folder_name = loop_date.strftime("%y%b%d")
        target_directory = os.path.join(base_search_directory, folder_name)
        current_date += timedelta(days=1)
        
        current_arcsec_pixel = 0.322 if loop_date < threshold_date else 0.336
        
        output_csv_path = os.path.join(output_directory, f"{folder_name}_p.csv")
        if not os.path.isdir(target_directory) or os.path.exists(output_csv_path):
            continue

        im_lst = glob.glob(os.path.join(target_directory, "O*.fits"))
        if len(im_lst) == 0: continue
            
        print(f"\nAnalyzing: {folder_name} | Files: {len(im_lst)} | Scale: {current_arcsec_pixel}")

        with tqdm_joblib(tqdm(desc=f"Processing {folder_name}", total=len(im_lst))) as progress_bar:
            results_lst = Parallel(n_jobs=num_cores)(
                delayed(process_single_fits)(f) for f in im_lst
            )

        save_csv_to_disk(results_lst, output_csv_path, current_arcsec_pixel)
        
    print("\nProcessing complete.")