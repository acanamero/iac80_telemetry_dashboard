=============================================================================
                  LIVE TELESCOPE TELEMETRY PIPELINE
=============================================================================

This software provides a real-time, threaded telemetry dashboard for 
monitoring telescope focus, seeing (FWHM), guiding errors, and ambient 
conditions during an observing night. It processes FITS files 
as they are saved by the camera control software.

--- HOW TO RUN (IAC80 TUTORIAL) ---

If you are using the IAC80 telescope, follow these steps to start the pipeline:

    1. Open a terminal.

    2. Type the command live_monitor and press Enter.

    3. Two windows should automatically open on your screen: Firefox (with an error message) and the Ginga image viewer.

    4. Look back at your terminal. It will ask you if you want to observe today's date:

        -If you decline (e.g., you want to analyze past data), it will prompt you to manually enter a specific date.

        -If you agree (e.g., saying yes to observing tonight's night), the system will lock onto the current directory.

    5. Once you have made your selection and the terminal confirms the start, simply go to your open web browser page and refresh it.

You will now see the real-time telemetry dashboard actively monitoring your session, good luck and clear skies!

--- ARCHITECTURE OVERVIEW ---

The software is divided into 7 modular files to separate the UI, the math, 
the state, and the configuration. 

1. main.py (The Entry Point)
   Initializes the system. Prompts the user to target an observing directory, 
   loads any existing CSV data to resume an interrupted session, spins up the 
   folder Watchdog daemon, and starts the Flask web server.

2. config.py (The Telescope Profile)
   The single source for all hardcoded telescope constants. It 
   contains the observatory's GPS coordinates, the camera's read-modes/speeds, 
   the detector's pixel scale, and the naming conventions for FITS files and 
   directories.

3. processing.py (The Astrometry Engine)
   The heavy lifter. Triggered automatically when a new FITS file arrives. 
   It opens the file, runs DAOStarFinder, fits a custom 2D Moffat profile 
   to the sources, calculates guiding drift (dx/dy) using cKDTree, and saves 
   the final telemetry to the global state.

4. web.py (The Dashboard Interface)
   A Flask server that serves the frontend HTML/JS/Plotly UI. It provides 
   API endpoints (/data) that the frontend aggressively polls every 20 seconds 
   to update the graphs in real-time. Also handles CSV uploads and downloads.

5. state.py (Global Memory)
   A purely structural file that holds the runtime arrays (history_fwhm, 
   history_dx, etc.) and threading locks. Isolating the state prevents 
   circular import errors between the web server and the processing engine.

6. models.py (Custom Mathematics)
   Houses the `Moffat2Dell` class, a customized Astropy Fittable2DModel 
   used to accurately measure the Point Spread Function (PSF) and elliptical 
   rotation angle (Phi) of the stars.

7. utils.py (I/O & Helpers)
   Handles writing the session data to CSVs on disk, loading CSVs back into 
   memory upon script restart, interacting with the local Ginga FITS viewer, 
   and calculating rollover dates for directories.


--- ADAPTING TO A NEW TELESCOPE ---

This pipeline is designed to be completely telescope-agnostic. To deploy 
this software at a new observatory, you ONLY need to edit `config.py`. 

Open `config.py` and modify the following parameters:

1. Astrometry & Physical Location
   * LOCATION: Change the latitude, longitude, and elevation to match the 
     new observatory (used for Alt/Az all-sky mapping).
   * ARCSEC_PIXEL: Update to match the new camera's pixel scale so FWHM 
     and guiding errors are accurately scaled to arcseconds.

2. File & Directory Naming Rules
   * BASE_DIR: The root folder where the camera saves images.
   * FITS_GLOB_PATTERN: The search string for science images (e.g., '*.fits').
   * get_night_dir_name(): Modify how the script looks for tonight's folder 
     (e.g., 'YYYY-MM-DD' vs 'YYMMMDD').
   * get_fits_index() & get_fits_sort_number(): Adjust the string slicing 
     so the script correctly extracts the unique ID and chronological sort 
     number from the new telescope's file naming scheme.

3. Camera Hardware
   * get_camera_settings(): Update the string-matching logic to parse the 
     FITS header for the new camera's specific readout speeds and saturation 
     limits.

Once `config.py` is updated, simply run `python3 main.py` 
