#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:29:57 2026

@author: acanamero-ext
"""

import os
import time
import queue
import threading
import glob
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

import state
from config import BASE_DIR, FITS_GLOB_PATTERN, get_fits_sort_number, get_fits_index
from utils import get_observing_night_dir, load_csv_to_history, save_csv_to_disk, update_ginga_viewer
from processing import process_single_fits
from web import app

class FitsHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.fits'): 
            time.sleep(2)
            state.fits_queue.put(event.src_path) 

def process_queue_loop():
    while True:
        try:
            new_fits = state.fits_queue.get(timeout=1.0)
            time.sleep(0.5) 
            try:
                process_single_fits(new_fits)
                update_ginga_viewer(new_fits)
            except Exception as e:
                print(f"\n[ERROR] Failed to process {new_fits}: {e}")
        except queue.Empty: continue

if __name__ == "__main__": 
    tonights_dir = get_observing_night_dir(BASE_DIR)
    tonights_folder_name = os.path.basename(tonights_dir)
    
    while True:
        choice = input(f"Do you want to observe tonight's night ({tonights_folder_name})? (y/n): ").strip().lower()
        if choice in ['y', 'yes', 'n', 'no']: break
        print("Please enter 'y' or 'n'.")

    INCOMING_DIR = tonights_dir if choice in ['y', 'yes'] else ""
    if INCOMING_DIR: print(f"Targeting observation directory: {INCOMING_DIR}")

    while not os.path.exists(INCOMING_DIR):
        if INCOMING_DIR != "": print(f"\nWARNING: The directory {INCOMING_DIR} does not exist yet.")
        manual_night = input("Which night do you want to observe? Enter format yyMmmdd (e.g. 26Apr13) or 'q' to quit: ").strip()
        if manual_night.lower() == 'q': exit(1)
            
        INCOMING_DIR = os.path.join(BASE_DIR, manual_night)
        if os.path.exists(INCOMING_DIR):
            print(f"Success! Targeting new observation directory: {INCOMING_DIR}\n")
            break 
        else: print(f"ERROR: The directory {INCOMING_DIR} does not exist. Let's try again.\n")

    processed_indices = load_csv_to_history(INCOMING_DIR)
    existing_fits_files = glob.glob(os.path.join(INCOMING_DIR, FITS_GLOB_PATTERN))
    existing_fits_files.sort(key=get_fits_sort_number)
    
    queued_count = 0
    for file_path in existing_fits_files:
        idx_to_check = get_fits_index(file_path)
        if idx_to_check not in processed_indices:
            state.fits_queue.put(file_path)
            queued_count += 1
            
    print(f"Skipped {len(existing_fits_files) - queued_count} previously processed FITS files.")
    print(f"Queued {queued_count} new existing FITS files.")
    
    event_handler = FitsHandler()
    observer = PollingObserver(timeout=30)
    observer.schedule(event_handler, path=INCOMING_DIR, recursive=False)
    observer.start()
    
    processor_thread = threading.Thread(target=process_queue_loop, daemon=True)
    processor_thread.start()

    print(f"Listening for new FITS files in {INCOMING_DIR}...")
    print("Starting Web Dashboard on http://127.0.0.1:5000/")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"\nServer stopped due to an error: {e}")
    finally:
        print("\nStopping server...")
        save_csv_to_disk(INCOMING_DIR)
        observer.stop()
        observer.join()
        print("Shutdown complete. Have a good sleep!")