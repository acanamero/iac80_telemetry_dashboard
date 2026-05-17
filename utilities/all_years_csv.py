#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:41:09 2026

@author: acanamero-ext
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# English month abbreviations to numbers map (ignores system locale)
MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def parse_custom_date(date_str):
    if len(date_str) != 7:
        raise ValueError("Invalid length")
        
    year = 2000 + int(date_str[0:2])
    month_abbr = date_str[2:5].capitalize()
    day = int(date_str[5:7])
    
    if month_abbr not in MONTH_MAP:
        raise KeyError(f"Unknown month abbreviation: {month_abbr}")
        
    return datetime(year, MONTH_MAP[month_abbr], day)

def get_pixel_scale(date_obj):
    threshold_date = datetime(2025, 7, 4)
    return 0.322 if date_obj < threshold_date else 0.336

def generate_score_comparison_plot(base_directory):
    # Search for all _p.csv files recursively to grab 2023, 2024, 2025, and 2026
    search_pattern = os.path.join(base_directory, "**", "*_p.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {base_directory} or its subdirectories.")
        return
        
    data_points = []
    print(f"Found {len(csv_files)} CSV files. Processing metrics...")
    
    for file in csv_files:
        basename = os.path.basename(file)
        date_str = basename.split('_')[0]
        
        try:
            date_obj = parse_custom_date(date_str)
        except Exception as e:
            continue
            
        try:
            df = pd.read_csv(file)
            
            required_cols = ['FWHM_X', 'FWHM_Y', 'ERR_X', 'ERR_Y', 'PHI']
            if all(col in df.columns for col in required_cols):
                
                arcsec_pixel = get_pixel_scale(date_obj)
                
                df['FWHM_X_arcsec'] = df['FWHM_X'] * arcsec_pixel
                df['FWHM_Y_arcsec'] = df['FWHM_Y'] * arcsec_pixel
                df['ERR_X_arcsec'] = df['ERR_X'] * arcsec_pixel
                df['ERR_Y_arcsec'] = df['ERR_Y'] * arcsec_pixel
                
                # --- WEIGHTED SCORE PARAMETERS ---
                # >1 gives strength to the desired parameter
                w_baseline = 1.0
                w_ellipticity = 1.0
                w_consistency = 1.0
                
                # Image Quality Score calculation
                overall_fwhm = (df['FWHM_X_arcsec'] + df['FWHM_Y_arcsec']) / 2.0
                baseline_score = 1.0 / overall_fwhm
                
                ellipticity = df[['FWHM_X', 'FWHM_Y']].max(axis=1) / df[['FWHM_X', 'FWHM_Y']].min(axis=1)
                ellipticity_penalty = 1.0 / ellipticity
                
                avg_err = (df['ERR_X_arcsec'] + df['ERR_Y_arcsec']) / 2.0
                field_consistency = overall_fwhm / (overall_fwhm + avg_err)
                
                # Apply weights using exponents for the multiplicative model
                df['Quality_Score'] = (baseline_score ** w_baseline) * \
                                      (ellipticity_penalty ** w_ellipticity) * \
                                      (field_consistency ** w_consistency)

                # Get Medians for the night
                med_qs = df['Quality_Score'].median()
                med_fwhm = overall_fwhm.median() # Merged FWHM metric
                
                if pd.notna(med_qs) and pd.notna(med_fwhm):
                    data_points.append({
                        'Date': date_obj, 
                        'Year': date_obj.year,
                        'Quality_Score': med_qs,
                        'Overall_FWHM': med_fwhm
                    })
        except Exception as e:
            print(f"Error reading {basename}: {e}")
            
    if not data_points:
        print("No valid data points extracted.")
        return
        
    results_df = pd.DataFrame(data_points)
    
    # --- ALIGN YEARS FOR COMPARISON ---
    results_df['Plot_Date'] = results_df['Date'].apply(lambda d: d.replace(year=2024))
    
    # Get distinct years and assign colors
    years = sorted(results_df['Year'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
    
    print("\nGenerating Plot 1: Smoothed Quality Score...")
    
    # ==========================================
    # PLOT 1: QUALITY SCORE
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    
    for year, color in zip(years, colors):
        year_data = results_df[results_df['Year'] == year].sort_values(by='Plot_Date').copy()
        
        #Fix for the gaps
        # Remove duplicate days if multiple exist
        year_data = year_data.drop_duplicates(subset=['Plot_Date'])
        # Set the date as the index
        year_data.set_index('Plot_Date', inplace=True)
        # Resample to strict daily frequency (fills missing days with NaN)
        year_data = year_data.resample('D').asfreq()

        # Raw daily data
        ax1.plot(year_data.index, year_data['Quality_Score'], 
                marker='D', markersize=3, linestyle='-', color=color, linewidth=0.5, alpha=0.25)
        
        # Smoothed trend line
        # min_periods=2 ensures that a trend line isn't drawn over long empty gaps
        year_data['Trend_QS'] = year_data['Quality_Score'].rolling(window=14, min_periods=2).mean()
        ax1.plot(year_data.index, year_data['Trend_QS'], 
                linestyle='-', color=color, linewidth=3, alpha=1.0, label=str(year))

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.tick_params(axis='x', rotation=45, labelsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax1.set_ylabel('Quality Score Index', fontsize=12)
    ax1.set_xlabel('Day and Month', fontsize=12)
    ax1.legend(title='Observation Year\n(14-Day Trend)', fontsize=11, loc='best')
    fig1.suptitle('Image Quality Score Over the Years', fontsize=15)
    
    plt.tight_layout()
    fig1.savefig("3_all_years_qs_smoothed.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Generating Plot 2: Smoothed Overall FWHM...")

    # ==========================================
    # PLOT 2: OVERALL FWHM
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    
    for year, color in zip(years, colors):
        year_data = results_df[results_df['Year'] == year].sort_values(by='Plot_Date').copy()
        
        # fix for gaps
        year_data = year_data.drop_duplicates(subset=['Plot_Date'])
        year_data.set_index('Plot_Date', inplace=True)
        year_data = year_data.resample('D').asfreq()
        
        # Raw daily data
        ax2.plot(year_data.index, year_data['Overall_FWHM'], 
                marker='o', markersize=3, linestyle='-', color=color, linewidth=0.5, alpha=0.25)
        
        #Smoothed trend line
        year_data['Trend_FWHM'] = year_data['Overall_FWHM'].rolling(window=14, min_periods=2).mean()
        ax2.plot(year_data.index, year_data['Trend_FWHM'], 
                linestyle='-', color=color, linewidth=3, alpha=1.0, label=str(year))

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.tick_params(axis='x', rotation=45, labelsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax2.set_ylabel('Overall FWHM (arcseconds)', fontsize=12)
    ax2.set_xlabel('Day and Month', fontsize=12)
    ax2.legend(title='Observation Year\n(14-Day Trend)', fontsize=11, loc='best')
    fig2.suptitle('Daily Median FWHM Over the Years', fontsize=15)
    
    plt.tight_layout()
    fig2.savefig("4_all_years_fwhm_smoothed.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Success! Created both smoothed comparison plots with accurate gap handling.")

if __name__ == "__main__":
    # Point this to the parent directory containing all the year folders (2023, 2024, etc.)
    INPUT_CSV_DIRECTORY = "/home/acanamero-ext/practicas/scripts/nights_csv"
    generate_score_comparison_plot(INPUT_CSV_DIRECTORY)