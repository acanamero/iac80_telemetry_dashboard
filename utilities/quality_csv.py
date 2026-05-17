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

def generate_quality_plots(csv_directory):
    search_pattern = os.path.join(csv_directory, "*_p.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {csv_directory}")
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
                
                # Convert PHI to degrees and map to [-90, 90]
                raw_deg = df['PHI'] * (180.0 / np.pi)
                df['PHI_deg'] = ((raw_deg + 90.0) % 180.0) - 90.0
                
                # --- WEIGHTED SCORE PARAMETERS ---
                # Set these to >1.0 to increase a factor's impact, or <1.0 to decrease it.
                # Kept at 1.0 for now 
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

                # Medians and Averages
                med_x = df['FWHM_X_arcsec'].median()
                med_y = df['FWHM_Y_arcsec'].median()
                std_x = 0 if pd.isna(df['FWHM_X_arcsec'].std()) else df['FWHM_X_arcsec'].std()
                std_y = 0 if pd.isna(df['FWHM_Y_arcsec'].std()) else df['FWHM_Y_arcsec'].std()
                avg_phi = df['PHI_deg'].mean()
                med_qs = df['Quality_Score'].median()
                
                if pd.notna(med_x) and pd.notna(med_y):
                    data_points.append({
                        'Date': date_obj, 'FWHM_X_med': med_x, 'FWHM_Y_med': med_y,
                        'FWHM_X_std': std_x, 'FWHM_Y_std': std_y, 'PHI_avg': avg_phi,
                        'Quality_Score': med_qs
                    })
        except Exception as e:
            print(f"Error reading {basename}: {e}")
            
    if not data_points:
        print("No valid data points extracted.")
        return
        
    results_df = pd.DataFrame(data_points)
    results_df.sort_values(by='Date', inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    
    # --- DYNAMIC DATA SPLITTER ---
    # Find the largest gap in the dates
    time_diffs = results_df['Date'].diff()
    max_gap_idx = time_diffs.idxmax()
    
    # If the largest gap is more than 40 days, we split the data into two panels
    if pd.notna(max_gap_idx) and time_diffs.max().days > 40:
        df1 = results_df.iloc[:max_gap_idx]
        df2 = results_df.iloc[max_gap_idx:]
        
        # Calculate proportional widths so the x-axis scale is visually identical in both panels
        span1 = max((df1['Date'].max() - df1['Date'].min()).days, 15)
        span2 = max((df2['Date'].max() - df2['Date'].min()).days, 15)
        width_ratios = [span1, span2]
    else:
        # Fallback if there are no major gaps
        df1 = results_df
        df2 = None
        width_ratios = [1]
    # -----------------------------

    print("\nGenerating and saving plots...")
    
    # Define a helper function to format the dates on axes
    def format_xaxis(ax):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # PLOT 1: Median FWHM
    # ---------------------------------------------------------
    if df2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.05})
        axes = [ax1, ax2]
        dfs = [df1, df2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax1]
        dfs = [df1]

    for ax, df_plot in zip(axes, dfs):
        ax.errorbar(df_plot['Date'], df_plot['FWHM_X_med'], yerr=df_plot['FWHM_X_std'], 
                    fmt='-o', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.5, capsize=0, 
                    alpha=0.8, label='Median FWHM_X (±1 Std Dev)')
        ax.errorbar(df_plot['Date'], df_plot['FWHM_Y_med'], yerr=df_plot['FWHM_Y_std'], 
                    fmt='-s', color='#ff7f0e', ecolor='#ff7f0e', elinewidth=1.5, capsize=0, 
                    alpha=0.8, label='Median FWHM_Y (±1 Std Dev)')
        format_xaxis(ax)

    axes[0].set_ylabel('FWHM (arcseconds)', fontsize=12)
    axes[0].legend(fontsize=11, loc='upper right')
    fig.suptitle('Daily Median FWHM', fontsize=14)
    
    plt.savefig("1_fwhm_median_with_variability_2026.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ---------------------------------------------------------
    # PLOT 2: Average PHI
    # ---------------------------------------------------------
    if df2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.05})
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax1]

    for ax, df_plot in zip(axes, dfs):
        ax.plot(df_plot['Date'], df_plot['PHI_avg'], marker='^', linestyle='-', color='#2ca02c', label='Average PHI')
        format_xaxis(ax)
        ax.set_ylim(-100, 100)
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])

    axes[0].set_ylabel('PHI (Degrees)', fontsize=12)
    axes[0].legend(fontsize=11)
    fig.suptitle('Daily Average Rotation Angle (PHI)', fontsize=14)
    
    plt.savefig("2_average_phi_2026.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ---------------------------------------------------------
    # PLOT 3: Composite Image Quality Score
    # ---------------------------------------------------------
    if df2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.05})
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax1]

    for ax, df_plot in zip(axes, dfs):
        ax.plot(df_plot['Date'], df_plot['Quality_Score'], marker='D', linestyle='-', color='#9467bd', label='Nightly Quality Score', linewidth=2)
        format_xaxis(ax)

    axes[0].set_ylabel('Quality Score Index', fontsize=12)
    axes[0].legend(fontsize=11)
    fig.suptitle('Composite Image Quality Score', fontsize=14)
    
    plt.savefig("3_composite_quality_score_2026.png", dpi=300, bbox_inches='tight')
    plt.show()
    

    print("Success! Created panel plots.")

if __name__ == "__main__":
    INPUT_CSV_DIRECTORY = "/home/acanamero-ext/practicas/scripts/nights_csv/2026"
    generate_quality_plots(INPUT_CSV_DIRECTORY)