import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration for the plots, here i can change the font and adjust everything
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.2
})

df = pd.read_csv('/home/acanamero-ext/practicas/results/roi/fwhm_register_roi.csv')
# Needed to convert the data from utc to datetime.
df['UTC'] = pd.to_datetime(df['UTC'])

fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True, constrained_layout=True)

# 1. FWHM
ax[0].plot(df['UTC'], df['FWHM_X']*0.336, label='FWHM X', color='#1f77b4')
ax[0].plot(df['UTC'], df['FWHM_Y']*0.336, label='FWHM Y', color='#ff7f0e')
ax[0].set_ylabel('FWHM [arcmin]')

# 2. TEMPERATURE
ax[1].plot(df['UTC'], df['TEMPERATURE'], color='tab:red', label='Temp')
ax[1].set_ylabel('TEMP. [°C]')

# 3. HUMIDITY
ax[2].plot(df['UTC'], df['HUMIDITY'], color='tab:green', label='Hum')
ax[2].set_ylabel('HUM. [%]')

# 4. AIRMASS
ax[3].plot(df['UTC'], df['AIRMASS'], color='black', label='Airmass')
ax[3].set_ylabel('AIRMASS')

# 5. TELFOCUS
ax[4].plot(df['UTC'], df['TELFOCUS'], color='tab:purple', label='Focus')
ax[4].set_ylabel('TEL. FOCUS')


# X axis configuration
locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
formatter = mdates.ConciseDateFormatter(locator)
ax[4].xaxis.set_major_locator(locator)
ax[4].xaxis.set_major_formatter(formatter)

for i in range(5):
    # Only main lines shown
    ax[i].grid(True, which='major', linestyle='--', alpha=0.4, color='gray')
    # Here i can eliminate some lines 
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    # Legends
    ax[i].legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

fig.suptitle('Instrumental Conditions', fontsize=14, fontweight='bold')
ax[4].set_xlabel('Time (UTC)')


plt.savefig("/home/acanamero-ext/practicas/results/roi/fwhm_comparison_plot_roi.pdf", dpi=300, bbox_inches='tight')
plt.show()
"""

0.336 arcmin/pixel

"""