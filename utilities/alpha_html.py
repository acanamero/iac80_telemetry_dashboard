#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:17:56 2026
@author: acanamero-ext
"""
import io
import csv
import os
import time
import warnings
import queue
import threading
import numpy as np
from datetime import datetime, timedelta
import glob

from scipy.spatial import cKDTree

# Astropy & Photutils
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyUserWarning
from astropy.units import Quantity, UnitsError
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry
from astropy.modeling.core import Fittable1DModel, Fittable2DModel
from astropy.modeling.parameters import InputParameterError, Parameter
from astropy.modeling.utils import ellipse_extent
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.coordinates import match_coordinates_sky


# Watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Flask for the Web Server
from flask import Flask, jsonify, render_template_string, Response, request

base_coord = None
base_phot = None
guiding_error_x = None
guiding_error_y = None

# ---------------- Custom Moffat Model ---------------------
class Moffat2Dell(Fittable2DModel):
    """
    Two dimensional Moffat elliptical model.
    """

    flux = Parameter(default=1, description="Scaling factor (peak value) of the model")
    x_0 = Parameter(default=0, description="X position of the maximum of the Moffat model")
    y_0 = Parameter(default=0, description="Y position of the maximum of the Moffat model")
    gammax = Parameter(default=1, description="Core width of the Moffat model (x-axis)")
    gammay = Parameter(default=1, description="Core width of the Moffat model (y-axis)")
    phi = Parameter(default=0, description="Azimuthal angle of the largest FWHM (clockwise)")
    alpha = Parameter(default=1, description="Power index of the Moffat model")

    @property
    def fwhmx(self):
        return 2.0 * np.abs(self.gammax) * np.sqrt(2.0 ** (1.0 / self.alpha) - 1.0)

    @property
    def fwhmy(self):
        return 2.0 * np.abs(self.gammay) * np.sqrt(2.0 ** (1.0 / self.alpha) - 1.0)

    @staticmethod
    def evaluate(x, y, flux, x_0, y_0, gammax, gammay, phi, alpha):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        A = (cos_phi/gammax)**2 + (sin_phi/gammay)**2
        B = (sin_phi/gammax)**2 + (cos_phi/gammay)**2
        C = 2*cos_phi*sin_phi*(1/gammax**2 - 1/gammay**2)
        rr_gg = A*(x-x_0)**2 + B*(y-y_0)**2 + C*(x-x_0)*(y-y_0)
        return flux * (1 + rr_gg) ** (-alpha)

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, gammax, gammay, phi, alpha):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        A = (cos_phi/gammax)**2 + (sin_phi/gammay)**2
        B = (sin_phi/gammax)**2 + (cos_phi/gammay)**2
        C = 2*cos_phi*sin_phi*(1/gammax**2 - 1/gammay**2)
        rr_gg = A*(x-x_0)**2 + B*(y-y_0)**2 + C*(x-x_0)*(y-y_0)
        d_amp = (1 + rr_gg) ** (-alpha)
        
        # Derivatives (replacing amplitude with flux)
        d_flux = d_amp
        d_alpha = -flux * d_amp * np.log(1 + rr_gg)
        d_x_0 =  flux * alpha * d_amp / (1 + rr_gg) * (2*A*(x - x_0) + C*(y - y_0))
        d_y_0 =  flux * alpha * d_amp / (1 + rr_gg) * (2*B*(y - y_0) + C*(x - x_0))
        d_gammax = flux * alpha * d_amp / (1 + rr_gg) * 2 / gammax**3 * \
            ((x-x_0)**2*cos_phi**2 + (y-y_0)**2*sin_phi**2 + 2*(x-x_0)*(y-y_0)*sin_phi*cos_phi)
        d_gammay = flux * alpha * d_amp / (1 + rr_gg) * 2 / gammay**3 * \
            ((x-x_0)**2*sin_phi**2 + (y-y_0)**2*cos_phi**2 - 2*(x-x_0)*(y-y_0)*sin_phi*cos_phi)
        d_phi = -flux * alpha * d_amp / (1 + rr_gg) * 2 * \
            ((x-x_0)**2*cos_phi*sin_phi*(-1/gammax**2+1/gammay**2) + \
            (y-y_0)**2*cos_phi*sin_phi*(1/gammax**2-1/gammay**2) +\
            (x-x_0)*(y-y_0)*(cos_phi**2-sin_phi**2)*(1/gammax**2-1/gammay**2))
        return [d_flux, d_x_0, d_y_0, d_gammax, d_gammay, d_phi, d_alpha]

    @property
    def input_units(self):
        if self.x_0.input_unit is None:
            return None
        else:
            return {
                self.inputs[0]: self.x_0.input_unit, 
                self.inputs[1]: self.y_0.input_unit
            }

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {
            "x_0": inputs_unit[self.inputs[0]],
            "y_0": inputs_unit[self.inputs[0]],
            "gammax": inputs_unit[self.inputs[0]],
            "gammay": inputs_unit[self.inputs[0]],
            "flux": outputs_unit[self.outputs[0]],
        }


# --- GLOBAL STATE ---
history_times = []
history_indexes = []
history_fwhm_x = []
history_err_x = []
history_err_y = []
history_fwhm_y = []
history_colors = []
history_alt = []
history_az = []

# Global State for the 5 extra plots
history_airmass = []
history_telfocus = []
history_humidity = []
history_temperature = []
history_sources = []

# Variables for CSV Export
history_filters = []
history_ra = []
history_dec = []
history_gamma_x = []
history_gamma_y = []
history_alpha = []
history_phi = []

# Guiding error
history_dx = []
history_dy = []
history_base_changes = [] # Stores UTC times of field changes

data_lock = threading.Lock() # avoids index problems

# Location of the IAC80 (Teide Observatory, Tenerife)
IAC_LOCATION = EarthLocation(lat=28.299667 * u.deg, lon=-16.511027 * u.deg,
                             height=2381.25 * u.m)

latest_observer = "Observer"

FILTER_COLORS = {
    # Johnson-Cousins
    'U': 'purple', 'B': 'blue', 'V': 'green', 'R': 'red', 'I': 'darkred',
    # SDSS
    'SDSSu': '#9c27b0', 'SDSSg': '#4caf50', 'SDSSr': '#f44336', 'SDSSi': '#b71c1c', 'SDSSz': '#4a148c',
    # Failsafe
    'UNKNOWN_FILTER': 'black'
}
fits_queue = queue.Queue() 

# --- FLASK APP SETUP ---
app = Flask(__name__)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Telemetry Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    
    <style>
        /* Global Reset and Font */
        body { 
            font-family: 'Inter', sans-serif; 
            background-color: #f0f2f5; 
            margin: 0; 
            padding: 0; 
            color: #333;
        }
        
        /* The Top Navigation Bar */
        .navbar {
            background-color: #1e293b;
            color: white;
            padding: 15px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .logo {
            height: 45px; 
            object-fit: contain;
        }

        .status-badge {
            background-color: #10b981;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background-color 0.3s;
        }
        
        .status-badge.paused {
            background-color: #64748b;
        }
        
        .dot {
            height: 8px;
            width: 8px;
            background-color: white;
            border-radius: 50%;
            display: inline-block;
            animation: blink 1.5s infinite;
        }
        
        .status-badge.paused .dot {
            animation: none;
            background-color: #cbd5e1;
        }

        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }

        .dashboard-container { 
            max-width: 1600px; 
            margin: 30px auto; 
            padding: 0 20px; 
        }

        .grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .card { 
            background: white; 
            border-radius: 12px; 
            padding: 20px; 
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); 
        }
        
        .span-2 { grid-column: span 2; }
        
        .plot-main { height: 500px; width: 100%; }
        .plot-sub { height: 350px; width: 100%; }
        
        .btn-download, .btn-upload, .btn-resume, .btn-settings {
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        .btn-download { background-color: #3b82f6; }
        .btn-download:hover { background-color: #2563eb; }
        
        .btn-upload { background-color: #10b981; }
        .btn-upload:hover { background-color: #059669; }
        
        .btn-resume {
            background-color: #f59e0b;
            display: none;
        }
        .btn-resume:hover { background-color: #d97706; }

        .navbar-controls {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        /* Settings Dropdown Style */
        .settings-dropdown {
            position: relative;
            display: inline-block;
        }

        .btn-settings {
            background-color: #475569;
        }

        .dropdown-content {
            display: none;
            position: absolute;
            right: 0;
            background-color: #ffffff;
            min-width: 200px;
            box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
            z-index: 1000;
            border-radius: 8px;
            padding: 12px;
            color: #333;
        }

        .settings-dropdown:hover .dropdown-content {
            display: block;
        }

        .setting-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 14px;
        }

        .setting-item:last-child { margin-bottom: 0; }

        input[type="checkbox"] {
            cursor: pointer;
            width: 16px;
            height: 16px;
        }
        
        .footer-contact {
            position: fixed;
            bottom: 10px;
            right: 15px;
            background: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 11px;
            color: #475569;
            z-index: 1001;
            border: 1px solid #e2e8f0;
            max-width: 250px;
        }
        .footer-contact a { color: #3b82f6; text-decoration: none; }

    </style>
</head>
<body>
    
    <div class="footer-contact">
        <p>Developed by <strong>Alejandro Cañamero Herrera</strong></p>
        <p>If you have any enquiries or find bugs regarding the webpage contact: 
            <a href="mailto:alejandrocahe9@gmail.com">alejandrocahe9@gmail.com</a>
        </p>
    </div>
    <div class="navbar">
        <div class="navbar-brand">
            <img src="/static/logo.png" alt="Institution Logo" class="logo" onerror="this.src='https://via.placeholder.com/150x45?text=YOUR+LOGO'">
            <div>
                <h2 style="margin: 0; font-size: 1.5rem;">IAC80 Telemetry Dashboard</h2>
                <div id="obs-night" style="font-size: 1.1rem; font-weight: 600; color: #10b981; margin-top: 4px;">
                    Night of Observation: --
                </div>
                <div id="live-coords" style="font-size: 0.9rem; color: #94a3b8; font-family: 'Courier New', Courier, monospace; margin-top: 4px;">
                    Target: (RA, DEC) = (--, --)
                </div>
            </div>
        </div>

        <div class="navbar-controls">
            <button id="btn-resume-live" class="btn-resume" onclick="resumeLive()">Resume Live Monitoring</button>
            
            <div class="settings-dropdown">
                <button class="btn-settings">&#x2699;&#xFE0E; Settings</button>
                
                <div class="dropdown-content">
                    <div class="setting-item">
                        <label for="toggle-lines">Show Lines</label>
                        <input type="checkbox" id="toggle-lines" checked onchange="toggleSettings()">
                    </div>
                    <div class="setting-item">
                        <label for="toggle-errors">Show Error Bars</label>
                        <input type="checkbox" id="toggle-errors" checked onchange="toggleSettings()">
                    </div>
                </div>
            </div>

            <label for="csv-upload" class="btn-upload">Upload CSV</label>
            <input type="file" id="csv-upload" accept=".csv" style="display: none;" onchange="handleCSVUpload(event)">
            
            <a href="/download_csv" class="btn-download">Download CSV</a>
            <div id="status-indicator" class="status-badge">
                <span class="dot"></span> <span id="status-text">Live Monitoring</span>
            </div>
        </div>
    </div>

    <div class="dashboard-container">
        <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
            <h2 style="margin: 0; color: #1e293b;">Welcome, <span id="observer-name">Observer</span>!</h2>
            <p style="margin: 8px 0 0 0; color: #64748b; font-size: 1.1rem; font-style: italic;">
                "Good luck and clear skies! May the seeing be ever in your favor."
            </p>
        </div>
        
        <div class="grid-container">
            <div class="card span-2"><div id="plot-fwhm" class="plot-main"></div></div>
            <div class="card"><div id="plot-phi" class="plot-sub"></div></div>
            <div class="card"><div id="plot-sources" class="plot-sub"></div></div>
            <div class="card span-2"><div id="plot-dxdy-time" class="plot-main"></div></div>
            <div class="span-2" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                <div class="card"><div id="plot-guiding" class="plot-sub"></div></div>
                <div class="card"><div id="plot-error-hist" class="plot-sub"></div></div>
            </div>
            <div class="card"><div id="plot-sky" class="plot-sub"></div></div>
            <div class="card"><div id="plot-airmass" class="plot-sub"></div></div>
            <div class="card"><div id="plot-temperature" class="plot-sub"></div></div>
            <div class="card"><div id="plot-humidity" class="plot-sub"></div></div>
            
        </div>
    </div>

    <script>
        let liveInterval;
        let isLive = true;
        let lastData = null;

        // Settings State
        let showLines = true;
        let showErrorBars = true;

        function toggleSettings() {
            showLines = document.getElementById('toggle-lines').checked;
            showErrorBars = document.getElementById('toggle-errors').checked;
            if (lastData) renderPlotsData(lastData);
        }

        function handleCSVUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            fetch('/upload_csv', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    isLive = false;
                    clearInterval(liveInterval);
                    document.getElementById('status-indicator').classList.add('paused');
                    document.getElementById('status-text').innerText = 'Historical Data';
                    document.getElementById('btn-resume-live').style.display = 'inline-block';
                    renderPlotsData(data.csv_data);
                } else { alert('Error loading CSV: ' + data.error); }
                event.target.value = '';
            });
        }
        
        function resumeLive() {
            isLive = true;
            document.getElementById('status-indicator').classList.remove('paused');
            document.getElementById('status-text').innerText = 'Live Monitoring';
            document.getElementById('btn-resume-live').style.display = 'none';
            fetchLiveData();
            liveInterval = setInterval(fetchLiveData, 20000);
        }

        function createLayout(title, yAxisTitle) {
            return {
                title: { text: title, font: { size: 18 } },
                xaxis: { title: 'Time (UTC)', gridcolor: '#e2e8f0' },
                yaxis: { title: yAxisTitle, gridcolor: '#e2e8f0' },
                hovermode: 'closest',
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                margin: { t: 50, r: 20, b: 50, l: 60 },
                font: { family: 'Inter, sans-serif' },
                uirevision: 'true'
            };
        }

        function renderPlotsData(data) {
            lastData = data;
            if(data.times.length === 0) return;
            
            document.getElementById('live-coords').innerText = `Target: (RA, DEC) = (${data.latest_ra}, ${data.latest_dec})`; 
            document.getElementById('obs-night').innerText = `Night of Observation: ${data.obs_night}`;
            document.getElementById('observer-name').innerText = data.observer;
            
            const currentMode = showLines ? 'lines+markers' : 'markers';
            const filterMap = { 'U': 'purple', 'B': 'blue', 'V': 'green', 'R': 'red', 'I': 'darkred', 
                               'SDSSu': '#9c27b0', 'SDSSg': '#4caf50', 'SDSSr': '#f44336', 'SDSSi': '#b71c1c', 'SDSSz': '#4a148c', 
                               'UNKNOWN_FILTER': 'black' 
                               };
            
            // --- Plot 1: FWHM & Telescope Focus ---
            var traceX = { 
                x: data.times, y: data.fwhm_x, mode: currentMode, name: 'FWHM X',
                line: { color: '#3b82f6', width: 2 }, marker: { size: 8, color: data.colors },
                error_y: { type: 'data', array: data.err_x, visible: showErrorBars, color: '#3b82f6', thickness: 1, width: 0},
                text: data.indexes, customdata: data.err_x,
                hovertemplate: '<b>Idx:</b> %{text}<br><b>FWHM X:</b> %{y:.4f} ± %{customdata:.4f}<extra></extra>' 
            };
            
            var traceY = { 
                x: data.times, y: data.fwhm_y, mode: currentMode, name: 'FWHM Y',
                line: { color: '#ef4444', width: 2 }, marker: { symbol: 'triangle-up', size: 8, color: data.colors },
                error_y: { type: 'data', array: data.err_y, visible: showErrorBars, color: '#ef4444', thickness: 1, width: 0},
                text: data.indexes, customdata: data.err_y,
                hovertemplate: '<b>Idx:</b> %{text}<br><b>FWHM Y:</b> %{y:.4f} ± %{customdata:.4f}<extra></extra>' 
            };
            
            var traceFocus = { 
                x: data.times, y: data.telfocus, mode: 'lines+markers', name: 'Tel Focus',
                line: { color: '#f59e0b', dash: 'dot', width: 2 }, marker: { size: 6 }, 
                text: data.indexes, yaxis: 'y2', 
                hovertemplate: '<b>Idx:</b> %{text}<br><b>Focus:</b> %{y:d}<extra></extra>' 
            };

            var filterLegendTraces = Object.keys(filterMap).map(filter => ({
                x: [null], y: [null], mode: 'markers', name: `Filter ${filter}`,
                marker: { color: filterMap[filter], size: 10 }, showlegend: true
            }));

            var layoutFWHM = createLayout('PSF FWHM & Telescope Focus', 'Median FWHM (arcsec)');
            layoutFWHM.margin.r = 160;
            layoutFWHM.legend = { x: 1.08, xanchor: 'left', y: 1, yanchor: 'top' };
            layoutFWHM.yaxis2 = { title: 'Focus', overlaying: 'y', side: 'right', showgrid: false, color: '#f59e0b' };

            Plotly.react('plot-fwhm', [traceX, traceY, traceFocus, ...filterLegendTraces], layoutFWHM, { responsive: true });
            
            // --- NEW Plot 2: dx and dy over Time ---
            var traceDxTime = { 
                x: data.times, y: data.dx, mode: currentMode, name: 'dx',
                line: { color: '#3b82f6', width: 2 }, marker: { size: 6 },
                text: data.indexes,
                hovertemplate: '<b>Idx:</b> %{text}<br><b>dx:</b> %{y:.3f} arcsec<extra></extra>' 
            };
            
            var traceDyTime = { 
                x: data.times, y: data.dy, mode: currentMode, name: 'dy',
                line: { color: '#ef4444', width: 2 }, marker: { size: 6 },
                text: data.indexes,
                hovertemplate: '<b>Idx:</b> %{text}<br><b>dy:</b> %{y:.3f} arcsec<extra></extra>' 
            };

            var layoutDxDyTime = createLayout('Guiding Offsets (dx & dy) over Time', 'Offset (arcsec)');
            
            // Add vertical shapes based on data.base_change_times from the backend
            layoutDxDyTime.shapes = [];
            if (data.base_change_times && data.base_change_times.length > 0) {
                data.base_change_times.forEach(function(changeTime) {
                    layoutDxDyTime.shapes.push({
                        type: 'line',
                        x0: changeTime, x1: changeTime,
                        y0: 0, y1: 1, // 0 to 1 scales relative to the plotting area height
                        yref: 'paper',
                        line: { color: 'grey', width: 1.5, dash: 'solid' },
                        opacity: 0.6
                    });
                });
            }

            Plotly.react('plot-dxdy-time', [traceDxTime, traceDyTime], layoutDxDyTime, { responsive: true });


            // --- Standard Subplots ---
            const standardPlot = (id, yData, color, title, label) => {
                Plotly.react(id, [{
                    x: data.times, y: yData, mode: currentMode, line: { color: color }, marker: { size: 6 },
                    text: data.indexes, hovertemplate: `<b>Idx:</b> %{text}<br><b>${label}:</b> %{y:.2f}<extra></extra>`
                }], createLayout(title, label), { responsive: true });
            };

            standardPlot('plot-phi', data.phi, '#8b5cf6', 'PSF Angle (Phi)', 'Median Phi (°)');
            standardPlot('plot-humidity', data.humidity, '#0ea5e9', 'Humidity', 'Humidity (%)');
            standardPlot('plot-temperature', data.temperature, '#f43f5e', 'Temperature', 'Temp (°C)');
            

            // --- Plot: Sources (Bar) ---
            Plotly.react('plot-sources', [{ x: data.times, y: data.sources, type: 'bar', marker: { color: '#10b981' }, text: data.indexes, textposition: 'none', hovertemplate: '<b>Idx:</b> %{text}<br><b>Sources:</b> %{y}<extra></extra>' }], createLayout('Detected Sources', 'Count'), { responsive: true });

            // --- Plot: Airmass ---
            var layoutAir = createLayout('Airmass', 'Airmass');
            layoutAir.margin = { t: 50, r: 60, b: 50, l: 60 };
            layoutAir.yaxis.autorange = 'reversed';
            layoutAir.yaxis2 = { title: 'Altitude (°)', overlaying: 'y', side: 'right', showgrid: false, color: '#64748b' };
            Plotly.react('plot-airmass', [
                { x: data.times, y: data.airmass, mode: currentMode, line: { color: '#8b5cf6' }, marker: { size: 6 }, text: data.indexes, customdata: data.altitude, showlegend: false, hovertemplate: '<b>Idx:</b> %{text}<br><b>Airmass:</b> %{y:.4f}<br><b>Altitude:</b> %{customdata:.1f}°<extra></extra>' },
                { x: data.times, y: data.altitude, yaxis: 'y2', mode: 'none', showlegend: false, hoverinfo: 'skip' }
            ], layoutAir, { responsive: true });
            
            // --- Plot: Guiding Error (dx vs dy scatter/bullseye) ---
            const latestTime = new Date(data.times[data.times.length - 1]).getTime();
            const ages_in_minutes = data.times.map(t => {
                const pointTime = new Date(t).getTime();
                return -(latestTime - pointTime) / 60000; 
            });

            var traceGuiding = { 
                x: data.dx, y: data.dy, mode: 'markers', 
                marker: { 
                    size: 8, 
                    color: ages_in_minutes,      
                    colorscale: 'Viridis',       
                    reversescale: false,          
                    cauto: false,
                    cmin: -60,                     
                    cmax: 0,                    
                    colorbar: { title: { text: 'Mins Ago', side: 'right' }, thickness: 15, len: 0.9 },
                    opacity: 0.8 
                }, 
                text: data.indexes,
                customdata: ages_in_minutes,     
                hovertemplate: '<b>Idx:</b> %{text}<br><b>dx:</b> %{x:.3f}<br><b>dy:</b> %{y:.3f}<br><b>Age:</b> %{customdata:.1f} mins<extra></extra>' 
            };
            
            var layoutGuiding = {
                title: { text: 'Guiding Error', font: { size: 18 } },
                xaxis: { title: 'dx (arcsec)', gridcolor: '#e2e8f0', zeroline: true, zerolinewidth: 2, zerolinecolor: '#94a3b8' },
                yaxis: { title: 'dy (arcsec)', gridcolor: '#e2e8f0', zeroline: true, zerolinewidth: 2, zerolinecolor: '#94a3b8', scaleanchor: 'x', scaleratio: 1 },
                hovermode: 'closest',
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                margin: { t: 50, r: 20, b: 50, l: 60 },
                font: { family: 'Inter, sans-serif' },
                uirevision: 'true'
            };
            Plotly.react('plot-guiding', [traceGuiding], layoutGuiding, { responsive: true });
            
            // PLOT- HISTOGRAMS DX AND DY
            var traceDxHist = { x: data.dx, type: 'histogram', marker: { color: '#3b82f6' }, opacity: 0.7, name: 'dx' };
            var traceDyHist = { x: data.dy, type: 'histogram', marker: { color: '#ef4444' }, opacity: 0.7, name: 'dy' };

            var layoutErrorHist = createLayout('Guiding Error Distribution', 'Count');
            layoutErrorHist.xaxis.title = 'Error (arcsec)';
            layoutErrorHist.barmode = 'overlay'; 

            Plotly.react('plot-error-hist', [traceDxHist, traceDyHist], layoutErrorHist, { responsive: true });

            // --- Plot: Sky Pointing ---
            if (data.latest_alt !== null && data.latest_az !== null) {
                var traceSky = {
                    type: 'scatterpolar', mode: 'markers', r: [90 - data.latest_alt], theta: [data.latest_az],
                    marker: { color: '#ef4444', size: 14, symbol: 'circle', line: { color: 'white', width: 2 } },
                    hovertemplate: '<b>Az:</b> %{theta:.1f}°<br><b>Alt:</b> ' + data.latest_alt.toFixed(1) + '°<extra></extra>'
                };
                var layoutSky = {
                    title: { text: 'Current Telescope Pointing', font: { size: 18 } },
                    polar: {
                        radialaxis: { range: [0, 90], visible: true, tickmode: 'array', tickvals: [0, 30, 60, 90], ticktext: ['Zenith', '60°', '30°', 'Horizon'], gridcolor: '#e2e8f0', angle: 45 },
                        angularaxis: { direction: "counterclockwise", rotation: 90, tickvals: [0, 90, 180, 270], ticktext: ['N', 'E', 'S', 'W'], gridcolor: '#e2e8f0' },
                        bgcolor: '#f8fafc'
                    },
                    showlegend: false, margin: { t: 50, r: 40, b: 40, l: 40 }, font: { family: 'Inter, sans-serif' },
                    uirevision: 'true'
                };
                Plotly.react('plot-sky', [traceSky], layoutSky, { responsive: true });
            }
        }

        function fetchLiveData() {
            if (!isLive) return;
            fetch('/data').then(r => r.json()).then(d => renderPlotsData(d)).catch(err => console.error(err));
        }
        liveInterval = setInterval(fetchLiveData, 20000); 
        fetchLiveData();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/data')
def get_data():
    with data_lock:
        arcsec_pixel = 0.336
        
        if len(history_times) > 0:
            dt_objects = [datetime.fromisoformat(t) for t in history_times]
            sorted_indices = np.argsort(dt_objects)
            
            # HELPER FUNCTION: Safely forces values into floats
            def get_sorted(lst, multiplier=1.0):
                result = []
                for i in sorted_indices:
                    val = lst[i]
                    if val is not None:
                        try:
                            # Convert to float before multiplying!
                            result.append(float(val) * multiplier)
                        except (ValueError, TypeError):
                            # If it's a weird string like "N/A" or "UNKNOWN", treat as missing data
                            result.append(None)
                    else:
                        result.append(None)
                return result

            # FWHM Variables
            t_sorted = [history_times[i] for i in sorted_indices]
            idx_sorted = [history_indexes[i] for i in sorted_indices]
            c_sorted = [history_colors[i] for i in sorted_indices]
            x_sorted = get_sorted(history_fwhm_x, arcsec_pixel)
            err_x_sorted = get_sorted(history_err_x, arcsec_pixel)
            y_sorted = get_sorted(history_fwhm_y, arcsec_pixel)
            err_y_sorted = get_sorted(history_err_y, arcsec_pixel)
            
            # Telemetry Variables
            air_sorted = get_sorted(history_airmass)
            foc_sorted = get_sorted(history_telfocus)
            hum_sorted = get_sorted(history_humidity)
            tem_sorted = get_sorted(history_temperature)
            src_sorted = get_sorted(history_sources)
            rad_to_deg = 180.0 / np.pi
            phi_sorted = get_sorted(history_phi, rad_to_deg)
            dx_sorted = get_sorted(history_dx)
            dy_sorted = get_sorted(history_dy)
            
            
            # --- Get the absolute latest coordinates ---
            latest_idx = sorted_indices[-1] # The last item chronologically
            latest_ra = history_ra[latest_idx]
            latest_dec = history_dec[latest_idx]
            
            # Extract just the YYYY-MM-DD from the latest datetime object
            latest_dt = dt_objects[-1]
            obs_night = latest_dt.strftime('%Y-%m-%d')
            
            # --- Grab latest Alt/Az ---
            latest_alt = history_alt[latest_idx]
            latest_az = history_az[latest_idx]
            
            alt_sorted = get_sorted(history_alt)
            

        else:
            t_sorted, idx_sorted, x_sorted, err_x_sorted, y_sorted, err_y_sorted, c_sorted = [], [], [], [], [], [], []
            air_sorted, foc_sorted, hum_sorted, tem_sorted, src_sorted, phi_sorted, alt_sorted, dx_sorted, dy_sorted = [], [], [], [], [], [], [], [], []
            obs_night = '--'
            latest_ra, latest_dec = '--', '--'
            latest_alt, latest_az = None, None
            

        return jsonify({
            'times': t_sorted,
            'indexes': idx_sorted,
            'fwhm_x': x_sorted, 'err_x': err_x_sorted,
            'fwhm_y': y_sorted, 'err_y': err_y_sorted,
            'colors': c_sorted,
            'airmass': air_sorted,
            'telfocus': foc_sorted,
            'humidity': hum_sorted,
            'temperature': tem_sorted,
            'sources': src_sorted,
            'phi': phi_sorted,
            'latest_ra': latest_ra,
            'latest_dec': latest_dec,
            'obs_night': obs_night,
            'latest_alt': latest_alt,
            'altitude': alt_sorted,
            'latest_az': latest_az,
            'observer': latest_observer,
            'dx': dx_sorted,
            'dy': dy_sorted,
            'base_change_times': history_base_changes
        })


def get_observing_night_dir(base_dir="/home/acanamero-ext/practicas/data"):
    #hefestoe/data_raw
    """
    Calculates the observational directory based on the current time.
    If the time is between midnight and 8:00 AM, it assigns it to the previous calendar day.
    """
    now = datetime.now()
    
    # The rollover threshold is 8:00 AM (08:00)
    if now.hour < 8:
        # Subtract 1 day if we are past midnight but before 8 AM
        obs_date = now - timedelta(days=1)
    else:
        obs_date = now

    # Hardcoded English month abbreviations to prevent OS locale issues 
    # (e.g., ensuring we get 'Apr' and not 'Abr' on a Spanish system)
    months = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    # Format components
    year_str = obs_date.strftime("%y")     # 2-digit year (e.g., '26')
    month_str = months[obs_date.month]     # 3-letter month (e.g., 'Apr')
    day_str = obs_date.strftime("%d")      # 2-digit day with leading zero (e.g., '03')

    # Construct the final directory name (e.g., 26Apr03)
    dir_name = f"{year_str}{month_str}{day_str}"
    
    return os.path.join(base_dir, dir_name)


@app.route('/download_csv')
def download_csv():
    # Create an in-memory text buffer
    si = io.StringIO()
    writer = csv.writer(si)
    
    # Standardized 20-column header
    writer.writerow([
        'INDEX', 'UTC', 'AIRMASS', 'FILTER', 'TELFOCUS', 
        'HUMIDITY', 'TEMPERATURE', 'RA', 'DEC', 'ALT', 'AZ',
        'GAMMA_X', 'FWHM_X', 'ERR_X', 'GAMMA_Y', 'FWHM_Y', 'ERR_Y',
        'ALPHA', 'PHI', 'NSOURCES', 'DX', 'DY'
    ])
    
    if len(history_times) > 0:
        # Ensure chronological order
        dt_objects = [datetime.fromisoformat(t) for t in history_times]
        sorted_indices = np.argsort(dt_objects)
        
        with data_lock:
            for i in sorted_indices:
                writer.writerow([
                    history_indexes[i],                 # INDEX
                    history_times[i],                   # UTC
                    history_airmass[i],                 # AIRMASS
                    history_filters[i],                 # FILTER
                    history_telfocus[i],                # TELFOCUS
                    history_humidity[i],                # HUMIDITY
                    history_temperature[i],             # TEMPERATURE
                    history_ra[i],                      # RA
                    history_dec[i],                     # DEC
                    history_alt[i],                     # ALT
                    history_az[i],                      # AZ
                    history_gamma_x[i],                 # GAMMA_X
                    history_fwhm_x[i],                  # FWHM_X
                    history_err_x[i],                   # ERR_X
                    history_gamma_y[i],                 # GAMMA_Y
                    history_fwhm_y[i],                  # FWHM_Y
                    history_err_y[i],                   # ERR_Y
                    history_alpha[i],                   # ALPHA
                    history_phi[i],                     # PHI
                    history_sources[i],                 # NSOURCES
                    history_dx[i],                      # DX GUIDING
                    history_dy[i]                       # DY GUIDING
                ])
            
    # Return the text buffer as a downloadable CSV file
    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=telemetry_{get_observing_night_dir()[-7:]}.csv"}
    )

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify(success=False, error="No file part in request")
        
    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, error="No file selected")
        
    try:
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_reader = csv.DictReader(stream)
        rows = list(csv_reader)
        
        arcsec_pixel = 0.336
        rad_to_deg = 180.0 / np.pi
        
        # Sort rows chronologically by UTC, safely handling missing or None values
        # We default to an empty string '' so it just pushes bad rows to the top/bottom 
        # without crashing the sorted() function.
        rows.sort(key=lambda x: str(x.get('UTC') or ''))
        
        # Prepare the isolated dictionary for the frontend
        csv_data = {
            'times': [], 'indexes': [], 'fwhm_x': [], 'err_x': [], 'fwhm_y': [], 'err_y': [], 'colors': [],
            'airmass': [], 'telfocus': [], 'humidity': [], 'temperature': [], 'sources': [], 'phi': [],
            'latest_ra': '--', 'latest_dec': '--', 'obs_night': '--', 'latest_alt': None, 'latest_az': None,
            'observer': '(Loaded Data)', 'dx': [], 'dy': []
        }
        
        def parse_val(val, multiplier=1.0):
            if val in ('', 'None', 'UNKNOWN', '--', None): return None
            try: return float(val) * multiplier
            except ValueError: return None

        for row in rows:
            csv_data['times'].append(row.get('UTC'))
            csv_data['indexes'].append(row.get('INDEX'))
            
            csv_data['airmass'].append(parse_val(row.get('AIRMASS')))
            filt = row.get('FILTER', 'UNKNOWN_FILTER')
            csv_data['colors'].append(FILTER_COLORS.get(filt.upper(), 'black'))
            csv_data['telfocus'].append(parse_val(row.get('TELFOCUS')))
            csv_data['humidity'].append(parse_val(row.get('HUMIDITY')))
            csv_data['temperature'].append(parse_val(row.get('TEMPERATURE')))
            
            csv_data['fwhm_x'].append(parse_val(row.get('FWHM_X'), arcsec_pixel))
            csv_data['err_x'].append(parse_val(row.get('ERR_X'), arcsec_pixel) or 0.0)
            csv_data['fwhm_y'].append(parse_val(row.get('FWHM_Y'), arcsec_pixel))
            csv_data['err_y'].append(parse_val(row.get('ERR_Y'), arcsec_pixel) or 0.0)
            
            csv_data['phi'].append(parse_val(row.get('PHI'), rad_to_deg))
            csv_data['sources'].append(parse_val(row.get('NSOURCES')))
            
            csv_data['latest_ra'] = row.get('RA', '--')
            csv_data['latest_dec'] = row.get('DEC', '--')
            csv_data['dx'].append(parse_val(row.get('DX')))
            csv_data['dy'].append(parse_val(row.get('DY')))

        # Determine the local projection for the final row in the CSV
        if rows:
            last_row = rows[-1]
            # Safely grab the UTC using .get() instead of strict dictionary indexing
            last_utc = last_row.get('UTC')
            
            if last_utc and last_utc not in ('UNKNOWN', '--', ''):
                try:
                    obs_dt = datetime.fromisoformat(last_utc)
                    csv_data['obs_night'] = obs_dt.strftime('%Y-%m-%d')
                    
                    ra_str, dec_str = last_row.get('RA'), last_row.get('DEC')
                    if ra_str not in ('UNKNOWN', '--', '', None) and dec_str not in ('UNKNOWN', '--', '', None):
                        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                        altaz = coord.transform_to(AltAz(obstime=Time(obs_dt), location=IAC_LOCATION))
                        csv_data['latest_alt'] = altaz.alt.degree
                        csv_data['latest_az'] = altaz.az.degree
                except Exception:
                    pass # Fail silently if coordinates or time in the CSV are malformed
            
        return jsonify(success=True, csv_data=csv_data)
        
    except Exception as e:
        return jsonify(success=False, error=str(e))

# --- FITS PROCESSING LOGIC ---
def get_camera_settings(header):
    mode_str = (header.get('READOUTM', '')).upper()
    settings = {"mode": "Unknown", "speed": None, "saturation": 40000, "recommended_flat": None}
    if "100KHZ" in mode_str:
        settings.update({"mode": "Mode 3", "speed": 100, "saturation": 60000})
    elif "855KHZ" in mode_str:
        settings.update({"mode": "Mode 4", "speed": 855, "saturation": 12000})
    elif "709KHZ" in mode_str:
        settings.update({"mode": "Mode 2", "speed": 709, "saturation": 22000})
    elif "344KHZ" in mode_str:
        if "CCD ATTN0" in mode_str:
            settings.update({"mode": "Mode 0", "speed": 344, "saturation": 56000})
        else:
            settings.update({"mode": "Mode 1", "speed": 344, "saturation": 40000})
    return settings

def process_single_fits(fits_file):
    global history_times, history_indexes, history_fwhm_x, history_fwhm_y, history_err_x, history_err_y, history_colors
    global history_airmass, history_telfocus, history_humidity, history_temperature, history_sources
    global latest_observer
    global base_coord, base_phot
    global guiding_error_x, guiding_error_y
    
    init = time.perf_counter()
    fwhm_min = 2.0
    index_csv = os.path.basename(fits_file)[-18:-5]
    
    try:
        file_data = fits.open(fits_file)
        image_data = file_data[0].data.astype(float)
        header = file_data[0].header
        
        # Grab the type of .fits:
        # We use .strip().upper() to protect against trailing spaces (e.g. 'OBJECT   ')
        fits_type = str(header.get('IMAGETYP', 'UNKNOWN')).strip().upper()
        
        # --- FAST-SKIP LOGIC ---
        if fits_type != 'OBJECT':
            print(f"\n--- SKIPPED: {os.path.basename(fits_file)} (IMAGETYP is '{fits_type}') ---")
            return  # Exits the function immediately, moving to the next file!
        # ---------------------------
        
        obs_time = header.get('DATE-OBS', 'Unknown_Time')
        filter_name = header.get('INSFILTE', 'Unknown_Filter')
        settings = get_camera_settings(header)
        latest_observer = header.get('OBSERVER', 'Astronomer')
        
        print(f"\n--- NEW FILE DETECTED: {os.path.basename(fits_file)} ---")
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
            print(f">>> Photometry Done. Sources: {len(phot)}")
            
            # Compute outlier boundaries
            filtered_fwhm = np.percentile(phot["fwhm"], [5, 95])
            
            # Filter out extreme outliers first
            phot = phot[(phot["fwhm"] >= filtered_fwhm[0]) & (phot["fwhm"] <= filtered_fwhm[1])]
            
            # Compute the error without the extreme outliers
            perc_x = np.percentile(phot["fwhm_x"], 75) - np.percentile(phot["fwhm_x"], 25)
            perc_y = np.percentile(phot["fwhm_y"], 75) - np.percentile(phot["fwhm_y"], 25)
            
            
            try:
                obs_dt = datetime.fromisoformat(obs_time)
            except ValueError:
                obs_dt = datetime.now()
                
            # Grab RA and DEC from the header 
            header_ra = header.get('RA', header.get('OBJRA', 'UNKNOWN'))
            header_dec = header.get('DEC', header.get('OBJDEC', 'UNKNOWN'))
            
            
            # --- Convert to Alt/Az for the All-Sky Plot ---
            alt_val, az_val = None, None
            if header_ra != 'UNKNOWN' and header_dec != 'UNKNOWN':
                try:
                    # Assume standard RA (Hour angle) DEC (deg) format.
                    coord = SkyCoord(header_ra, header_dec, unit=(u.hourangle, u.deg))
                except Exception as e:
                    print(f">>> Coord conversion error: {e}")
                
                if coord is not None:
                    try:
                        altaz = coord.transform_to(AltAz(obstime=Time(obs_dt), location=IAC_LOCATION))
                        alt_val = altaz.alt.degree
                        az_val = altaz.az.degree
                    except Exception as e:
                        print(f">>> AltAz transform error: {e}")
                        
            
            # =================== AUTO-GUIDING ERROR LOGIC =================
            
            
            guiding_error_x = None
            guiding_error_y = None
            is_new_base = False
            if header_ra != 'UNKNOWN' and header_dec != 'UNKNOWN':
                
                current_coord = SkyCoord(header_ra, header_dec, unit=(u.hourangle, u.deg))

                if base_coord is None:
                    base_coord = current_coord
                    base_phot = phot.copy()
                    print(">>> Set base frame for guiding.")
                    history_base_changes.append(obs_dt.isoformat())
                    is_new_base = True
                    guiding_error_x = 0.0
                    guiding_error_y = 0.0

                else:
                    separation = current_coord.separation(base_coord)

                    if separation < 3 * u.arcmin:
                        print(">>> Same field")

                        base_xy = np.vstack([base_phot['x_fit'], base_phot['y_fit']]).T
                        curr_xy = np.vstack([phot['x_fit'], phot['y_fit']]).T

                        tree = cKDTree(base_xy)
                        dist, idx = tree.query(curr_xy, k=1)

                        good = dist < 5  # pixels

                        if np.sum(good) < 5:
                            print(">>> Not enough matches for guiding")
                        else:
                            dx = curr_xy[good, 0] - base_xy[idx[good], 0]
                            dy = curr_xy[good, 1] - base_xy[idx[good], 1]

                            # Outlier rejection
                            mask = (np.abs(dx - np.median(dx)) < 2) & (np.abs(dy - np.median(dy)) < 2)

                            dx = dx[mask]
                            dy = dy[mask]
                            
                            if len(dx) > 0 and len(dy) > 0:
                                arcsec_pixel = 0.336
                                guiding_error_x = np.median(dx) * arcsec_pixel
                                guiding_error_y = np.median(dy) * arcsec_pixel
                                print(f">>> Guiding error (arcsec): dx={guiding_error_x:.3f}, dy={guiding_error_y:.3f}")
                            else:
                                print(">>> Not enough matches after outlier rejection")
                            
                            

                    else:
                        print(">>> New field → resetting base")
                        base_coord = current_coord
                        base_phot = phot.copy()
                        is_new_base = True
                        guiding_error_x = 0.0
                        guiding_error_y = 0.0
                
                        
                
            with data_lock:
                history_alt.append(alt_val)
                history_az.append(az_val)
                    
                history_times.append(obs_dt.isoformat())
                history_indexes.append(index_csv)
                history_fwhm_x.append(np.median(phot['fwhm_x']))
                history_err_x.append(perc_x)
                history_fwhm_y.append(np.median(phot['fwhm_y']))
                history_err_y.append(perc_y)
                history_colors.append(FILTER_COLORS.get(filter_name.upper(), 'black'))
                
                history_airmass.append(header.get('AIRMASS', None))
                history_telfocus.append(header.get('TELFOCUS', None))
                history_humidity.append(header.get('HUMIDITY', None))
                history_temperature.append(header.get('TEMP', None))
                history_sources.append(len(phot))
                
                history_filters.append(filter_name)
                history_ra.append(header_ra)
                history_dec.append(header_dec)
                history_gamma_x.append(np.median(phot['gammax_fit']))
                history_gamma_y.append(np.median(phot['gammay_fit']))
                history_alpha.append(np.median(phot['alpha_fit']))
                history_phi.append(np.median(phot['phi_fit']))
                
                history_dx.append(guiding_error_x)
                history_dy.append(guiding_error_y)
                
                if is_new_base:
                    history_base_changes.append(obs_dt.isoformat())
            # --- END OF LOCK ---

        else:
            print(">>> Error: No valid sources found.")

        print(f">>> Time taken: {(time.perf_counter() - init):.2f} seconds\n")
        
    except Exception as e:
        print(f"Error processing {fits_file}: {e}")
    finally:
        file_data.close()


# --- WATCHDOG HANDLER & THREAD ---
class FitsHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.fits'): 
            time.wait(2)
            fits_queue.put(event.src_path)  

def process_queue_loop():
    while True:
        try:
            new_fits = fits_queue.get(timeout=1.0)
            time.sleep(0.5) 
            process_single_fits(new_fits)
            update_ginga_viewer(new_fits)
        except queue.Empty:
            continue
        
# Helper function to extract the number for sorting
def extract_sort_number(filepath):
    # Get just the filename (e.g., 'O20260220_1217.fits')
    filename = os.path.basename(filepath)
    try:
        # Split by '_' and take the last part ('1217.fits')
        number_part = filename.split('_')[-1]
        # Remove the extension to leave just the number ('1217')
        clean_number = number_part.replace('.fits', '')
        # 4. Convert to integer so 2 processes before 10
        return int(clean_number)
    except (IndexError, ValueError):
        # Fallback just in case a badly named file sneaks into the folder
        return 0


def save_csv_to_disk(target_dir):
    """Saves the telemetry data to a physical CSV file in the specified directory."""
    if len(history_times) == 0:
        print("No telemetry data collected. Skipping CSV generation.")
        return

    # Create a clean filename
    filename = os.path.join(target_dir, f"{get_observing_night_dir()[-7:]}.csv")
    print(f"\nSaving final telemetry data to {filename}...")

    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'INDEX', 'UTC', 'AIRMASS', 'FILTER', 'TELFOCUS', 
                'HUMIDITY', 'TEMPERATURE', 'RA', 'DEC', 'ALT', 'AZ',
                'GAMMA_X', 'FWHM_X', 'ERR_X', 'GAMMA_Y', 'FWHM_Y', 'ERR_Y',
                'ALPHA', 'PHI', 'NSOURCES', 'DX', 'DY'
            ])
            
            # Ensure chronological order
            dt_objects = [datetime.fromisoformat(t) for t in history_times]
            sorted_indices = np.argsort(dt_objects)
            
            # Write data rows
            for i in sorted_indices:
                writer.writerow([
                    history_indexes[i], history_times[i], history_airmass[i],
                    history_filters[i], history_telfocus[i], history_humidity[i],
                    history_temperature[i], history_ra[i], history_dec[i],
                    history_alt[i], history_az[i],
                    history_gamma_x[i], history_fwhm_x[i], history_err_x[i],
                    history_gamma_y[i], history_fwhm_y[i], history_err_y[i],
                    history_alpha[i], history_phi[i], history_sources[i],
                    history_dx[i], history_dy[i]
                    
                ])
        print("CSV saved successfully.")
    except Exception as e:
        print(f"Error saving CSV to disk: {e}")


def load_csv_to_history(target_dir):
    """Loads existing session data from CSV to prevent reprocessing FITS files."""
    global history_times, history_indexes, history_fwhm_x, history_fwhm_y
    global history_err_x, history_err_y, history_colors, history_alt, history_az
    global history_airmass, history_telfocus, history_humidity, history_temperature
    global history_sources, history_filters, history_ra, history_dec
    global history_gamma_x, history_gamma_y, history_alpha, history_phi
    global history_dx, history_dy

    csv_file = os.path.join(target_dir,  f"{get_observing_night_dir()[-7:]}.csv")
    if not os.path.exists(csv_file):
        return set() # No previous data found

    processed_indices = set()
    print(f"\n>>> Found existing session in {csv_file}. Restoring history...")

    def safe_float(val):
        """Helper to safely parse floats and handle None/empty strings."""
        return float(val) if val not in ('', 'None', None) else 0

    try:
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row['INDEX']
                processed_indices.add(idx)

                # Rebuild globals
                history_indexes.append(idx)
                history_times.append(row['UTC'])
                history_airmass.append(safe_float(row['AIRMASS']))
                
                filter_val = row['FILTER']
                history_filters.append(filter_val)
                history_colors.append(FILTER_COLORS.get(filter_val.upper(), 'black'))
                
                history_telfocus.append(safe_float(row['TELFOCUS']))
                history_humidity.append(safe_float(row['HUMIDITY']))
                history_temperature.append(safe_float(row['TEMPERATURE']))
                
                history_ra.append(row['RA'])
                history_dec.append(row['DEC'])
                history_alt.append(safe_float(row['ALT']))
                history_az.append(safe_float(row['AZ']))
                
                history_gamma_x.append(safe_float(row['GAMMA_X']))
                history_fwhm_x.append(safe_float(row['FWHM_X']))
                history_err_x.append(safe_float(row['ERR_X']))
                
                history_gamma_y.append(safe_float(row['GAMMA_Y']))
                history_fwhm_y.append(safe_float(row['FWHM_Y']))
                history_err_y.append(safe_float(row['ERR_Y']))
                
                history_alpha.append(safe_float(row['ALPHA']))
                history_phi.append(safe_float(row['PHI']))
                history_sources.append(safe_float(row['NSOURCES']))
                history_dx.append(safe_float(row['DX']))
                history_dy.append(safe_float(row['DY']))
                
        print(f">>> Restored {len(processed_indices)} frames from CSV.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        
    return processed_indices



def update_ginga_viewer(fits_file_path, channel_name='Image'):
    try:
        from ginga.util import grc
        # Connect to the local Ginga instance
        viewer = grc.RemoteClient('127.0.0.1', 11771)
        
        # Load the file into Ginga via the shell command
        viewer.shell().load_file(fits_file_path, channel_name)
        
    except ConnectionRefusedError:
        print("Warning: Ginga is not open or the RC plugin is not running.")
    except Exception as e:
        print(f"Error sending image to Ginga: {e}")

if __name__ == "__main__": 
    INCOMING_DIR = get_observing_night_dir()
    
    print(f"Targeting observation directory: {INCOMING_DIR}")
    

    if not os.path.exists(INCOMING_DIR):
        print(f"ERROR: The directory {INCOMING_DIR} does not exist yet.")
        print("Please ensure the camera/telescope software has created tonight's folder before running this script.")
        exit(1)  # Stop the program cleanly
    """
    # Read previous existing files if there are any
    existing_fits_files = glob.glob(os.path.join(INCOMING_DIR, '*.fits'))
    # Sort them in numerical order
    existing_fits_files.sort(key=extract_sort_number)
    
    # Add them to the queue
    for file_path in existing_fits_files:
        fits_queue.put(file_path)
        
    print(f"Found and queued {len(existing_fits_files)} existing FITS files.")
    # ------------------------------------------
    """
    
    # Load any existing data from a previous run tonight
    processed_indices = load_csv_to_history(INCOMING_DIR)
    
    # Read previous existing FITS files
    existing_fits_files = glob.glob(os.path.join(INCOMING_DIR, '*.fits'))
    existing_fits_files.sort(key=extract_sort_number)
    
    # Add them to the queue ONLY if they haven't been processed yet
    queued_count = 0
    for file_path in existing_fits_files:
        # Match the exact string slicing used in process_single_fits to get the index
        idx_to_check = os.path.basename(file_path)[-18:-5] 
        
        if idx_to_check not in processed_indices:
            fits_queue.put(file_path)
            queued_count += 1
            
    skipped_count = len(existing_fits_files) - queued_count
    print(f"Skipped {skipped_count} previously processed FITS files.")
    print(f"Queued {queued_count} new existing FITS files.")
    event_handler = FitsHandler()
    observer = Observer() 
    observer.schedule(event_handler, path=INCOMING_DIR, recursive=False)
    observer.start()
    
    processor_thread = threading.Thread(target=process_queue_loop, daemon=True)
    processor_thread.start()

    print(f"Listening for new FITS files in {INCOMING_DIR}...")
    print("Starting Web Dashboard on http://127.0.0.1:5000/")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        # Catch any unexpected errors just in case
        print(f"\nServer stopped due to an error: {e}")
        save_csv_to_disk(INCOMING_DIR)
        print(f"Saved a .csv to: {INCOMING_DIR}")
    finally:
        # The finally block ALWAYS runs when app.run() finishes, 
        # guaranteeing the CSV is saved on shutdown.
        print("\nStopping server...")
        save_csv_to_disk(INCOMING_DIR)
        
        observer.stop()
        observer.join()
        print("Shutdown complete. Have a good sleep!")