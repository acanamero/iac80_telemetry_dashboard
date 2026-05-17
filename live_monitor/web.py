#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:28:25 2026

@author: acanamero-ext
"""

import io
import os
import csv
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, render_template_string, Response, request
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time

import state
from config import LOCATION, ARCSEC_PIXEL, FILTER_COLORS, BASE_DIR
from utils import get_observing_night_dir

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
    with state.data_lock:
        if len(state.history_times) > 0:
            dt_objects = [datetime.fromisoformat(t) for t in state.history_times]
            sorted_indices = np.argsort(dt_objects)
            
            def get_sorted(lst, multiplier=1.0):
                result = []
                for i in sorted_indices:
                    val = lst[i]
                    if val is not None:
                        try: result.append(float(val) * multiplier)
                        except (ValueError, TypeError): result.append(None)
                    else: result.append(None)
                return result

            t_sorted = [state.history_times[i] for i in sorted_indices]
            idx_sorted = [state.history_indexes[i] for i in sorted_indices]
            c_sorted = [state.history_colors[i] for i in sorted_indices]
            x_sorted = get_sorted(state.history_fwhm_x, ARCSEC_PIXEL)
            err_x_sorted = get_sorted(state.history_err_x, ARCSEC_PIXEL)
            y_sorted = get_sorted(state.history_fwhm_y, ARCSEC_PIXEL)
            err_y_sorted = get_sorted(state.history_err_y, ARCSEC_PIXEL)
            
            air_sorted = get_sorted(state.history_airmass)
            foc_sorted = get_sorted(state.history_telfocus)
            hum_sorted = get_sorted(state.history_humidity)
            tem_sorted = get_sorted(state.history_temperature)
            src_sorted = get_sorted(state.history_sources)
            phi_sorted = get_sorted(state.history_phi, 180.0 / np.pi)
            dx_sorted = get_sorted(state.history_dx)
            dy_sorted = get_sorted(state.history_dy)
            
            latest_idx = sorted_indices[-1]
            latest_ra, latest_dec = state.history_ra[latest_idx], state.history_dec[latest_idx]
            obs_night = dt_objects[-1].strftime('%Y-%m-%d')
            latest_alt, latest_az = state.history_alt[latest_idx], state.history_az[latest_idx]
            alt_sorted = get_sorted(state.history_alt)

        else:
            t_sorted, idx_sorted, x_sorted, err_x_sorted, y_sorted, err_y_sorted, c_sorted = [], [], [], [], [], [], []
            air_sorted, foc_sorted, hum_sorted, tem_sorted, src_sorted, phi_sorted, alt_sorted, dx_sorted, dy_sorted = [], [], [], [], [], [], [], [], []
            obs_night, latest_ra, latest_dec, latest_alt, latest_az = '--', '--', '--', None, None
            
        return jsonify({
            'times': t_sorted, 'indexes': idx_sorted, 'fwhm_x': x_sorted, 'err_x': err_x_sorted,
            'fwhm_y': y_sorted, 'err_y': err_y_sorted, 'colors': c_sorted, 'airmass': air_sorted,
            'telfocus': foc_sorted, 'humidity': hum_sorted, 'temperature': tem_sorted, 'sources': src_sorted,
            'phi': phi_sorted, 'latest_ra': latest_ra, 'latest_dec': latest_dec, 'obs_night': obs_night,
            'latest_alt': latest_alt, 'altitude': alt_sorted, 'latest_az': latest_az, 'observer': state.latest_observer,
            'dx': dx_sorted, 'dy': dy_sorted, 'base_change_times': state.history_base_changes
        })

@app.route('/download_csv')
def download_csv():
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow([
        'INDEX', 'UTC', 'AIRMASS', 'FILTER', 'TELFOCUS', 'HUMIDITY', 'TEMPERATURE', 
        'RA', 'DEC', 'ALT', 'AZ', 'GAMMA_X', 'FWHM_X', 'ERR_X', 'GAMMA_Y', 'FWHM_Y', 
        'ERR_Y', 'ALPHA', 'PHI', 'NSOURCES', 'DX', 'DY'
    ])
    if len(state.history_times) > 0:
        dt_objects = [datetime.fromisoformat(t) for t in state.history_times]
        sorted_indices = np.argsort(dt_objects)
        with state.data_lock:
            for i in sorted_indices:
                writer.writerow([
                    state.history_indexes[i], state.history_times[i], state.history_airmass[i],
                    state.history_filters[i], state.history_telfocus[i], state.history_humidity[i],
                    state.history_temperature[i], state.history_ra[i], state.history_dec[i],
                    state.history_alt[i], state.history_az[i], state.history_gamma_x[i], 
                    state.history_fwhm_x[i], state.history_err_x[i], state.history_gamma_y[i], 
                    state.history_fwhm_y[i], state.history_err_y[i], state.history_alpha[i], 
                    state.history_phi[i], state.history_sources[i], state.history_dx[i], state.history_dy[i]
                ])
    folder_name = os.path.basename(get_observing_night_dir(BASE_DIR))
    return Response(si.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename=telemetry_{folder_name}.csv"})

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files: return jsonify(success=False, error="No file part in request")
    file = request.files['file']
    if file.filename == '': return jsonify(success=False, error="No file selected")
    try:
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        rows = list(csv.DictReader(stream))
        rows.sort(key=lambda x: str(x.get('UTC') or ''))
        
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
            csv_data['fwhm_x'].append(parse_val(row.get('FWHM_X'), ARCSEC_PIXEL))
            csv_data['err_x'].append(parse_val(row.get('ERR_X'), ARCSEC_PIXEL) or 0.0)
            csv_data['fwhm_y'].append(parse_val(row.get('FWHM_Y'), ARCSEC_PIXEL))
            csv_data['err_y'].append(parse_val(row.get('ERR_Y'), ARCSEC_PIXEL) or 0.0)
            csv_data['phi'].append(parse_val(row.get('PHI'), 180.0 / np.pi))
            csv_data['sources'].append(parse_val(row.get('NSOURCES')))
            csv_data['latest_ra'], csv_data['latest_dec'] = row.get('RA', '--'), row.get('DEC', '--')
            csv_data['dx'].append(parse_val(row.get('DX')))
            csv_data['dy'].append(parse_val(row.get('DY')))

        if rows:
            last_row = rows[-1]
            last_utc = last_row.get('UTC')
            if last_utc and last_utc not in ('UNKNOWN', '--', ''):
                try:
                    obs_dt = datetime.fromisoformat(last_utc)
                    csv_data['obs_night'] = obs_dt.strftime('%Y-%m-%d')
                    ra_str, dec_str = last_row.get('RA'), last_row.get('DEC')
                    if ra_str and dec_str:
                        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                        altaz = coord.transform_to(AltAz(obstime=Time(obs_dt), location=LOCATION))
                        csv_data['latest_alt'], csv_data['latest_az'] = altaz.alt.degree, altaz.az.degree
                except Exception: pass 
            
        return jsonify(success=True, csv_data=csv_data)
    except Exception as e: return jsonify(success=False, error=str(e))