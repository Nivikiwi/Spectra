SMART SCAN - SCILAB + IRSA BASELINE VALIDATION
================================================

This version is a corrected baseline for the RF environment and
3-station IRSA-style receiver.

DATA FLOW
---------
Dataset
  -> Scilab RF environment replay
  -> 3 IRSA receiving stations
  -> receiver PDWs
  -> multi-station fusion / geolocation
  -> ML-safe receiver observation
  -> dashboard

IMPORTANT BASELINE MODE
-----------------------
The current main.sce uses EVENT-COVERAGE VALIDATION mode.
For each 10-ms RF time dwell, every 50-MHz frequency window that
contains an event is evaluated. This is deliberately a validation
mode so the complete dataset can be exercised before Smart Scan is
introduced.

It is NOT the final Smart Scan scheduler.
The later ML workflow will replace this with:

  ML belief -> Smart Scan scheduler -> selected 50-MHz window
             -> IRSA receiver -> observation -> feedback

FILES
-----
main.sce                     Main replay and dashboard
irsa_environment_adapter.sci RF environment -> station adapter
irsa_station.sci             Receiver station and PDW generation
irsa_geolocation.sci         Multi-station bearing fusion
irsa_observation.sci         Receiver / ML-safe observation
irsa_receiver_config.sci     Receiver parameters
live_plots.sci               9 live dashboard plots

virtual_receiver.sci         Retained for reference; NOT executed

DATA
----
The included dataset is whole_dataset_1.24m.csv.
Configuration 0 is used by default.
The included pulse_cache_config_0.sod may be reused.

RUN
---
Open this folder as the Scilab working directory, then execute:

  exec("main.sce",-1);

EXPECTED OUTPUT
---------------
The console should report approximately:
  Transmitters found: 14
  Pulse events: 79340
  Receiving stations: 3
  Instantaneous bandwidth: 50 MHz

The dashboard contains:
  1 Frequency vs Time
  2 Amplitude / Power vs Frequency
  3 SNR vs Frequency
  4 Signal Detection Timeline
  5 Pulse Width Distribution
  6 Inter-Arrival Time / Pulse Interval
  7 Detection Probability vs SNR
  8 Receiver Detection Confusion Matrix
  9 Multi-Station Spatial Geolocation

GROUND TRUTH VS ML
------------------
Ground-truth transmitter IDs and true positions are used internally
for simulator evaluation and receiver association. They are not copied
into obs.ml / ml_observation.

The ML-safe observation is exposed by the main replay as:

  ml_observation

It contains receiver-derived scan and RF/geolocation features.

NEXT STEP
---------
After this baseline produces the expected dashboard, connect the ML
workflow to ml_observation. The ML scheduler should then replace the
EVENT-COVERAGE validation loop rather than replacing the RF environment
or IRSA receiver.

LIVE RECEIVER / ML INTERFACE
----------------------------
The corrected dashboard now opens a tenth window:
  Smart Scan - Receiver Observation Table

The receiver also writes, during replay:
  ml_observation_live.csv

This CSV contains one row per receiver scan window and includes:
  scan time, scan center/band, dwell, total PDWs, station hit count,
  track count, RS1/RS2/RS3 PDW counts, per-station mean SNR/AoA,
  mean/max RF features, and hit.

External ML control is prepared through:
  ml_scan_command.txt

Command format:
  <center_frequency_mhz> <dwell_seconds>

Example:
  8250.0 0.010

In the current baseline validation run, the command is displayed but does
not replace the coverage sweep. After the ML workflow is connected, the
scheduler will use this command channel to select the next 50-MHz window.

SPATIAL GRAPH
-------------
The spatial display is scatter-based. The initial axes are fixed from the
receiver geometry and dataset position range. After the first draw, the
Scilab figure zoom tool can be used interactively without the next refresh
resetting the spatial view.

CONFUSION MATRIX
----------------
The receiver confusion matrix is rendered as a coloured 2x2 matrix using
explicit TN/FP/FN/TP cells.
