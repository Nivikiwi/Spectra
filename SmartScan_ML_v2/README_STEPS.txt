SPECTRA - COMPLETE STEPS
========================
Folder layout (all side by side):
    C:\Users\bansa\Documents\7finalSSS\SmartScan_fast          (Scilab)
    C:\Users\bansa\Documents\7finalSSS\SmartScan_ML_v2         (ML + dashboard_server.py)
    C:\Users\bansa\Documents\7finalSSS\smart_scan_dashboard    (the web page)

ONE-TIME SETUP
 1. Copy the 3 files from SmartScan_fast_update.zip into SmartScan_fast (Replace).
 2. Copy pulse_cache_config_9.sod, _10.sod, _11.sod from pulseSSS into SmartScan_fast\data\
 3. Scilab: open Scilab, then in the console type (change the path if yours differs):
        cd("C:\Users\bansa\Documents\7finalSSS\SmartScan_fast")
        exec("sweep_fast.sce",-1);
    Wait for "SWEEP DONE" (about 1 minute). This creates ml_observation_sweep.csv. Close Scilab.
 4. PowerShell (venv active), in SmartScan_ML_v2:
        python run_all_training.py --scilab-dir "C:\Users\bansa\Documents\7finalSSS\SmartScan_fast"
    Takes about 25 minutes (steps 1-7). Step 4 now trains 3 PPO models and keeps the best one
    (PPO training is random - this removes the run-to-run swing). For a quick run:
        python run_all_training.py --scilab-dir "...\SmartScan_fast" --ppo-timesteps 60000
    ...but with one seed the result varies; add nothing to keep the default 3 seeds. Last line: ALL STEPS DONE.
    Step 7 compares the custom model with every other algorithm -> reports\algorithm_comparison.xlsx
    (the custom model with adaptive dwell runs on live Scilab there; add --scilab-exe "...\WScilex-cli.exe"
    if Scilab is not found automatically).

SETTINGS (smartscan_config.py, smartscan_core.py)
  ML_GAMMA = 0.5 (smartscan_core.py)  weight of the LightGBM+DBN probability ADDED to the Whittle index
  SPECTRA_ADAPTIVE_DWELL = True   adaptive 5-15 ms dwell for Spectra (0 false alarms on the test);
                                  False = fixed 10 ms (more detections and pulses on the test)
  LPD_DWELL_GAIN = 0.0            longer dwell on weak signals - tested, no gain, therefore off
  Test settings (environment variables, PowerShell):  $env:SPECTRA_FORCE_TIER="2"  (always Whittle)
                                                      $env:SPECTRA_PPO_TIMEOUT_MS="1"  (real PPO timeouts)

EVERY DEMO
 5. PowerShell (venv active), in SmartScan_ML_v2:
        python dashboard_server.py
    Check the lines it prints:
        Receivers     : 3 x LIVE Scilab (background, no windows)
        Configurations: [0, 1, ..., 11]
        Dashboard ready: http://localhost:8000
    If it says "Scilab program: NOT found", start it like this instead (use YOUR Scilab version folder):
        python dashboard_server.py --scilab-exe "C:\Program Files\scilab-2024.1.0\bin\WScilex-cli.exe"
 6. Open http://localhost:8000 in Chrome or Edge.
 7. Simulation Control -> START. All configurations 0 -> 11 run one after another (about 2-5 min).
    POLICY (top bar) = the ONE policy that runs (Spectra / Whittle / Round Robin).
    COMPARE ALL: ON  = all three run side by side (for the Strategy Comparison tab).
 8. Stop the server with Ctrl+C.

WHERE TO LOOK
  LIVE MONITOR          band heatmap, live decision, KPI tiles, frequency/time, geolocation
  DECISION LOG          every decision (10 columns) + hit/miss
  STRATEGY COMPARISON   COMPARE ALL results + all algorithms (step 7 report)
  MODEL HEALTH          Spectra tiers (PPO / Whittle failover), failover events, decision time, LPI/LPD, events
  Excel audit           saved automatically in SmartScan_ML_v2\reports\spectra_audit_<date>.xlsx
                        and via DOWNLOAD EXCEL AUDIT

PROBLEMS
  Header says BACKEND: OFFLINE     -> the server is not running (step 5), or you opened index.html
                                      by double-click instead of http://localhost:8000
  Status line shows ERROR ...      -> read SmartScan_fast\scilab_background.log (last lines)
  "Port 8000 in use"              -> python dashboard_server.py --port 8010, open http://localhost:8010
  "feature count does not match"   -> run step 4 again (the training)
