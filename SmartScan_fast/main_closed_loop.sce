// ============================================================
// SMART SCAN - CLOSED-LOOP RUN (RF environment + IRSA receiver + ML)
// ============================================================
// Each 10-ms dwell:
//   1. wait for the ML command   ml_scan_command.txt  "<step> <center_MHz> <dwell_s>"
//   2. the 3-station IRSA receiver scans ONLY that 50-MHz window
//   3. write the observation to  ml_observation_closed_loop.csv
//   4. Python (online_controller.py) reads it, updates LightGBM/DBN/PPO
//      history, and writes the command for the next dwell
//
// Start THIS script first, then start online_controller.py.
// main.sce and ml_observation_live.csv are NOT touched by this script.
// ============================================================

clear;
clc;

BASE_DIR = get_absolute_file_path("main_closed_loop.sce");

// Several instances can run side by side (one per strategy): the dashboard sets the
// environment variable SPECTRA_TAG (e.g. ai / whittle / base) and every file gets that suffix.
SPECTRA_TAG = getenv("SPECTRA_TAG", "");
if SPECTRA_TAG == "" then
    SUFFIX = "";
else
    SUFFIX = "_" + SPECTRA_TAG;
end
DWELL_MIN_S = 0.001;              // commanded dwell is accepted between 1 ms ...
DWELL_MAX_S = 0.100;              // ... and 100 ms; anything else -> default 10 ms

// ------------------------------------------------------------
// Settings you may change
// ------------------------------------------------------------
CONFIG_ID = 7;                 // 7 or 8 = configurations the ML never trained on
ENABLE_PLOTS = %f;             // %t = also open Scilab graph windows (the dashboard shows all graphs)
PLOT_EVERY_N_DWELLS = 5;       // redraw graphs every N dwells
COMMAND_TIMEOUT_S = 600;       // stop if the controller sends nothing for this long (covers dashboard PAUSE)
MAX_DWELLS = %inf;             // e.g. 300 for a short demo

// The dashboard (dashboard_server.py) can choose the configuration and plots:
// it writes "<config_id> <plots 0/1>" to ml_closed_loop_request.txt before starting Scilab.
REQUEST_FILE = BASE_DIR + "ml_closed_loop_request" + SUFFIX + ".txt";
if isfile(REQUEST_FILE) then
    [fdq, qerr] = mopen(REQUEST_FILE,"rt");
    if qerr == 0 then
        rq = mgetl(fdq,1);
        mclose(fdq);
        if size(rq,"*") > 0 then
            rv = msscanf(rq(1),"%d %d");
            if size(rv,"*") >= 1 then CONFIG_ID = rv(1); end
            if size(rv,"*") >= 2 then ENABLE_PLOTS = (rv(2) == 1); end
        end
    end
    mdelete(REQUEST_FILE);
end

CSV_FILE = BASE_DIR + "data/whole_dataset_1.24m.csv";
CACHE_FILE = BASE_DIR + "data/pulse_cache_config_" + string(CONFIG_ID) + ".sod";
OBS_FILE = BASE_DIR + "ml_observation_closed_loop" + SUFFIX + ".csv";
CMD_FILE = BASE_DIR + "ml_scan_command" + SUFFIX + ".txt";
STATUS_FILE = BASE_DIR + "ml_closed_loop_status" + SUFFIX + ".txt";
EXTRA_FILE = BASE_DIR + "ml_closed_loop_extra" + SUFFIX + ".csv";
TRACKS_FILE = BASE_DIR + "ml_closed_loop_tracks" + SUFFIX + ".csv";
POSITIONS_FILE = BASE_DIR + "ml_closed_loop_positions" + SUFFIX + ".csv";
TOA_TO_SECONDS = 1e-6;

// Global receiver tuning range = the ML band grid (0 .. 16100 MHz, 322 bands)
RX_MIN_MHZ = 0;
RX_MAX_MHZ = 16100;

// ------------------------------------------------------------
// Load code
// ------------------------------------------------------------
exec(BASE_DIR + "load_pulse_dataset.sci",-1);
exec(BASE_DIR + "create_transmitters.sci",-1);
exec(BASE_DIR + "pulse_engine.sci",-1);
exec(BASE_DIR + "rf_environment.sci",-1);
exec(BASE_DIR + "irsa_receiver_config.sci",-1);
exec(BASE_DIR + "irsa_station.sci",-1);
exec(BASE_DIR + "irsa_geolocation.sci",-1);
exec(BASE_DIR + "irsa_observation.sci",-1);
exec(BASE_DIR + "irsa_environment_adapter.sci",-1);
exec(BASE_DIR + "ml_interface.sci",-1);
if ENABLE_PLOTS then
    exec(BASE_DIR + "live_plots.sci",-1);
end

// ------------------------------------------------------------
// RF environment (pulse cache)
// ------------------------------------------------------------
if isfile(CACHE_FILE) then
    mprintf("\nLoading pulse cache: %s\n",CACHE_FILE);
    load(CACHE_FILE,"events","tx_cfg","meta");
else
    if ~isfile(CSV_FILE) then
        error("Neither cache nor dataset found for configuration " + string(CONFIG_ID));
    end
    mprintf("\nParsing RF dataset for configuration %d (slow, one time)...\n",CONFIG_ID);
    [events,tx_cfg,meta] = load_pulse_dataset(CSV_FILE,CONFIG_ID);
    events = sort_events_by_toa(events);
    events.time_s = events.toa * TOA_TO_SECONDS;
    save(CACHE_FILE,"events","tx_cfg","meta");
end
if ~isfield(events,"time_s") then
    events.time_s = events.toa * TOA_TO_SECONDS;
end

tx = create_transmitters(tx_cfg);
irsa_state = irsa_receiver_init(tx);
irsa_state.st1.receiver.rf.min_frequency_mhz = RX_MIN_MHZ;
irsa_state.st1.receiver.rf.max_frequency_mhz = RX_MAX_MHZ;
irsa_state.st2.receiver.rf.min_frequency_mhz = RX_MIN_MHZ;
irsa_state.st2.receiver.rf.max_frequency_mhz = RX_MAX_MHZ;
irsa_state.st3.receiver.rf.min_frequency_mhz = RX_MIN_MHZ;
irsa_state.st3.receiver.rf.max_frequency_mhz = RX_MAX_MHZ;
stations = irsa_state.stations;
bw = irsa_state.st1.receiver.rf.instantaneous_bandwidth_mhz;
noise_floor = irsa_state.st1.receiver.detector.noise_floor_dbm;

mprintf("\n==============================================\n");
mprintf(" SMART SCAN - CLOSED LOOP  (configuration %d)\n",CONFIG_ID);
mprintf("==============================================\n");
mprintf("Transmitters: %d   Pulse events: %d\n",meta.n_transmitters,meta.n_events);
mprintf("Receiver range: %.0f - %.0f MHz, %.0f-MHz window, dwell %.3f s\n", ...
        RX_MIN_MHZ,RX_MAX_MHZ,bw,irsa_state.scan_dwell_s);

// ------------------------------------------------------------
// Dashboard (same graphs as main.sce)
// ------------------------------------------------------------
if ENABLE_PLOTS then
    init_live_plot_windows();
    plot_state = init_live_plots(20000);
    plot_state.axis.time_min = min(events.time_s);
    plot_state.axis.time_max = max(events.time_s);
    if plot_state.axis.time_max <= plot_state.axis.time_min then
        plot_state.axis.time_max = plot_state.axis.time_min + 1;
    end
    plot_state.axis.freq_min = max(0,floor(min(events.frequency)/50)*50);
    plot_state.axis.freq_max = ceil(max(events.frequency)/50)*50;
    if plot_state.axis.freq_max <= plot_state.axis.freq_min then
        plot_state.axis.freq_max = plot_state.axis.freq_min + 50;
    end
    plot_state = init_frequency_profile(plot_state,plot_state.axis.freq_min,plot_state.axis.freq_max);
    plot_state.axis.amp_min = floor(min(events.amplitude)/5)*5 - 5;
    plot_state.axis.amp_max = ceil(max(events.amplitude)/5)*5 + 5;
    plot_state.axis.snr_min = 0;
    plot_state.axis.snr_max = max(60,ceil(max(events.amplitude - noise_floor)/5)*5);
    plot_state.axis.pw_min = 0;
    plot_state.axis.pw_max = ceil(max(events.pulse_width)/5)*5 + 5;
    if plot_state.axis.pw_max <= 0 then
        plot_state.axis.pw_max = 100;
    end
    dt_all = diff(events.time_s) * 1e6;
    dt_all = dt_all(dt_all >= 0 & ~isnan(dt_all) & ~isinf(dt_all));
    if size(dt_all,"*") > 0 then
        dt_upper = min(max(dt_all),100000);
        plot_state.axis.dt_max_us = max(1000,ceil(dt_upper/1000)*1000);
    end
    sp_x = [stations(:,1); tx_cfg(:,2)];
    sp_y = [stations(:,2); tx_cfg(:,3)];
    plot_state.axis.spatial_x_min = floor(min(sp_x)/10)*10 - 10;
    plot_state.axis.spatial_x_max = ceil(max(sp_x)/10)*10 + 10;
    plot_state.axis.spatial_y_min = floor(min(sp_y)/10)*10 - 10;
    plot_state.axis.spatial_y_max = ceil(max(sp_y)/10)*10 + 10;
end

// ------------------------------------------------------------
// Closed-loop interface
// ------------------------------------------------------------
ml_cl_init(OBS_FILE,CMD_FILE,STATUS_FILE);
ml_cl_init_extra(EXTRA_FILE,TRACKS_FILE);
ml_cl_write_positions(POSITIONS_FILE,stations,tx_cfg);
ml_cl_write_status(STATUS_FILE,"READY " + string(CONFIG_ID));
mprintf("\nREADY. Now start the Python controller (online_controller.py).\n");
mprintf("Waiting for ML command for dwell 1 ...\n");

TP = 0; FP = 0; TN = 0; FN = 0;
stopped_by_controller = %f;
hits = 0;
n_total = length(events.time_s);
scan_start_idx = 1;
step = 0;
stopped_by_timeout = %f;
t_wall = getdate("s");

while scan_start_idx <= n_total & step < MAX_DWELLS

    step = step + 1;

    // ---- 1. wait for the ML decision (band + dwell length) ----
    [ok,center_mhz,dwell_cmd,stop_req] = ml_cl_wait_command(CMD_FILE,step,COMMAND_TIMEOUT_S);
    if stop_req then
        mprintf("\nSTOP received from the controller at dwell %d.\n",step);
        stopped_by_controller = %t;
        step = step - 1;
        break;
    end
    if ~ok then
        mprintf("\nNo ML command for dwell %d within %d s - stopping.\n",step,COMMAND_TIMEOUT_S);
        stopped_by_timeout = %t;
        break;
    end
    lo_mhz = center_mhz - bw/2;
    hi_mhz = center_mhz + bw/2;

    // ---- 2. the dwell of the RF environment, with the COMMANDED length ----
    dwell_s = dwell_cmd;
    if isnan(dwell_s) | dwell_s < DWELL_MIN_S | dwell_s > DWELL_MAX_S then
        dwell_s = 0.010;
    end
    irsa_state.scan_dwell_s = dwell_s;
    irsa_state.st1.receiver.scan.dwell_time_s = dwell_s;
    irsa_state.st2.receiver.scan.dwell_time_s = dwell_s;
    irsa_state.st3.receiver.scan.dwell_time_s = dwell_s;
    scan_start_s = events.time_s(scan_start_idx);
    scan_end_s = scan_start_s + dwell_s;
    scan_end_idx = scan_start_idx;
    while scan_end_idx < n_total & events.time_s(scan_end_idx+1) <= scan_end_s
        scan_end_idx = scan_end_idx + 1;
    end
    ev_range = scan_start_idx:scan_end_idx;
    rf_slice = [events.time_s(ev_range), events.frequency(ev_range), events.pulse_width(ev_range), ...
                events.aoa(ev_range), events.amplitude(ev_range), events.tx_id(ev_range)];

    // ---- 2. receiver scans ONLY the commanded window ----
    local_idx = find(rf_slice(:,2) >= lo_mhz & rf_slice(:,2) <= hi_mhz);
    gt_present = %f;
    detected = %f;
    tracks = [];
    if size(local_idx,"*") == 0 then
        ml_cl_write_extra(EXTRA_FILE,TRACKS_FILE,step,scan_start_s,%f,[]);
        ml_cl_write_empty_observation(OBS_FILE,CONFIG_ID,step,scan_start_s, ...
                                      center_mhz,lo_mhz,hi_mhz,dwell_s);
    else
        window_events = rf_slice(local_idx,:);
        gt_idx = find(window_events(:,6) > 0);
        gt_present = (size(gt_idx,"*") > 0);
        [irsa_state,receiver_observation] = irsa_receiver_scan(irsa_state,window_events,scan_start_s,center_mhz);
        p1 = irsa_state.st1.last_pdws;
        p2 = irsa_state.st2.last_pdws;
        p3 = irsa_state.st3.last_pdws;
        tracks = receiver_observation.tracks;
        detected = (size(p1,"*") > 0) | (size(p2,"*") > 0) | (size(p3,"*") > 0);

        // ---- 3. observation for the ML (+ truth flag and geolocation for the dashboard) ----
        ml_cl_write_extra(EXTRA_FILE,TRACKS_FILE,step,scan_start_s,gt_present,tracks);
        ml_cl_write_observation(OBS_FILE,receiver_observation,CONFIG_ID,step, ...
                                lo_mhz,hi_mhz,dwell_s);

        if ENABLE_PLOTS then
            for g = 1:size(gt_idx,"*")
                gi = gt_idx(g);
                gt_snr = window_events(gi,5) - noise_floor;
                bin = find(gt_snr >= plot_state.snr_bins(1:$-1) & gt_snr < plot_state.snr_bins(2:$));
                if size(bin,"*") > 0 then
                    plot_state.snr_total(bin(1)) = plot_state.snr_total(bin(1)) + 1;
                    ids_now = [];
                    if size(p1,"*") > 0 then ids_now = [ids_now; p1(:,1)]; end
                    if size(p2,"*") > 0 then ids_now = [ids_now; p2(:,1)]; end
                    if size(p3,"*") > 0 then ids_now = [ids_now; p3(:,1)]; end
                    if size(find(ids_now == window_events(gi,6)),"*") > 0 then
                        plot_state.snr_detected(bin(1)) = plot_state.snr_detected(bin(1)) + 1;
                    end
                end
            end
            all_pdw = [];
            if size(p1,"*") > 0 then all_pdw = [all_pdw;p1]; end
            if size(p2,"*") > 0 then all_pdw = [all_pdw;p2]; end
            if size(p3,"*") > 0 then all_pdw = [all_pdw;p3]; end
            for q = 1:size(all_pdw,1)
                plot_state = append_receiver_pdw(plot_state,all_pdw(q,:),all_pdw(q,2));
            end
            plot_state = update_spatial_data(plot_state,stations,tracks);
        end
    end

    // ---- confusion matrix for the SCANNED window ----
    if gt_present & detected then
        TP = TP + 1;
    elseif ~gt_present & detected then
        FP = FP + 1;
    elseif gt_present & ~detected then
        FN = FN + 1;
    else
        TN = TN + 1;
    end
    if detected then
        hits = hits + 1;
    end

    if ENABLE_PLOTS then
        plot_state = append_detection_scan(plot_state,scan_start_s,detected);
        plot_state.TP = TP; plot_state.FP = FP; plot_state.TN = TN; plot_state.FN = FN;
        if modulo(step,PLOT_EVERY_N_DWELLS) == 0 then
            plot_state = update_live_plots(plot_state,step,n_total);
        end
    end

    if modulo(step,100) == 0 then
        mprintf("Dwell %d | t=%.3f s | scanned %.0f MHz | detections so far %d (%.1f%%)\n", ...
                step,scan_start_s,center_mhz,hits,100*hits/step);
    end

    if scan_end_idx >= n_total then
        break;
    end
    scan_start_idx = scan_end_idx + 1;
end

if ENABLE_PLOTS then
    plot_state = update_live_plots(plot_state,step,n_total);
end
ml_cl_write_status(STATUS_FILE,"DONE " + string(CONFIG_ID) + " " + string(step));

mprintf("\n==============================================\n");
mprintf(" CLOSED-LOOP RUN COMPLETE (configuration %d)\n",CONFIG_ID);
mprintf("==============================================\n");
if stopped_by_timeout then
    mprintf("Stopped early: no command from Python.\n");
end
if stopped_by_controller then
    mprintf("Stopped by the controller / dashboard.\n");
end
mprintf("Dwells scanned: %d\n",step);
mprintf("Receiver detections: %d (%.1f%% of scans)\n",hits,100*hits/max(step,1));
mprintf("TP=%d  FP=%d  TN=%d  FN=%d\n",TP,FP,TN,FN);
mprintf("Wall-clock time: %.0f s\n",getdate("s") - t_wall);
mprintf("Observations: %s\n",OBS_FILE);
mprintf("Python writes the audit when it sees DONE.\n");
