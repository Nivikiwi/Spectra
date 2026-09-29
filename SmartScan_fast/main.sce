// ============================================================
// SMART SCAN - SCILAB RF ENVIRONMENT + IRSA RECEIVER
// BASELINE VALIDATION REPLAY
// ============================================================
//
// IMPORTANT:
// This baseline intentionally performs SPECTRUM COVERAGE validation.
// For each RF time dwell, every 50-MHz window containing data is
// evaluated. This is NOT yet the Smart Scan policy.
//
// The later ML workflow will replace this coverage loop with:
//     ML belief -> scheduler -> one selected window -> receiver
//
// virtual_receiver.sci is retained for reference only and is NOT used.
// ============================================================

clear;
clc;

BASE_DIR = get_absolute_file_path("main.sce");

// ------------------------------------------------------------
// Dataset / replay
// ------------------------------------------------------------
CSV_FILE = BASE_DIR + "data/whole_dataset_1.24m.csv";
CONFIG_ID_TO_RUN = 0;
RUN_ALL_CONFIGURATIONS = %t;
TOA_TO_SECONDS = 1e-6;

USE_DATA_CACHE = %t;
FORCE_REBUILD_CACHE = %f;
// CACHE_FILE is created per configuration inside the replay loop.

// ------------------------------------------------------------
// Dashboard
// ------------------------------------------------------------
PLOT_HISTORY_LENGTH = 20000;
PLOT_UPDATE_EVERY_SCAN = 1;
ML_INTERFACE_MODE = "BASELINE"; // change to "EXTERNAL" after ML workflow is connected
PLOT_PAUSE_SECONDS = 0.04;
MAX_SCANS_TO_PROCESS = %inf;

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
exec(BASE_DIR + "live_plots.sci",-1);
exec(BASE_DIR + "ml_interface.sci",-1);

if ~isfile(CSV_FILE) then
    error("Dataset not found: " + CSV_FILE);
end

// FAST ALL-CONFIGURATION MODE:
// The current whole_dataset_1.24m.csv contains configurations 0..11.
// Do not scan the entire 1.24M-row CSV just to rediscover these IDs.
if RUN_ALL_CONFIGURATIONS then
    CONFIG_IDS = 0:11;
else
    CONFIG_IDS = CONFIG_ID_TO_RUN;
end

ml_init_interface(BASE_DIR);

mprintf("\nConfigurations scheduled: %d\n",size(CONFIG_IDS,"*"));
mprintf("Configuration IDs: ");
for kk = 1:size(CONFIG_IDS,"*")
    mprintf("%d ",CONFIG_IDS(kk));
end
mprintf("\n");

// ------------------------------------------------------------
// Configuration replay loop
// ------------------------------------------------------------
for cfg_index = 1:size(CONFIG_IDS,"*")

    CONFIG_ID_CURRENT = CONFIG_IDS(cfg_index);
    CACHE_FILE = BASE_DIR + "data/pulse_cache_config_" + string(CONFIG_ID_CURRENT) + ".sod";

    mprintf("\n============================================================\n");
    mprintf("Starting configuration %d (%d of %d)\n", ...
            CONFIG_ID_CURRENT,cfg_index,size(CONFIG_IDS,"*"));
    mprintf("============================================================\n");

    // ------------------------------------------------------------
    // Load / build cache
    // ------------------------------------------------------------
    if USE_DATA_CACHE & isfile(CACHE_FILE) & ~FORCE_REBUILD_CACHE then
        mprintf("\nLoading prepared pulse cache...\n");
        load(CACHE_FILE,"events","tx_cfg","meta");
    else
        mprintf("\nParsing RF dataset...\n");
        [events,tx_cfg,meta] = load_pulse_dataset(CSV_FILE,CONFIG_ID_CURRENT);
        events = sort_events_by_toa(events);
        events.time_s = events.toa * TOA_TO_SECONDS;
        save(CACHE_FILE,"events","tx_cfg","meta");
        mprintf("Cache created: %s\n",CACHE_FILE);
    end

    if ~isfield(events,"time_s") then
        events.time_s = events.toa * TOA_TO_SECONDS;
    end

    // ------------------------------------------------------------
    // Transmitter model
    // ------------------------------------------------------------
    tx = create_transmitters(tx_cfg);

    mprintf("\n==============================================\n");
    mprintf(" SMART SCAN - IRSA MULTI-STATION RECEIVER\n");
    mprintf("==============================================\n");
    mprintf("Transmitters found: %d\n",meta.n_transmitters);
    mprintf("Pulse events: %d\n",meta.n_events);

    // ------------------------------------------------------------
    // IRSA receiver
    // ------------------------------------------------------------
    irsa_state = irsa_receiver_init(tx);

    // Match tunable RF range to the actual selected dataset.
    data_fmin = min(events.frequency);
    data_fmax = max(events.frequency);
    bw = irsa_state.st1.receiver.rf.instantaneous_bandwidth_mhz;
    rx_min = floor(data_fmin/bw)*bw;
    rx_max = ceil(data_fmax/bw)*bw;
    rx_min = max(0,rx_min);

    for s = 1:3
        if s == 1 then
            irsa_state.st1.receiver.rf.min_frequency_mhz = rx_min;
            irsa_state.st1.receiver.rf.max_frequency_mhz = rx_max;
        elseif s == 2 then
            irsa_state.st2.receiver.rf.min_frequency_mhz = rx_min;
            irsa_state.st2.receiver.rf.max_frequency_mhz = rx_max;
        else
            irsa_state.st3.receiver.rf.min_frequency_mhz = rx_min;
            irsa_state.st3.receiver.rf.max_frequency_mhz = rx_max;
        end
    end

    stations = irsa_state.stations;

    n_windows = round((rx_max-rx_min)/bw);

    mprintf("\nIRSA profile: HETEROGENEOUS_SDR\n");
    mprintf("Receiving stations: %d\n",size(stations,1));
    mprintf("Instantaneous bandwidth: %.1f MHz\n",bw);
    mprintf("Tunable frequency range: %.1f - %.1f MHz\n",rx_min,rx_max);
    mprintf("Number of 50-MHz windows: %d\n",n_windows);
    mprintf("Dwell time: %.5f s\n",irsa_state.scan_dwell_s);
    mprintf("Baseline mode: EVENT-COVERAGE VALIDATION\n");

    // ------------------------------------------------------------
    // Dashboard
    // ------------------------------------------------------------
    init_live_plot_windows();
    plot_state = init_live_plots(PLOT_HISTORY_LENGTH);

    // Fixed dashboard axes derived once from the complete dataset.
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

    // Initialize the persistent 50-MHz spectrum profiles.
    plot_state = init_frequency_profile( ...
        plot_state,plot_state.axis.freq_min,plot_state.axis.freq_max);

    plot_state.axis.amp_min = floor(min(events.amplitude)/5)*5 - 5;
    plot_state.axis.amp_max = ceil(max(events.amplitude)/5)*5 + 5;

    plot_state.axis.snr_min = 0;
    plot_state.axis.snr_max = max(60,ceil(max(events.amplitude - irsa_state.st1.receiver.detector.noise_floor_dbm)/5)*5);

    plot_state.axis.pw_min = 0;
    plot_state.axis.pw_max = ceil(max(events.pulse_width)/5)*5 + 5;
    if plot_state.axis.pw_max <= 0 then
        plot_state.axis.pw_max = 100;
    end

    // Keep the inter-arrival axis fixed.  Limit the upper bound so a
    // single large dataset gap cannot flatten the useful pulse timing.
    dt_all = diff(events.time_s) * 1e6;
    dt_all = dt_all(dt_all >= 0 & ~isnan(dt_all) & ~isinf(dt_all));
    if size(dt_all,"*") > 0 then
        dt_upper = max(dt_all);
        dt_upper = min(dt_upper,100000);
        plot_state.axis.dt_max_us = max(1000,ceil(dt_upper/1000)*1000);
    end

    // Receiver geometry / expected signal area.  Keep this constant so
    // moving tracks are easy to compare from one frame to the next.
    // Spatial limits are fixed from the receiver geometry and the
    // transmitter locations contained in the selected dataset.
    sp_x = [stations(:,1); tx_cfg(:,2)];
    sp_y = [stations(:,2); tx_cfg(:,3)];
    plot_state.axis.spatial_x_min = floor(min(sp_x)/10)*10 - 10;
    plot_state.axis.spatial_x_max = ceil(max(sp_x)/10)*10 + 10;
    plot_state.axis.spatial_y_min = floor(min(sp_y)/10)*10 - 10;
    plot_state.axis.spatial_y_max = ceil(max(sp_y)/10)*10 + 10;

    // ------------------------------------------------------------
    // Replay
    // ------------------------------------------------------------
    n_total = length(events.time_s);
    scan_start_idx = 1;
        scan_number = 0;
        step_number = 0;

    while scan_start_idx <= n_total

        // One temporal receiver dwell.
        scan_start_s = events.time_s(scan_start_idx);
        scan_end_s = scan_start_s + irsa_state.scan_dwell_s;

        scan_end_idx = scan_start_idx;
        while scan_end_idx < n_total & events.time_s(scan_end_idx+1) <= scan_end_s
            scan_end_idx = scan_end_idx + 1;
        end

        rf_slice = [ ...
            events.time_s(scan_start_idx:scan_end_idx), ...
            events.frequency(scan_start_idx:scan_end_idx), ...
            events.pulse_width(scan_start_idx:scan_end_idx), ...
            events.aoa(scan_start_idx:scan_end_idx), ...
            events.amplitude(scan_start_idx:scan_end_idx), ...
            events.tx_id(scan_start_idx:scan_end_idx) ...
        ];

        // Determine which 50-MHz windows contain RF events during this dwell.
        window_id = floor((rf_slice(:,2)-rx_min)/bw);

        for q = 1:size(window_id,1)
            if window_id(q) < 0 then
                window_id(q) = 0;
            end
            if window_id(q) > n_windows-1 then
                window_id(q) = n_windows-1;
            end
        end

        occupied_windows = unique(window_id);

        // Empty spectrum windows are known ground-truth negatives.
        // They do not need to be pushed through the receiver because
        // this baseline has no independent noise generator in empty windows.
        n_empty_windows = n_windows - size(occupied_windows,"*");

    plot_state.TN = plot_state.TN + n_empty_windows;

    // Write an observation for every empty receiver window.
    // This is important because the ML scheduler must learn
    // both successful detections AND unsuccessful scans.

    for empty_wid = 0:(n_windows-1)

        if size(find(occupied_windows == empty_wid),"*") == 0 then

            empty_lo = rx_min + empty_wid*bw;
            empty_hi = empty_lo + bw;
            empty_center = empty_lo + bw/2;

            step_number = step_number + 1;

            ml_write_empty_observation( ...
                BASE_DIR, ...
                CONFIG_ID_CURRENT, ...
                step_number, ...
                scan_start_s, ...
                empty_center, ...
                empty_lo, ...
                empty_hi, ...
                irsa_state.scan_dwell_s);

            // Record the negative scan in the live detection timeline.
            plot_state = append_detection_scan( ...
                plot_state,scan_start_s,%f);

        end

    end

        // --------------------------------------------------------
        // Process every occupied frequency window.
        // --------------------------------------------------------
        for w = 1:size(occupied_windows,"*")

            wid = occupied_windows(w);
            lo_mhz = rx_min + wid*bw;
            hi_mhz = lo_mhz + bw;
            center_mhz = lo_mhz + bw/2.0;

            local_idx = find( ...
                rf_slice(:,2) >= lo_mhz & ...
                rf_slice(:,2) <= hi_mhz);

            if size(local_idx,"*") == 0 then
                continue;
            end

            window_events = rf_slice(local_idx,:);

            // Ground truth: a positive transmitter exists in this window.
            gt_idx = find(window_events(:,6) > 0);
            gt_present = (size(gt_idx,"*") > 0);

            // Ground-truth SNR denominator for Pd(SNR).
            if size(gt_idx,"*") > 0 then
                for g = 1:size(gt_idx,"*")
                    gi = gt_idx(g);
                    gt_snr = window_events(gi,5) - ...
                             irsa_state.st1.receiver.detector.noise_floor_dbm;

                    bin = find( ...
                        gt_snr >= plot_state.snr_bins(1:$-1) & ...
                        gt_snr < plot_state.snr_bins(2:$));

                    if size(bin,"*") > 0 then
                        plot_state.snr_total(bin(1)) = ...
                            plot_state.snr_total(bin(1)) + 1;
                    end
                end
            end

            // ----------------------------------------------------
            // IRSA receiver
            // ----------------------------------------------------
            [irsa_state,receiver_observation] = ...
                irsa_receiver_scan( ...
                    irsa_state,window_events,scan_start_s,center_mhz);

            p1 = irsa_state.st1.last_pdws;
            p2 = irsa_state.st2.last_pdws;
            p3 = irsa_state.st3.last_pdws;
            tracks = receiver_observation.tracks;

            scan_number = scan_number + 1;
            step_number = step_number + 1;

            if scan_number > MAX_SCANS_TO_PROCESS then
                break;
            end

            // Receiver says "activity detected" if any station produced a PDW.
            detected = ...
                (size(p1,"*") > 0) | ...
                (size(p2,"*") > 0) | ...
                (size(p3,"*") > 0);

            plot_state = append_detection_scan( ...
                plot_state,scan_start_s,detected);

            // ----------------------------------------------------
            // Confusion matrix is RF detection, NOT geolocation.
            // ----------------------------------------------------
            if gt_present & detected then
                plot_state.TP = plot_state.TP + 1;
            elseif ~gt_present & detected then
                plot_state.FP = plot_state.FP + 1;
            elseif gt_present & ~detected then
                plot_state.FN = plot_state.FN + 1;
            end

            // ----------------------------------------------------
            // Pd(SNR): emitter is considered detected if its ID
            // appears in at least one station PDW.
            // ID 0 is background and is ignored for Pd.
            // ----------------------------------------------------
            detected_ids = [];
            for pp = 1:3
                if pp == 1 then
                    ppdw = p1;
                elseif pp == 2 then
                    ppdw = p2;
                else
                    ppdw = p3;
                end

                if size(ppdw,"*") > 0 then
                    ii = find(ppdw(:,1) > 0);
                    if size(ii,"*") > 0 then
                        detected_ids = [detected_ids;ppdw(ii,1)];
                    end
                end
            end
            detected_ids = unique(detected_ids);

            if size(gt_idx,"*") > 0 then
                for g = 1:size(gt_idx,"*")
                    gi = gt_idx(g);
                    gt_id = window_events(gi,6);
                    gt_snr = window_events(gi,5) - ...
                             irsa_state.st1.receiver.detector.noise_floor_dbm;

                    bin = find( ...
                        gt_snr >= plot_state.snr_bins(1:$-1) & ...
                        gt_snr < plot_state.snr_bins(2:$));

                    if size(bin,"*") > 0 then
                        if size(find(detected_ids == gt_id),"*") > 0 then
                            plot_state.snr_detected(bin(1)) = ...
                                plot_state.snr_detected(bin(1)) + 1;
                        end
                    end
                end
            end

            // ----------------------------------------------------
            // Dashboard uses the strongest/available observations from
            // all three stations.  This keeps the display representative
            // of the multi-station receiver instead of RS1 alone.
            // ----------------------------------------------------
            all_pdw = [];
            if size(p1,"*") > 0 then all_pdw = [all_pdw;p1]; end
            if size(p2,"*") > 0 then all_pdw = [all_pdw;p2]; end
            if size(p3,"*") > 0 then all_pdw = [all_pdw;p3]; end

            if size(all_pdw,1) > 0 then
                for q = 1:size(all_pdw,1)
                    plot_state = append_receiver_pdw( ...
                        plot_state,all_pdw(q,:),all_pdw(q,2));
                end
            end

            plot_state = update_spatial_data( ...
                plot_state,stations,tracks);

            // ML-safe observation available after every receiver scan.
            ml_observation = receiver_observation.ml;
            ml_observation.dwell_s = irsa_state.scan_dwell_s;
            ml_observation.scan_lower_mhz = lo_mhz;
            ml_observation.scan_upper_mhz = hi_mhz;

            // Persist the live receiver observation so the external ML workflow
            // can read it while Scilab continues replaying the RF environment.
            receiver_observation.ml.dwell_s = irsa_state.scan_dwell_s;
            receiver_observation.scan_lower_mhz = lo_mhz;
            receiver_observation.scan_upper_mhz = hi_mhz;
            ml_write_observation(BASE_DIR,receiver_observation,CONFIG_ID_CURRENT,step_number);

            // The ML command channel is ready now.  In BASELINE mode it is
            // displayed but does not override the validation sweep.
       

            if modulo(scan_number,PLOT_UPDATE_EVERY_SCAN) == 0 then
                plot_state = update_live_plots( ...
                    plot_state,scan_number,n_total);

                if PLOT_PAUSE_SECONDS > 0 then
                    sleep(PLOT_PAUSE_SECONDS*1000);
                end
            end

            if modulo(scan_number,500) == 0 then
                mprintf( ...
                    "Scan %d | time rows %d-%d | window %.1f-%.1f MHz | RS1=%d RS2=%d RS3=%d | tracks=%d\n", ...
                    scan_number,scan_start_idx,scan_end_idx, ...
                    lo_mhz,hi_mhz,size(p1,1),size(p2,1),size(p3,1),size(tracks,1));
            end
        end

        if scan_number > MAX_SCANS_TO_PROCESS then
            break;
        end

        if scan_end_idx >= n_total then
            break;
        end

        scan_start_idx = scan_end_idx + 1;
    end


    // Final redraw for this configuration.
    plot_state = update_live_plots(plot_state,scan_number,n_total);

    mprintf("\n==============================================\n");
    mprintf(" CONFIGURATION %d COMPLETE\n",CONFIG_ID_CURRENT);
    mprintf("==============================================\n");
    mprintf("Scans/windows processed: %d\n",scan_number);
    mprintf("RF events loaded: %d\n",n_total);
    mprintf("TP=%d  FP=%d  TN=%d  FN=%d\n", ...
            plot_state.TP,plot_state.FP,plot_state.TN,plot_state.FN);
end

// ------------------------------------------------------------
// All configuration replays complete
// ------------------------------------------------------------
mprintf("\n============================================================\n");
mprintf(" SMART SCAN - ALL CONFIGURATIONS COMPLETE\n");
mprintf(" Configurations processed: %d\n",size(CONFIG_IDS,"*"));
mprintf(" Live observation file: %s\n",BASE_DIR + "ml_observation_live.csv");
mprintf("============================================================\n");
mprintf("The ML scheduler will replace EVENT-COVERAGE mode later.\n");
mprintf("Receiver ML interface: ml_observation_live.csv + ml_scan_command.txt\n");
mprintf("ML_INTERFACE_MODE=%s\n",ML_INTERFACE_MODE);
mprintf("virtual_receiver.sci is retained but not executed.\n");
