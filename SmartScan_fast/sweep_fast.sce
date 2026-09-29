// ============================================================
// SMART SCAN - FAST BASELINE SWEEP (configurations 8, 9, 10, 11)
// ============================================================
// Same RF environment, same IRSA receiver calls, same 10-ms dwells and
// 50-MHz windows as main.sce - but no graphs, no pauses, and one file write
// per dwell instead of one per row (main.sce took all night; this takes minutes).
//
// Output: ml_observation_sweep.csv  (same 27 columns as ml_observation_live.csv)
// Your ml_observation_live.csv is NOT touched.
// ============================================================

clear;
clc;

BASE_DIR = get_absolute_file_path("sweep_fast.sce");

SWEEP_CONFIGS = [8 9 10 11];      // 8 = complete re-run (your overnight run stopped at dwell 655)
MAX_DWELLS = %inf;                // for a quick test, e.g. 100

OUT_FILE = BASE_DIR + "ml_observation_sweep.csv";
CSV_FILE = BASE_DIR + "data/whole_dataset_1.24m.csv";
TOA_TO_SECONDS = 1e-6;

exec(BASE_DIR + "load_pulse_dataset.sci",-1);
exec(BASE_DIR + "create_transmitters.sci",-1);
exec(BASE_DIR + "pulse_engine.sci",-1);
exec(BASE_DIR + "rf_environment.sci",-1);
exec(BASE_DIR + "irsa_receiver_config.sci",-1);
exec(BASE_DIR + "irsa_station.sci",-1);
exec(BASE_DIR + "irsa_geolocation.sci",-1);
exec(BASE_DIR + "irsa_observation.sci",-1);
exec(BASE_DIR + "irsa_environment_adapter.sci",-1);

FMT = "%d,%d,%.9f,%.6f,%.6f,%.6f,%.9f,%.0f,%.0f,%.0f,%.0f,%.6f,%.6f,%.0f,%.6f,%.6f,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.0f\n";

[fd, ferr] = mopen(OUT_FILE,"wt");
if ferr <> 0 then
    error("Cannot create " + OUT_FILE + " (is it open in Excel?)");
end
mfprintf(fd,"config_id,step,scan_time_s,scan_center_mhz,scan_lower_mhz,scan_upper_mhz,dwell_s,total_pdws,station_hit_count,track_count,rs1_pdws,rs1_mean_snr_db,rs1_mean_aoa_deg,rs2_pdws,rs2_mean_snr_db,rs2_mean_aoa_deg,rs3_pdws,rs3_mean_snr_db,rs3_mean_aoa_deg,mean_frequency_mhz,mean_amplitude_dbm,max_amplitude_dbm,mean_snr_db,max_snr_db,mean_pulse_width_us,mean_aoa_deg,hit\n");
t_all = getdate("s");

for CONFIG_ID = SWEEP_CONFIGS

    CACHE_FILE = BASE_DIR + "data/pulse_cache_config_" + string(CONFIG_ID) + ".sod";
    if isfile(CACHE_FILE) then
        load(CACHE_FILE,"events","tx_cfg","meta");
    else
        mprintf("\nParsing RF dataset for configuration %d (one time)...\n",CONFIG_ID);
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

    // Same tunable range rule as main.sce
    bw = irsa_state.st1.receiver.rf.instantaneous_bandwidth_mhz;
    rx_min = max(0, floor(min(events.frequency)/bw)*bw);
    rx_max = ceil(max(events.frequency)/bw)*bw;
    irsa_state.st1.receiver.rf.min_frequency_mhz = rx_min;
    irsa_state.st1.receiver.rf.max_frequency_mhz = rx_max;
    irsa_state.st2.receiver.rf.min_frequency_mhz = rx_min;
    irsa_state.st2.receiver.rf.max_frequency_mhz = rx_max;
    irsa_state.st3.receiver.rf.min_frequency_mhz = rx_min;
    irsa_state.st3.receiver.rf.max_frequency_mhz = rx_max;
    n_windows = round((rx_max-rx_min)/bw);
    dwell = irsa_state.scan_dwell_s;

    mprintf("\nConfiguration %d: %d pulse events, %d windows (%.0f-%.0f MHz)\n", ...
            CONFIG_ID,meta.n_events,n_windows,rx_min,rx_max);

    n_total = length(events.time_s);
    scan_start_idx = 1;
    step_number = 0;
    n_dwell = 0;
    t_cfg = getdate("s");

    while scan_start_idx <= n_total & n_dwell < MAX_DWELLS
        n_dwell = n_dwell + 1;
        scan_start_s = events.time_s(scan_start_idx);
        scan_end_s = scan_start_s + dwell;
        scan_end_idx = scan_start_idx;
        while scan_end_idx < n_total & events.time_s(scan_end_idx+1) <= scan_end_s
            scan_end_idx = scan_end_idx + 1;
        end
        r = scan_start_idx:scan_end_idx;
        rf_slice = [events.time_s(r), events.frequency(r), events.pulse_width(r), ...
                    events.aoa(r), events.amplitude(r), events.tx_id(r)];

        window_id = floor((rf_slice(:,2)-rx_min)/bw);
        window_id = min(max(window_id,0),n_windows-1);
        occupied_windows = unique(window_id);

        // ---- empty windows (known negatives), written in one block ----
        empty_wid = setdiff(0:(n_windows-1), occupied_windows);
        ne = size(empty_wid,"*");
        E = [];
        if ne > 0 then
            ew = empty_wid(:);
            lo = rx_min + ew*bw;
            z = zeros(ne,1);
            n = %nan*ones(ne,1);
            E = [CONFIG_ID*ones(ne,1), step_number+(1:ne)', scan_start_s*ones(ne,1), lo+bw/2, lo, lo+bw, ...
                 dwell*ones(ne,1), z, z, z, z, n, n, z, n, n, z, n, n, n, n, n, n, n, n, n, z];
            step_number = step_number + ne;
        end

        // ---- occupied windows: through the IRSA receiver ----
        O = [];
        for w = 1:size(occupied_windows,"*")
            wid = occupied_windows(w);
            lo_mhz = rx_min + wid*bw;
            hi_mhz = lo_mhz + bw;
            center_mhz = lo_mhz + bw/2.0;
            local_idx = find(rf_slice(:,2) >= lo_mhz & rf_slice(:,2) <= hi_mhz);
            if size(local_idx,"*") == 0 then
                continue;
            end
            [irsa_state,obs] = irsa_receiver_scan(irsa_state,rf_slice(local_idx,:),scan_start_s,center_mhz);
            x = obs.ml;
            step_number = step_number + 1;
            O = [O; CONFIG_ID, step_number, x.scan_start_s, x.scan_center_mhz, lo_mhz, hi_mhz, dwell, ...
                 x.receiver_pdw_count, x.station_hit_count, x.track_count, ...
                 x.rs1_pdws, x.rs1_mean_snr_db, x.rs1_mean_aoa_deg, ...
                 x.rs2_pdws, x.rs2_mean_snr_db, x.rs2_mean_aoa_deg, ...
                 x.rs3_pdws, x.rs3_mean_snr_db, x.rs3_mean_aoa_deg, ...
                 x.mean_frequency_mhz, x.mean_amplitude_dbm, x.max_amplitude_dbm, ...
                 x.mean_snr_db, x.max_snr_db, x.mean_pulse_width_us, x.mean_aoa_deg, bool2s(x.hit)];
        end

        M = [E; O];
        if size(M,1) > 0 then
            mfprintf(fd, FMT, M);
        end

        if modulo(n_dwell,200) == 0 then
            mprintf("  config %d: dwell %d, t = %.2f s (%.0f s elapsed)\n",CONFIG_ID,n_dwell,scan_start_s,getdate("s")-t_cfg);
        end
        if scan_end_idx >= n_total then
            break;
        end
        scan_start_idx = scan_end_idx + 1;
    end
    mprintf("Configuration %d done: %d dwells in %.0f s\n",CONFIG_ID,n_dwell,getdate("s")-t_cfg);
end

mclose(fd);
mprintf("\nSWEEP DONE in %.0f s -> %s\n",getdate("s")-t_all,OUT_FILE);
