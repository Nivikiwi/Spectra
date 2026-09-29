// ============================================================
// SMART SCAN - LIVE RECEIVER / ML INTERFACE
// ============================================================
// Receiver output: ml_observation_live.csv
// ML command input: ml_scan_command.txt
// Command format: <center_frequency_mhz> <dwell_seconds>
// ============================================================

function ml_init_interface(base_dir)
    out_file = base_dir + "ml_observation_live.csv";
    cmd_file = base_dir + "ml_scan_command.txt";

    fd = mopen(out_file,"wt");
    if fd >= 0 then
        mfprintf(fd,"config_id,step,scan_time_s,scan_center_mhz,scan_lower_mhz,scan_upper_mhz,dwell_s,total_pdws,station_hit_count,track_count,rs1_pdws,rs1_mean_snr_db,rs1_mean_aoa_deg,rs2_pdws,rs2_mean_snr_db,rs2_mean_aoa_deg,rs3_pdws,rs3_mean_snr_db,rs3_mean_aoa_deg,mean_frequency_mhz,mean_amplitude_dbm,max_amplitude_dbm,mean_snr_db,max_snr_db,mean_pulse_width_us,mean_aoa_deg,hit\n");
        mclose(fd);
    end

    if isfile(cmd_file) then
        mdelete(cmd_file);
    end
endfunction

function ml_write_observation(base_dir,obs,config_id,step)
    out_file = base_dir + "ml_observation_live.csv";
    fd = mopen(out_file,"at");
    if fd < 0 then return; end

    x = obs.ml;
    mfprintf(fd,"%d,%d,%.9f,%.6f,%.6f,%.6f,%.9f,%d,%d,%d,%d,%.6f,%.6f,%d,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n", ...
        config_id,step, ...
        x.scan_start_s,x.scan_center_mhz,obs.scan_lower_mhz,obs.scan_upper_mhz,x.dwell_s, ...
        x.receiver_pdw_count,x.station_hit_count,x.track_count, ...
        x.rs1_pdws,x.rs1_mean_snr_db,x.rs1_mean_aoa_deg, ...
        x.rs2_pdws,x.rs2_mean_snr_db,x.rs2_mean_aoa_deg, ...
        x.rs3_pdws,x.rs3_mean_snr_db,x.rs3_mean_aoa_deg, ...
        x.mean_frequency_mhz,x.mean_amplitude_dbm,x.max_amplitude_dbm, ...
        x.mean_snr_db,x.max_snr_db,x.mean_pulse_width_us,x.mean_aoa_deg,x.hit);
    mclose(fd);
endfunction

function [has_command,center_mhz,dwell_s] = ml_read_scan_command(base_dir)
    has_command = %f;
    center_mhz = %nan;
    dwell_s = %nan;
    cmd_file = base_dir + "ml_scan_command.txt";

    if ~isfile(cmd_file) then return; end
    fd = mopen(cmd_file,"rt");
    if fd < 0 then return; end
    line = mgetl(fd,1);
    mclose(fd);
    if size(line,"*") == 0 then return; end

    vals = msscanf(line(1),"%f %f");
    if size(vals,"*") >= 2 then
        center_mhz = vals(1);
        dwell_s = vals(2);
        has_command = %t;
    end
endfunction
function ml_write_empty_observation( ...
    base_dir,config_id,step,scan_time_s,center_mhz,lower_mhz,upper_mhz,dwell_s)

    out_file = base_dir + "ml_observation_live.csv";
    fd = mopen(out_file,"at");

    if fd < 0 then
        return;
    end

    // Always provide exactly 25 numeric values for the 25 CSV columns.
    // Empty scans contain no receiver measurements, so the measurement
    // fields are NaN and all count/hit fields are zero.
    mfprintf(fd, ...
        "%d,%d,%.9f,%.6f,%.6f,%.6f,%.9f,%.0f,%.0f,%.0f,%.0f,%.6f,%.6f,%.0f,%.6f,%.6f,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.0f\n", ...
        config_id,step,scan_time_s, ...
        center_mhz, ...
        lower_mhz, ...
        upper_mhz, ...
        dwell_s, ...
        0,0,0,0, ...
        %nan,%nan, ...
        0,%nan,%nan, ...
        0,%nan,%nan, ...
        %nan,%nan,%nan,%nan,%nan,%nan,%nan, ...
        0);

    mclose(fd);
endfunction
