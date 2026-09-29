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

    // FIX (v2): exactly 27 format codes for the 27 CSV columns, so the last
    // column (hit) is written. The old version had 26 codes and dropped 'hit'.
    // Empty scans contain no receiver measurements, so the measurement
    // fields are NaN and all count/hit fields are zero.
    mfprintf(fd, ...
        "%d,%d,%.9f,%.6f,%.6f,%.6f,%.9f,%.0f,%.0f,%.0f,%.0f,%.6f,%.6f,%.0f,%.6f,%.6f,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.0f\n", ...
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


// ============================================================
// CLOSED-LOOP ML INTERFACE (used by main_closed_loop.sce)
// ------------------------------------------------------------
// Command file format (written by Python online_controller.py):
//     <step> <center_frequency_mhz> <dwell_seconds>
// Scilab scans dwell <step> only after it sees a command whose
// step number matches. Status file tells Python READY / DONE.
// ============================================================

function ml_cl_write_status(status_file,text)
    [fd, ferr] = ml_cl_open_retry(status_file,"wt");
    if ferr == 0 then
        mfprintf(fd,"%s\n",text);
        mclose(fd);
    end
endfunction

function ml_cl_init(obs_file,cmd_file,status_file)
    if isfile(cmd_file) then
        mdelete(cmd_file);
    end
    if isfile(status_file) then
        mdelete(status_file);
    end
    [fd, ferr] = mopen(obs_file,"wt");
    if ferr <> 0 then
        error("Cannot create " + obs_file + " (is it open in Excel?)");
    end
    mfprintf(fd,"config_id,step,scan_time_s,scan_center_mhz,scan_lower_mhz,scan_upper_mhz,dwell_s,total_pdws,station_hit_count,track_count,rs1_pdws,rs1_mean_snr_db,rs1_mean_aoa_deg,rs2_pdws,rs2_mean_snr_db,rs2_mean_aoa_deg,rs3_pdws,rs3_mean_snr_db,rs3_mean_aoa_deg,mean_frequency_mhz,mean_amplitude_dbm,max_amplitude_dbm,mean_snr_db,max_snr_db,mean_pulse_width_us,mean_aoa_deg,hit\n");
    mclose(fd);
endfunction

function [ok,center_mhz,dwell_s,stop_req] = ml_cl_wait_command(cmd_file,expected_step,timeout_s)
    // stop_req = %t when the controller writes step -1 (STOP / RESET from the dashboard)
    ok = %f;
    stop_req = %f;
    center_mhz = %nan;
    dwell_s = %nan;
    t_start = getdate("s");
    while %t
        if isfile(cmd_file) then
            [fd, ferr] = mopen(cmd_file,"rt");
            if ferr == 0 then
                txt = mgetl(fd,1);
                mclose(fd);
                if size(txt,"*") > 0 then
                    vals = msscanf(txt(1),"%d %f %f");
                    if size(vals,"*") >= 3 then
                        if vals(1) == -1 then
                            stop_req = %t;
                            return;
                        end
                        if vals(1) == expected_step then
                            center_mhz = vals(2);
                            dwell_s = vals(3);
                            ok = %t;
                            return;
                        end
                    end
                end
            end
        end
        if getdate("s") - t_start > timeout_s then
            return;
        end
        sleep(2);
    end
endfunction

function [fd, ferr] = ml_cl_open_retry(path, mode)
    // Windows: another program may hold the file for a moment -> retry for up to 2 s
    for k = 1:1000
        [fd, ferr] = mopen(path, mode);
        if ferr == 0 then
            return;
        end
        sleep(2);
    end
endfunction

function ml_cl_write_observation(obs_file,obs,config_id,step,lower_mhz,upper_mhz,dwell_s)
    [fd, ferr] = ml_cl_open_retry(obs_file,"at");
    if ferr <> 0 then return; end
    x = obs.ml;
    mfprintf(fd,"%d,%d,%.9f,%.6f,%.6f,%.6f,%.9f,%d,%d,%d,%d,%.6f,%.6f,%d,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n", ...
        config_id,step, ...
        x.scan_start_s,x.scan_center_mhz,lower_mhz,upper_mhz,dwell_s, ...
        x.receiver_pdw_count,x.station_hit_count,x.track_count, ...
        x.rs1_pdws,x.rs1_mean_snr_db,x.rs1_mean_aoa_deg, ...
        x.rs2_pdws,x.rs2_mean_snr_db,x.rs2_mean_aoa_deg, ...
        x.rs3_pdws,x.rs3_mean_snr_db,x.rs3_mean_aoa_deg, ...
        x.mean_frequency_mhz,x.mean_amplitude_dbm,x.max_amplitude_dbm, ...
        x.mean_snr_db,x.max_snr_db,x.mean_pulse_width_us,x.mean_aoa_deg,bool2s(x.hit));
    mclose(fd);
endfunction

function ml_cl_write_empty_observation(obs_file,config_id,step,scan_time_s,center_mhz,lower_mhz,upper_mhz,dwell_s)
    [fd, ferr] = ml_cl_open_retry(obs_file,"at");
    if ferr <> 0 then return; end
    mfprintf(fd,"%d,%d,%.9f,%.6f,%.6f,%.6f,%.9f,0,0,0,0,Nan,Nan,0,Nan,Nan,0,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,0\n", ...
        config_id,step,scan_time_s,center_mhz,lower_mhz,upper_mhz,dwell_s);
    mclose(fd);
endfunction


// ------------------------------------------------------------
// Extra live data for the dashboard (geolocation + truth flag)
// ------------------------------------------------------------
function ml_cl_init_extra(extra_file,tracks_file)
    [fd, ferr] = mopen(extra_file,"wt");
    if ferr == 0 then
        mfprintf(fd,"step,scan_time_s,gt_present,n_tracks\n");
        mclose(fd);
    end
    [fd, ferr] = mopen(tracks_file,"wt");
    if ferr == 0 then
        mfprintf(fd,"step,scan_time_s,tx_id,n_stations,frequency_mhz,snr_db,x_km,y_km,residual_km,confidence\n");
        mclose(fd);
    end
endfunction

function ml_cl_write_positions(positions_file,stations,tx_cfg)
    // stations: 3 x 2 (km). tx_cfg columns 1..3 = id, x_km, y_km (initial true positions)
    [fd, ferr] = mopen(positions_file,"wt");
    if ferr <> 0 then return; end
    mfprintf(fd,"kind,id,x_km,y_km\n");
    for s = 1:size(stations,1)
        mfprintf(fd,"station,%d,%.6f,%.6f\n",s,stations(s,1),stations(s,2));
    end
    for k = 1:size(tx_cfg,1)
        mfprintf(fd,"emitter,%d,%.6f,%.6f\n",tx_cfg(k,1),tx_cfg(k,2),tx_cfg(k,3));
    end
    mclose(fd);
endfunction

function ml_cl_write_extra(extra_file,tracks_file,step,scan_time_s,gt_present,tracks)
    // Written BEFORE the observation row, so the controller always finds it.
    n = size(tracks,1);
    if n > 0 then
        [fd, ferr] = ml_cl_open_retry(tracks_file,"at");
        if ferr == 0 then
            for k = 1:n
                mfprintf(fd,"%d,%.9f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", ...
                    step,scan_time_s,tracks(k,1),tracks(k,2),tracks(k,3),tracks(k,4), ...
                    tracks(k,5),tracks(k,6),tracks(k,7),tracks(k,8));
            end
            mclose(fd);
        end
    end
    [fd, ferr] = ml_cl_open_retry(extra_file,"at");
    if ferr == 0 then
        mfprintf(fd,"%d,%.9f,%d,%d\n",step,scan_time_s,bool2s(gt_present),n);
        mclose(fd);
    end
endfunction
