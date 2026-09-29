function obs = irsa_build_observation(stations, station_pdws, tracks, scan_start_s, center_mhz)
    obs = struct();

    obs.scan_start_s = scan_start_s;
    obs.scan_center_mhz = center_mhz;
    obs.scan_bandwidth_mhz = 0;
    obs.num_stations = size(stations,1);
    obs.num_station_hits = zeros(size(stations,1),1);
    obs.station_pdws = zeros(size(stations,1),1);
    obs.station_mean_snr_db = %nan*ones(size(stations,1),1);
    obs.station_mean_aoa_deg = %nan*ones(size(stations,1),1);

    all_pdws = [];

    for s = 1:length(station_pdws)
        p = station_pdws(s);
        if size(p,"*") > 0 then
            obs.num_station_hits(s) = size(p,1);
            obs.station_pdws(s) = size(p,1);
            obs.station_mean_snr_db(s) = mean(p(:,6));
            obs.station_mean_aoa_deg(s) = irsa_circular_mean_deg(p(:,7));
            all_pdws = [all_pdws;p];

            if size(p,2) >= 10 then
                obs.scan_bandwidth_mhz = p(1,10) - p(1,9);
            end
        end
    end

    obs.total_receiver_pdws = size(all_pdws,1);
    obs.total_detected_signals = size(tracks,1);
    obs.tracks = tracks;
    obs.hit = (obs.total_receiver_pdws > 0);

    // --------------------------------------------------------
    // ML-safe observation.
    // No simulation transmitter IDs or true positions are copied.
    // --------------------------------------------------------
    obs.ml = struct();
    obs.ml.scan_start_s = scan_start_s;
    obs.ml.scan_center_mhz = center_mhz;
    obs.ml.scan_bandwidth_mhz = obs.scan_bandwidth_mhz;
    obs.ml.dwell_s = 0;
    obs.ml.station_count = obs.num_stations;
    obs.ml.station_hit_count = sum(obs.num_station_hits > 0);
    obs.ml.receiver_pdw_count = obs.total_receiver_pdws;
    obs.ml.rs1_pdws = obs.station_pdws(1);
    obs.ml.rs2_pdws = obs.station_pdws(2);
    obs.ml.rs3_pdws = obs.station_pdws(3);
    obs.ml.rs1_mean_snr_db = obs.station_mean_snr_db(1);
    obs.ml.rs2_mean_snr_db = obs.station_mean_snr_db(2);
    obs.ml.rs3_mean_snr_db = obs.station_mean_snr_db(3);
    obs.ml.rs1_mean_aoa_deg = obs.station_mean_aoa_deg(1);
    obs.ml.rs2_mean_aoa_deg = obs.station_mean_aoa_deg(2);
    obs.ml.rs3_mean_aoa_deg = obs.station_mean_aoa_deg(3);
    obs.ml.track_count = obs.total_detected_signals;
    obs.ml.hit = obs.hit;

    // Per-scan RF summary from receiver PDWs.
    if size(all_pdws,1) > 0 then
        obs.ml.mean_frequency_mhz = mean(all_pdws(:,3));
        obs.ml.mean_amplitude_dbm = mean(all_pdws(:,5));
        obs.ml.max_amplitude_dbm = max(all_pdws(:,5));
        obs.ml.mean_snr_db = mean(all_pdws(:,6));
        obs.ml.max_snr_db = max(all_pdws(:,6));
        obs.ml.mean_pulse_width_us = mean(all_pdws(:,4));
        obs.ml.mean_aoa_deg = irsa_circular_mean_deg(all_pdws(:,7));
    else
        obs.ml.mean_frequency_mhz = %nan;
        obs.ml.mean_amplitude_dbm = %nan;
        obs.ml.max_amplitude_dbm = %nan;
        obs.ml.mean_snr_db = %nan;
        obs.ml.max_snr_db = %nan;
        obs.ml.mean_pulse_width_us = %nan;
        obs.ml.mean_aoa_deg = %nan;
    end

    // Geolocation-derived features are available only when tracks exist.
    if size(tracks,1) > 0 then
        obs.ml.frequency_mhz = tracks(:,3);
        obs.ml.snr_db = tracks(:,4);
        obs.ml.x_km = tracks(:,5);
        obs.ml.y_km = tracks(:,6);
        obs.ml.geolocation_residual_km = tracks(:,7);
        obs.ml.confidence = tracks(:,8);
        obs.ml.mean_bearing_deg = tracks(:,9);
    else
        obs.ml.frequency_mhz = [];
        obs.ml.snr_db = [];
        obs.ml.x_km = [];
        obs.ml.y_km = [];
        obs.ml.geolocation_residual_km = [];
        obs.ml.confidence = [];
        obs.ml.mean_bearing_deg = [];
    end
endfunction
