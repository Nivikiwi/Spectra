// ============================================================
// IRSA ENVIRONMENT ADAPTER
// ============================================================
// Converts the Scilab RF-environment event stream into
// station-specific receiver inputs.
//
// transmitter_id > 0 : emitter / signal
// transmitter_id = 0 : background/noise event
//
// Ground-truth geometry is used internally only to simulate
// station bearing. It is never copied into obs.ml.
// ============================================================

function state = irsa_receiver_init(tx)
    state = struct();

    rx1 = irsa_create_receiver();
    rx2 = irsa_create_receiver();
    rx3 = irsa_create_receiver();

    state.st1 = irsa_create_station(1,0,0,rx1);
    state.st2 = irsa_create_station(2,100,0,rx2);
    state.st3 = irsa_create_station(3,50,80,rx3);

    state.stations = [ ...
        state.st1.x_km state.st1.y_km;
        state.st2.x_km state.st2.y_km;
        state.st3.x_km state.st3.y_km ...
    ];

    state.tx = tx;
    state.scan_center_mhz = 227.5;
    state.scan_dwell_s = rx1.scan.dwell_time_s;
    state.scan_count = 0;
    state.last_observation = [];
endfunction

function [x_km,y_km] = irsa_tx_position_at_time(tx,tid,time_s)
    ii = find(tx.id == tid);

    if size(ii,"*") == 0 then
        error("IRSA receiver: transmitter ID " + string(tid) + " not found.");
    end

    i = ii(1);
    angle_rad = tx.travel_angle_deg(i) * %pi / 180.0;

    x_km = tx.x0(i) + tx.speed(i) * time_s * cos(angle_rad);
    y_km = tx.y0(i) + tx.speed(i) * time_s * sin(angle_rad);
endfunction

function e = irsa_make_station_event(tx,tid,time_s,frequency_mhz,pw_us,aoa_source_deg,amp_dbm,station)

    // Background/noise rows have no transmitter geometry.
    // Keep their source AoA so they can produce a possible
    // receiver false alarm, but never use them for geolocation.
    if tid <= 0 then
        bearing = modulo(aoa_source_deg,360.0);
    else
        [tx_x,tx_y] = irsa_tx_position_at_time(tx,tid,time_s);
        bearing = irsa_bearing_deg( ...
            station.x_km,station.y_km,tx_x,tx_y);
    end

    e = [time_s frequency_mhz pw_us bearing amp_dbm tid];
endfunction

function e = irsa_build_station_event_batch(events,tx,station)
    if size(events,"*") == 0 then
        e = [];
        return;
    end

    e = zeros(size(events,1),6);

    for k = 1:size(events,1)
        e(k,:) = irsa_make_station_event( ...
            tx,events(k,6),events(k,1),events(k,2), ...
            events(k,3),events(k,4),events(k,5),station);
    end
endfunction

function [state,obs] = irsa_receiver_scan(state,rf_events,scan_start_s,center_mhz)
    state.scan_center_mhz = center_mhz;
    state.scan_count = state.scan_count + 1;

    e1 = irsa_build_station_event_batch(rf_events,state.tx,state.st1);
    e2 = irsa_build_station_event_batch(rf_events,state.tx,state.st2);
    e3 = irsa_build_station_event_batch(rf_events,state.tx,state.st3);

    state.st1 = irsa_station_scan(state.st1,e1,scan_start_s,center_mhz);
    state.st2 = irsa_station_scan(state.st2,e2,scan_start_s,center_mhz);
    state.st3 = irsa_station_scan(state.st3,e3,scan_start_s,center_mhz);

    pdws = list( ...
        state.st1.last_pdws, ...
        state.st2.last_pdws, ...
        state.st3.last_pdws);

    tracks = irsa_fuse_stations(state.stations,pdws);

    obs = irsa_build_observation( ...
        state.stations,pdws,tracks,scan_start_s,center_mhz);

    obs.ml.dwell_s = state.scan_dwell_s;
    state.last_observation = obs;
endfunction
