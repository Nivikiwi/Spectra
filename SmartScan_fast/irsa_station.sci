function st = irsa_create_station(id, x_km, y_km, rx)
    st = struct();
    st.id = id;
    st.x_km = x_km;
    st.y_km = y_km;
    st.receiver = rx;
    st.scan_count = 0;
    st.last_pdws = [];
endfunction

function st = irsa_station_scan(st, events, scan_start_s, center_mhz)
    // events columns:
    // 1 time_s
    // 2 frequency_mhz
    // 3 pulse_width_us
    // 4 true_AoA_deg (simulator only)
    // 5 amplitude_dbm
    // 6 simulation transmitter ID (internal association only)

    st.scan_count = st.scan_count + 1;
    st.receiver.scan.scan_number = st.scan_count;
    st.receiver.scan.center_frequency_mhz = center_mhz;

    [lo_mhz, hi_mhz] = irsa_rx_window(st.receiver, center_mhz);
    dwell = st.receiver.scan.dwell_time_s;

    if size(events,"*") == 0 then
        st.last_pdws = [];
        return;
    end

    t = events(:,1);
    f = events(:,2);
    pw = events(:,3);
    aoa_true = events(:,4);
    amp = events(:,5);
    txid = events(:,6);

    // Receiver detection threshold.
    snr = amp - st.receiver.detector.noise_floor_dbm;

    idx = find( ...
        t >= scan_start_s & ...
        t <= scan_start_s + dwell & ...
        f >= lo_mhz & ...
        f <= hi_mhz & ...
        snr >= st.receiver.detector.threshold_db);

    if size(idx,"*") == 0 then
        st.last_pdws = [];
        return;
    end

    ids = unique(txid(idx));
    n = min(size(ids,"*"), st.receiver.processing.max_simultaneous_signals);
    pdw = zeros(n,10);

    for k = 1:n
        id = ids(k);
        ii = idx(find(txid(idx) == id));
        [max_amp,jj] = max(amp(ii));
        i = ii(jj);

        measured_snr = amp(i) - st.receiver.detector.noise_floor_dbm;

        measured_aoa = aoa_true(i) + ...
            st.receiver.df.aoa_noise_std_deg * grand(1,1,"nor",0,1);
        measured_aoa = modulo(measured_aoa,360);

        pdw(k,:) = [ ...
            id, ...
            t(i), ...
            f(i), ...
            pw(i), ...
            amp(i), ...
            measured_snr, ...
            measured_aoa, ...
            scan_start_s, ...
            lo_mhz, ...
            hi_mhz ...
        ];
    end

    st.last_pdws = pdw;
endfunction
