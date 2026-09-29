function tx = create_transmitters(tx_cfg)

    n = size(tx_cfg,1);

    tx.id = zeros(n,1);
    tx.x0 = zeros(n,1);
    tx.y0 = zeros(n,1);
    tx.speed = zeros(n,1);
    tx.travel_angle_deg = zeros(n,1);
    tx.power_w = zeros(n,1);
    tx.gain_db = zeros(n,1);
    tx.beam_width_deg = zeros(n,1);
    tx.scan_rate_rpm = zeros(n,1);
    tx.scan_start_angle_deg = zeros(n,1);

    for k = 1:n
        tx.id(k) = tx_cfg(k,1);
        tx.x0(k) = tx_cfg(k,2);
        tx.y0(k) = tx_cfg(k,3);
        tx.speed(k) = tx_cfg(k,4);
        tx.travel_angle_deg(k) = tx_cfg(k,5);
        tx.power_w(k) = tx_cfg(k,6);
        tx.gain_db(k) = tx_cfg(k,7);
        tx.beam_width_deg(k) = tx_cfg(k,8);
        tx.scan_rate_rpm(k) = tx_cfg(k,9);
        tx.scan_start_angle_deg(k) = tx_cfg(k,10);
    end

    tx.n = n;
endfunction
