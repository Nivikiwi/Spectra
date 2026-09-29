function events = sort_events_by_toa(events)

    [sorted_toa, order] = gsort(events.toa, "g", "i");

    events.toa = sorted_toa;
    events.pulse_id = events.pulse_id(order);
    events.frequency = events.frequency(order);
    events.pulse_width = events.pulse_width(order);
    events.aoa = events.aoa(order);
    events.amplitude = events.amplitude(order);
    events.tx_id = events.tx_id(order);

endfunction

function prev = init_previous_state(max_tx_id)

    prev.valid = zeros(max_tx_id,1) == 1;
    prev.toa = zeros(max_tx_id,1);
    prev.frequency = zeros(max_tx_id,1);
    prev.pulse_width = zeros(max_tx_id,1);
    prev.amplitude = zeros(max_tx_id,1);
    prev.aoa = zeros(max_tx_id,1);
    prev.pri = zeros(max_tx_id,1);

endfunction
