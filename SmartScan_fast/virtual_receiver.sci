function obs = make_pulse_observation( ...
        time_s, tx_id, frequency_mhz, pulse_width_us, aoa_deg, ...
        amplitude_dbm, noise_floor_dbm, snr_db, pri_us, ...
        d_freq_mhz, d_pri_us, d_pw_us, d_amp_db, d_aoa_deg, has_previous)

    obs.time_s = time_s;
    obs.tx_id = tx_id;

    obs.frequency_mhz = frequency_mhz;
    obs.pulse_width_us = pulse_width_us;
    obs.aoa_deg = aoa_deg;
    obs.amplitude_dbm = amplitude_dbm;

    obs.noise_floor_dbm = noise_floor_dbm;
    obs.snr_db = snr_db;

    obs.pri_us = pri_us;

    obs.d_frequency_mhz = d_freq_mhz;
    obs.d_pri_us = d_pri_us;
    obs.d_pulse_width_us = d_pw_us;
    obs.d_amplitude_db = d_amp_db;
    obs.d_aoa_deg = d_aoa_deg;

    obs.has_previous = has_previous;
endfunction
