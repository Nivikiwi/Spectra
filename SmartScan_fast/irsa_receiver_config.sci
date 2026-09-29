function rx = irsa_create_receiver()
    rx = struct();
    rx.profile = "HETEROGENEOUS_SDR";

    rx.rf = struct();
    rx.rf.min_frequency_mhz = 100.0;
    rx.rf.max_frequency_mhz = 500.0;
    rx.rf.instantaneous_bandwidth_mhz = 50.0;
    rx.rf.frequency_resolution_mhz = 0.1;

    rx.adc = struct();
    rx.adc.sample_rate_msps = 125.0;
    rx.adc.bits = 14;

    rx.detector = struct();
    rx.detector.noise_floor_dbm = -95.0;
    rx.detector.threshold_db = 10.0;
    rx.detector.detection_level_dbm = -85.0;

    rx.df = struct();
    rx.df.enabled = %t;
    rx.df.aoa_resolution_deg = 1.0;
    rx.df.aoa_noise_std_deg = 1.0;

    rx.processing = struct();
    rx.processing.max_simultaneous_signals = 16;
    rx.processing.max_pdws_per_scan = 10000;

    rx.scan = struct();
    rx.scan.center_frequency_mhz = 250.0;
    rx.scan.dwell_time_s = 0.010;
    rx.scan.scan_number = 0;
endfunction

function [lo_mhz, hi_mhz] = irsa_rx_window(rx, center_mhz)
    bw = rx.rf.instantaneous_bandwidth_mhz;
    lo_mhz = max(rx.rf.min_frequency_mhz, center_mhz - bw/2.0);
    hi_mhz = min(rx.rf.max_frequency_mhz, center_mhz + bw/2.0);
endfunction
