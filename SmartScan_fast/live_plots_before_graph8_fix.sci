// ============================================================
// SMART SCAN LIVE DASHBOARD
//
// 1. Frequency vs Time              LINE
// 2. Amplitude / Power vs Frequency LINE
// 3. SNR vs Frequency               LINE
// 4. Signal Detection Timeline       LINE
// 5. Pulse Width Distribution       HISTOGRAM
// 6. Inter-Arrival Time              LINE
// 7. Detection Probability vs SNR    LINE
// 8. Receiver Detection Confusion Matrix
// 9. Multi-Station Spatial Geolocation
//
// Axes are fixed for the complete replay.  Only the plotted data
// changes.  The dashboard keeps a rolling history so the plots
// remain visibly populated during live replay.
// ============================================================

function init_live_plot_windows()

    for i = 1:9
        scf(i);
        clf();
        gcf().immediate_drawing = "on";
        gcf().visible = "on";
    end

    scf(1); gcf().figure_name = "Smart Scan - Frequency vs Time";
    scf(2); gcf().figure_name = "Smart Scan - Received Power Spectrum";
    scf(3); gcf().figure_name = "Smart Scan - SNR vs Frequency";
    scf(4); gcf().figure_name = "Smart Scan - Signal Detection Timeline";
    scf(5); gcf().figure_name = "Smart Scan - Pulse Width Distribution";
    scf(6); gcf().figure_name = "Smart Scan - Inter-Arrival Time";
    scf(7); gcf().figure_name = "Smart Scan - Detection Probability vs SNR";
    scf(8); gcf().figure_name = "Smart Scan - Confusion Matrix";
    scf(9); gcf().figure_name = "Smart Scan - Spatial Geolocation";

endfunction

function state = init_live_plots(history_length)
    state.max_len = history_length;
    state.n = 0;
    state.timeline_n = 0;

    state.time_s = %nan * ones(history_length,1);
    state.frequency = %nan * ones(history_length,1);
    state.amplitude = %nan * ones(history_length,1);
    state.snr = %nan * ones(history_length,1);
    state.pw = %nan * ones(history_length,1);
    state.aoa = %nan * ones(history_length,1);

    // Persistent frequency-domain profiles.  These are cumulative over
    // the replay so the spectrum plots do not disappear when the rolling
    // PDW history moves to another part of the spectrum.
    state.freq_bin_width_mhz = 50;
    state.freq_profile_ready = %f;
    state.freq_amp_max = [];
    state.freq_snr_max = [];
    state.freq_bin_centers = [];

    // One sample for EVERY receiver scan.  This makes the detection
    // timeline a real 0/1 line instead of a sparse list of hits.
    state.timeline_time_s = %nan * ones(history_length,1);
    state.timeline_detected = zeros(history_length,1);

    // 0-60 dB SNR, 5 dB bins.
    state.snr_bins = 0:5:60;
    state.snr_total = zeros(length(state.snr_bins)-1,1);
    state.snr_detected = zeros(length(state.snr_bins)-1,1);

    state.TP = 0;
    state.FP = 0;
    state.TN = 0;
    state.FN = 0;

    state.stations = [];
    state.tracks = [];
    state.track_history = [];

    // Live receiver -> ML observation table.

    // Fixed axis values.  main.sce overwrites these with the
    // complete-dataset limits after the dataset is loaded.
    state.axis = struct();
    state.axis.time_min = 0;
    state.axis.time_max = 1;
    state.axis.freq_min = 100;
    state.axis.freq_max = 12000;
    state.axis.amp_floor_dbm = -95;
    state.axis.amp_min = -100;
    state.axis.amp_max = 0;
    state.axis.snr_min = 0;
    state.axis.snr_max = 60;
    state.axis.pw_min = 0;
    state.axis.pw_max = 100;
    state.axis.dt_min = 0;
    state.axis.dt_max_us = 1000;
    state.axis.spatial_x_min = 0;
    state.axis.spatial_x_max = 110;
    state.axis.spatial_y_min = -10;
    state.axis.spatial_y_max = 90;
    state.spatial_first_draw = %t;

    // Persistent graphics handles.  The live dashboard creates each
    // curve ONCE and then updates its .data field.  This avoids clf()
    // during replay and prevents visible blank/flash frames.
    state.gfx_ready = %f;
    state.h_ft = [];
    state.h_ft_marker = [];
    state.h_amp = [];
    state.h_amp_marker = [];
    state.h_snr = [];
    state.h_snr_marker = [];
    state.h_timeline = [];
    state.h_timeline_marker = [];
    state.h_pw = [];
    state.h_dt = [];
    state.h_pd = [];
    state.h_cm = [];
    state.h_cm_num = [];
    state.h_sp_hist = [];
    state.h_sp_current = [];
    state.h_sp_stations = [];
endfunction

function state = init_frequency_profile(state,freq_min,freq_max)
    state.freq_bin_width_mhz = 50;
    nb = max(1,ceil((freq_max-freq_min)/state.freq_bin_width_mhz));
    state.freq_bin_centers = freq_min + ...
        (0:nb-1)'*state.freq_bin_width_mhz + ...
        state.freq_bin_width_mhz/2;
    state.freq_amp_max = %nan*ones(nb,1);
    state.freq_snr_max = %nan*ones(nb,1);
    state.freq_profile_ready = %t;
endfunction

function state = append_receiver_pdw(state,pdw,time_s)
    if size(pdw,"*") == 0 then
        return;
    end

    if state.n < state.max_len then
        state.n = state.n + 1;
        idx = state.n;
    else
        state.time_s(1:$-1) = state.time_s(2:$);
        state.frequency(1:$-1) = state.frequency(2:$);
        state.amplitude(1:$-1) = state.amplitude(2:$);
        state.snr(1:$-1) = state.snr(2:$);
        state.pw(1:$-1) = state.pw(2:$);
        state.aoa(1:$-1) = state.aoa(2:$);
        idx = state.max_len;
    end

    // PDW columns:
    // 1 ID, 2 ToA, 3 frequency, 4 PW, 5 amplitude,
    // 6 SNR, 7 measured AoA, 8 scan start, 9 lower BW, 10 upper BW.
    state.time_s(idx) = time_s;
    state.frequency(idx) = pdw(3);
    state.pw(idx) = pdw(4);
    state.amplitude(idx) = pdw(5);
    state.snr(idx) = pdw(6);
    state.aoa(idx) = pdw(7);

    // Update persistent spectrum envelope.
    if state.freq_profile_ready then
        b = floor((pdw(3) - (state.freq_bin_centers(1) - state.freq_bin_width_mhz/2)) / ...
                  state.freq_bin_width_mhz) + 1;
        if b >= 1 & b <= length(state.freq_amp_max) then
            if isnan(state.freq_amp_max(b)) then
                state.freq_amp_max(b) = pdw(5);
                state.freq_snr_max(b) = pdw(6);
            else
                state.freq_amp_max(b) = max(state.freq_amp_max(b),pdw(5));
                state.freq_snr_max(b) = max(state.freq_snr_max(b),pdw(6));
            end
        end
    end
endfunction


function state = append_detection_scan(state,time_s,detected)
    if state.timeline_n < state.max_len then
        state.timeline_n = state.timeline_n + 1;
        idx = state.timeline_n;
    else
        state.timeline_time_s(1:$-1) = state.timeline_time_s(2:$);
        state.timeline_detected(1:$-1) = state.timeline_detected(2:$);
        idx = state.max_len;
    end

    state.timeline_time_s(idx) = time_s;
    state.timeline_detected(idx) = detected;
endfunction

function state = update_spatial_data(state,stations,tracks)
    state.stations = stations;
    state.tracks = tracks;

    // Retain recent estimated positions so the spatial graph does not
    // disappear between scans when one scan has no valid triangulation.
    if size(tracks,"*") > 0 then
        valid = find(~isnan(tracks(:,5)) & ~isnan(tracks(:,6)) & ...
                     ~isinf(tracks(:,5)) & ~isinf(tracks(:,6)));
        if size(valid,"*") > 0 then
            state.track_history = [state.track_history; tracks(valid,5:6)];
            max_track_points = 2000;
            if size(state.track_history,1) > max_track_points then
                state.track_history = state.track_history($-max_track_points+1:$,:);
            end
        end
    end
endfunction

function [x,y] = finite_xy(x0,y0)
    x = [];
    y = [];
    if size(x0,"*") == 0 then
        return;
    end
    good = find(~isnan(x0) & ~isnan(y0) & ~isinf(x0) & ~isinf(y0));
    if size(good,"*") > 0 then
        x = x0(good);
        y = y0(good);
    end
endfunction

function set_fixed_bounds(ax,xmin,ymin,xmax,ymax)
    ax.data_bounds = [xmin ymin; xmax ymax];
endfunction

function r = rowvec(v)
    if size(v,"*") == 0 then
        r = [];
    else
        r = matrix(v,1,-1);
    end
endfunction


function state = update_live_plots(state,pulse_number,total_pulses)

    n = state.n;
    tn = state.timeline_n;

    // ============================================================
    // CREATE GRAPHICS OBJECTS ONCE
    // ============================================================
    // Scilab 2026 polyline entities expose a .data property containing
    // [x y]. Updating that property is much smoother than clearing and
    // recreating the complete figure on every receiver scan.
    if ~state.gfx_ready then

        // 1. Frequency vs Time
        scf(1);
        a = gca();
        a.auto_clear = "off";
        a.auto_scale = "off";
        a.data_bounds = [state.axis.time_min state.axis.freq_min; ...
                         state.axis.time_max state.axis.freq_max];
        state.h_ft = plot([state.axis.time_min state.axis.time_min], ...
                          [state.axis.freq_min state.axis.freq_min]);
        state.h_ft_marker = plot(state.axis.time_min,state.axis.freq_min,"o");
        xtitle("Frequency vs Time","Time (s)","Frequency (MHz)");
        xgrid();

        // 2. Received power spectrum
        scf(2);
        a = gca();
        a.auto_clear = "off";
        a.auto_scale = "off";
        a.data_bounds = [state.axis.freq_min state.axis.amp_min; ...
                         state.axis.freq_max state.axis.amp_max];
        state.h_amp = plot(matrix(state.freq_bin_centers,-1,1), ...
                           matrix(state.axis.amp_floor_dbm*ones(length(state.freq_bin_centers),1),-1,1));
        state.h_amp_marker = plot(state.axis.freq_min,state.axis.amp_floor_dbm,"o");
        xtitle("Received Power Spectrum","Frequency (MHz)","Peak Received Power (dBm)");
        xgrid();

        // 3. SNR vs frequency
        scf(3);
        a = gca();
        a.auto_clear = "off";
        a.auto_scale = "off";
        a.data_bounds = [state.axis.freq_min state.axis.snr_min; ...
                         state.axis.freq_max state.axis.snr_max];
        state.h_snr = plot(matrix(state.freq_bin_centers,-1,1),matrix(zeros(length(state.freq_bin_centers),1),-1,1));
        state.h_snr_marker = plot(state.axis.freq_min,0,"o");
        xtitle("Received SNR Spectrum","Frequency (MHz)","Peak SNR (dB)");
        xgrid();

        // 4. Detection timeline
        scf(4);
        a = gca();
        a.auto_clear = "off";
        a.auto_scale = "off";
        a.data_bounds = [state.axis.time_min -0.05; ...
                         state.axis.time_max 1.05];
        state.h_timeline = plot([state.axis.time_min state.axis.time_max],[0 0]);
        state.h_timeline_marker = plot(state.axis.time_min,0,"o");
        xtitle("Signal Detection Timeline","Time (s)","Detection (0/1)");
        xgrid();

        // 5. Pulse-width histogram
        scf(5);
        a = gca();
        a.auto_clear = "off";
        a.auto_scale = "off";
        a.data_bounds = [state.axis.pw_min 0; ...
                         state.axis.pw_max max(10,ceil(state.max_len/10))];
        state.h_pw = [];
        xtitle("Pulse Width Distribution","Pulse Width (us)","Count");
        xgrid();

        // 6. Inter-arrival time
        scf(6);
        a = gca();
        a.auto_clear = "off";
        a.auto_scale = "off";
        a.data_bounds = [0 0; max(2,n-1) state.axis.dt_max_us];
        state.h_dt = plot([1 2],[0 0]);
        xtitle("Inter-Arrival Time / Pulse Interval","Pulse","Interval (us)");
        xgrid();

        // 7. Detection probability
        scf(7);
        a = gca();
        a.auto_clear = "off";
        a.auto_scale = "off";
        a.data_bounds = [state.axis.snr_min 0; state.axis.snr_max 1.05];
        snr_centers = (state.snr_bins(1:$-1)+state.snr_bins(2:$))/2;
        state.h_pd = plot(matrix(snr_centers,-1,1),matrix(zeros(length(snr_centers),1),-1,1));
        xtitle("Detection Probability vs SNR","SNR (dB)","Probability of Detection");
        xgrid();

        // 8. Confusion matrix
        scf(8);
        clf();
        f = gcf();
        f.color_map = [ ...
            0.75 0.90 0.75;
            1.00 0.85 0.65;
            1.00 0.75 0.75;
            0.55 0.85 0.65];
        a = gca();
        a.data_bounds = [0 0; 2 2];
        a.axes_visible = "off";
        state.h_cm = Matplot1([1 2;3 4],[0,0,2,2]);
        xstring(0.43,1.62,"TN");
        xstring(1.43,1.62,"FP");
        xstring(0.43,0.62,"FN");
        xstring(1.43,0.62,"TP");
        xstring(0.40,1.38,"0");
        xstring(1.40,1.38,"0");
        xstring(0.40,0.38,"0");
        xstring(1.40,0.38,"0");
        xstring(0.48,1.93,"Predicted 0");
        xstring(1.48,1.93,"Predicted 1");
        xstring(0.02,1.40,"Actual 0");
        xstring(0.02,0.40,"Actual 1");
        xtitle("Receiver Detection Confusion Matrix","","");
        state.h_cm_num = xstring([0.40 1.40 0.40 1.40], ...
                                  [1.38 1.38 0.38 0.38], ...
                                  ["0" "0" "0" "0"]);

        // 9. Spatial geolocation
        scf(9);
        a = gca();
        a.auto_clear = "off";
        a.auto_scale = "off";
        a.data_bounds = [state.axis.spatial_x_min state.axis.spatial_y_min; ...
                         state.axis.spatial_x_max state.axis.spatial_y_max];
        state.h_sp_stations = [];
        state.h_sp_hist = [];
        state.h_sp_current = [];
        xtitle("Multi-Station Spatial Geolocation","X Position (km)","Y Position (km)");
        xgrid();

        state.gfx_ready = %t;
    end

    // ============================================================
    // UPDATE 1: FREQUENCY VS TIME
    // ============================================================
    scf(1);
    if n > 0 then
        [x,y] = finite_xy(state.time_s(1:n),state.frequency(1:n));
        if size(x,"*") > 0 then
            [xs,ord] = gsort(x,"g","i");
            state.h_ft.data = [matrix(xs,-1,1) matrix(y(ord),-1,1)];
            state.h_ft_marker.data = [xs($) y(ord($))];
        end
    end

    // ============================================================
    // UPDATE 2: POWER
    // ============================================================
    scf(2);
    amp_display = state.freq_amp_max;
    if length(amp_display) > 0 then
        for k = 1:length(amp_display)
            if isnan(amp_display(k)) | isinf(amp_display(k)) then
                amp_display(k) = state.axis.amp_floor_dbm;
            end
        end
        state.h_amp.data = [matrix(state.freq_bin_centers,-1,1) matrix(amp_display,-1,1)];
        if n > 0 & ~isnan(state.frequency(n)) then
            bb = floor((state.frequency(n) - ...
                (state.freq_bin_centers(1)-state.freq_bin_width_mhz/2)) / ...
                state.freq_bin_width_mhz) + 1;
            bb = max(1,min(length(amp_display),bb));
            state.h_amp_marker.data = [state.frequency(n) amp_display(bb)];
        end
    end

    // ============================================================
    // UPDATE 3: SNR
    // ============================================================
    scf(3);
    snr_display = state.freq_snr_max;
    if length(snr_display) > 0 then
        for k = 1:length(snr_display)
            if isnan(snr_display(k)) | isinf(snr_display(k)) then
                snr_display(k) = 0;
            end
        end
        state.h_snr.data = [matrix(state.freq_bin_centers,-1,1) matrix(snr_display,-1,1)];
        if n > 0 & ~isnan(state.frequency(n)) then
            bb = floor((state.frequency(n) - ...
                (state.freq_bin_centers(1)-state.freq_bin_width_mhz/2)) / ...
                state.freq_bin_width_mhz) + 1;
            bb = max(1,min(length(snr_display),bb));
            state.h_snr_marker.data = [state.frequency(n) snr_display(bb)];
        end
    end

    // ============================================================
    // UPDATE 4: DETECTION TIMELINE
    // ============================================================
    scf(4);
    if tn > 0 then
        state.h_timeline.data = [matrix(state.timeline_time_s(1:tn),-1,1) ...
                                 matrix(state.timeline_detected(1:tn),-1,1)];
        state.h_timeline_marker.data = ...
            [state.timeline_time_s(tn) state.timeline_detected(tn)];
    end

    // ============================================================
    // UPDATE 5: PULSE-WIDTH HISTOGRAM
    // ============================================================
    // Only the histogram bars are replaced; the figure itself is NOT
    // cleared. This prevents the white-frame flash seen in the video.
    scf(5);
    pw = state.pw(1:n);
    pw = pw(~isnan(pw) & ~isinf(pw));
    if size(pw,"*") > 0 then
        nbins = 20;
        edges = linspace(state.axis.pw_min,state.axis.pw_max,nbins+1);
        counts = zeros(nbins,1);
        for k = 1:size(pw,"*")
            b = floor((pw(k)-state.axis.pw_min) / ...
                (state.axis.pw_max-state.axis.pw_min) * nbins) + 1;
            b = max(1,min(nbins,b));
            counts(b) = counts(b) + 1;
        end
        centers = (edges(1:$-1)+edges(2:$))/2;
        if size(state.h_pw,"*") == 0 then
            state.h_pw = bar(centers,counts,1.0);
        else
            delete(state.h_pw);
            state.h_pw = bar(centers,counts,1.0);
        end
    end

    // ============================================================
    // UPDATE 6: INTER-ARRIVAL
    // ============================================================
    scf(6);
    if n > 1 then
        t = state.time_s(1:n);
        goodt = find(~isnan(t) & ~isinf(t));
        if size(goodt,"*") > 1 then
            tv = t(goodt);
            dt = diff(tv)*1e6;
            good = find(dt >= 0 & dt <= state.axis.dt_max_us);
            if size(good,"*") > 0 then
                state.h_dt.data = [matrix(good,-1,1) matrix(dt(good),-1,1)];
            end
        end
    end

    // ============================================================
    // UPDATE 7: DETECTION PROBABILITY
    // ============================================================
    scf(7);
    probability = zeros(length(state.snr_total),1);
    for i = 1:length(probability)
        if state.snr_total(i) > 0 then
            probability(i) = state.snr_detected(i)/state.snr_total(i);
        end
    end
    snr_centers = (state.snr_bins(1:$-1)+state.snr_bins(2:$))/2;
    state.h_pd.data = [matrix(snr_centers,-1,1) matrix(probability,-1,1)];

    // ============================================================
    // UPDATE 8: CONFUSION MATRIX
    // ============================================================
    // IMPORTANT: Matplot1 graphics objects are not updated through a
    // generic .data assignment here.  The previous implementation
    // stopped at this line, so the final drawnow() loop was never
    // reached and ALL nine figures stayed blank.
    //
    // Only figure 8 is rebuilt.  The other eight figures remain
    // persistent, so this does not cause the dashboard-wide flashing
    // problem.
    scf(8);
    clf();
    f = gcf();
    f.color_map = [ ...
        0.75 0.90 0.75; ...
        1.00 0.85 0.65; ...
        1.00 0.75 0.75; ...
        0.55 0.85 0.65];
    a = gca();
    a.data_bounds = [0 0; 2 2];
    a.axes_visible = "off";
    Matplot1([1 2;3 4],[0,0,2,2]);

    xstring(0.38,1.55,"TN");
    xstring(1.38,1.55,"FP");
    xstring(0.38,0.55,"FN");
    xstring(1.38,0.55,"TP");

    xstring(0.38,1.25,string(state.TN));
    xstring(1.38,1.25,string(state.FP));
    xstring(0.38,0.25,string(state.FN));
    xstring(1.38,0.25,string(state.TP));

    xtitle("Receiver Detection Confusion Matrix",""," ");

    // ============================================================
    // UPDATE 9: SPATIAL GEOLOCATION
    // ============================================================
    scf(9);

    if size(state.h_sp_stations,"*") == 0 & size(state.stations,"*") > 0 then
        state.h_sp_stations = plot(state.stations(:,1),state.stations(:,2),"o");
    end

    if size(state.track_history,"*") > 0 then
        [hx,hy] = finite_xy(state.track_history(:,1),state.track_history(:,2));
        if size(hx,"*") > 0 then
            if size(state.h_sp_hist,"*") == 0 then
                state.h_sp_hist = plot(hx,hy,".");
            else
                state.h_sp_hist.data = [matrix(hx,-1,1) matrix(hy,-1,1)];
            end
        end
    end

    if size(state.tracks,"*") > 0 then
        valid = find(~isnan(state.tracks(:,5)) & ...
                     ~isnan(state.tracks(:,6)) & ...
                     ~isinf(state.tracks(:,5)) & ...
                     ~isinf(state.tracks(:,6)) & ...
                     state.tracks(:,5) >= state.axis.spatial_x_min & ...
                     state.tracks(:,5) <= state.axis.spatial_x_max & ...
                     state.tracks(:,6) >= state.axis.spatial_y_min & ...
                     state.tracks(:,6) <= state.axis.spatial_y_max);
        if size(valid,"*") > 0 then
            cx = state.tracks(valid,5);
            cy = state.tracks(valid,6);
            if size(state.h_sp_current,"*") == 0 then
                state.h_sp_current = plot(cx,cy,"o");
            else
                state.h_sp_current.data = [matrix(cx,-1,1) matrix(cy,-1,1)];
            end
        end
    end

    // ============================================================
    // COMMIT ONE FRAME
    // ============================================================
    // IMPORTANT: drawnow() only renders the CURRENT figure.  The old
    // implementation turned all nine figures back on and then called
    // drawnow() once while figure 9 was current.  Figures 1-8 therefore
    // remained hidden.  Commit each figure explicitly.
    for i = 1:9
        scf(i);
        gcf().immediate_drawing = "on";
        drawnow();
    end

    // Leave all figures in immediate-drawing mode.  This is safer for
    // Scilab 2026.x than toggling the property off after every frame.
    for i = 1:9
        scf(i);
        gcf().immediate_drawing = "on";
    end

endfunction
