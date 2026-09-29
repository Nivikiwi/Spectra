function [events, tx_cfg, meta] = load_pulse_dataset(filename, config_id_to_run)

    fd = mopen(filename, "rt");
    if fd < 0 then
        error("Could not open dataset: " + filename);
    end

    // Preallocate for the stated ~1.24M rows.
    MAX_ROWS = 1300000;
    MAX_TX = 1000;

    pulse_id = zeros(MAX_ROWS,1);
    toa = zeros(MAX_ROWS,1);
    frequency = zeros(MAX_ROWS,1);
    pulse_width = zeros(MAX_ROWS,1);
    aoa = zeros(MAX_ROWS,1);
    amplitude = zeros(MAX_ROWS,1);
    tx_id = zeros(MAX_ROWS,1);

    // Transmitter configuration storage.
    tx_seen = zeros(MAX_TX,1);
    tx_cfg_rows = [];

    line = mgetl(fd, 1); // header
    n = 0;

    while ~meof(fd)
        line = mgetl(fd, 1);

        if size(line, "*") == 0 then
            continue;
        end

        cols = strsplit(line, ",");

        // We need at least columns A-I.
        if size(cols, "*") < 9 then
            continue;
        end

        cid = strtod(cols(1));

        if cid <> config_id_to_run then
            continue;
        end

        n = n + 1;

        if n > MAX_ROWS then
            mclose(fd);
            error("Dataset exceeded MAX_ROWS. Increase MAX_ROWS in load_pulse_dataset.sci.");
        end

        pulse_id(n) = strtod(cols(2));
        toa(n) = strtod(cols(3));
        frequency(n) = strtod(cols(4));
        pulse_width(n) = strtod(cols(5));
        aoa(n) = strtod(cols(6));
        amplitude(n) = strtod(cols(7));
        tid = strtod(cols(9));
        tx_id(n) = tid;

        // First occurrence of this transmitter: read geometry/config fields.
        if tid >= 1 & tid <= MAX_TX then
            if tx_seen(tid) == 0 then
                tx_seen(tid) = 1;

                // Y=25, Z=26, AA=27, AB=28, AC=29, AD=30,
                // AE=31, AF=32, AG=33, AH=34.
                if size(cols, "*") >= 34 then
                    row = [ ...
                        tid, ...
                        strtod(cols(25)), ...
                        strtod(cols(26)), ...
                        strtod(cols(27)), ...
                        strtod(cols(28)), ...
                        strtod(cols(29)), ...
                        strtod(cols(30)), ...
                        strtod(cols(31)), ...
                        strtod(cols(32)), ...
                        strtod(cols(33)) ...
                    ];

                    tx_cfg_rows = [tx_cfg_rows; row];
                end
            end
        end
    end

    mclose(fd);

    if n == 0 then
        error("No rows found for config_id = " + string(config_id_to_run));
    end

    // Trim.
    events.pulse_id = pulse_id(1:n);
    events.toa = toa(1:n);
    events.frequency = frequency(1:n);
    events.pulse_width = pulse_width(1:n);
    events.aoa = aoa(1:n);
    events.amplitude = amplitude(1:n);
    events.tx_id = tx_id(1:n);

    tx_cfg = tx_cfg_rows;

    meta.n_events = n;
    meta.n_transmitters = size(tx_cfg,1);
    meta.max_tx_id = max(tx_id(1:n));

    mclose("all");
endfunction


function config_ids = get_dataset_config_ids(filename)
    fd = mopen(filename, "rt");
    if fd < 0 then
        error("Could not open dataset: " + filename);
    end

    ids = [];
    line = mgetl(fd, 1); // header

    while ~meof(fd)
        line = mgetl(fd, 1);
        if size(line, "*") == 0 then
            continue;
        end

        cols = strsplit(line, ",");
        if size(cols, "*") < 1 then
            continue;
        end

        cid = strtod(cols(1));
        if ~isnan(cid) then
            ids = [ids; cid];
        end
    end

    mclose(fd);

    if size(ids, "*") == 0 then
        error("No configuration IDs found in dataset: " + filename);
    end

    config_ids = unique(ids);
endfunction
