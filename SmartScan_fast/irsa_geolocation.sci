function b = irsa_bearing_deg(x1,y1,x2,y2)
    b = atan(y2-y1,x2-x1) * 180/%pi;
    b = modulo(b + 360.0,360.0);
endfunction

function [x,y,residual] = irsa_triangulate_bearings(stations,bearings)
    m = size(stations,1);

    if m < 2 then
        x = %nan;
        y = %nan;
        residual = %inf;
        return;
    end

    A = zeros(m,2);
    rhs = zeros(m,1);

    for i = 1:m
        th = bearings(i)*%pi/180.0;
        nx = -sin(th);
        ny = cos(th);
        A(i,1) = nx;
        A(i,2) = ny;
        rhs(i) = nx*stations(i,1) + ny*stations(i,2);
    end

    ATA = A'*A;

    if abs(det(ATA)) < 1e-10 then
        x = %nan;
        y = %nan;
        residual = %inf;
        return;
    end

    p = ATA \ (A'*rhs);
    x = p(1);
    y = p(2);

    err = A*p-rhs;
    residual = sqrt(mean(err.^2));
endfunction

function b = irsa_circular_mean_deg(angles)
    if size(angles,"*") == 0 then
        b = %nan;
        return;
    end

    a = angles * %pi / 180.0;
    s = mean(sin(a));
    c = mean(cos(a));
    b = modulo(atan(s,c)*180/%pi + 360.0,360.0);
endfunction

function tracks = irsa_fuse_stations(stations,station_pdws)
    // Internal simulator association only.
    // ID 0 is background/noise and is never geolocated.

    all_ids = [];

    for s = 1:length(station_pdws)
        p = station_pdws(s);
        if size(p,"*") > 0 then
            ii = find(p(:,1) > 0);
            if size(ii,"*") > 0 then
                all_ids = [all_ids;p(ii,1)];
            end
        end
    end

    if size(all_ids,"*") == 0 then
        tracks = [];
        return;
    end

    ids = unique(all_ids);
    tracks = zeros(size(ids,"*"),9);

    for k = 1:size(ids,"*")
        id = ids(k);
        pts = [];
        bearings = [];
        freqs = [];
        snrs = [];

        for s = 1:length(station_pdws)
            p = station_pdws(s);

            if size(p,"*") == 0 then
                continue;
            end

            ii = find(p(:,1) == id);

            if size(ii,"*") > 0 then
                j = ii(1);
                pts = [pts;stations(s,1),stations(s,2)];
                bearings = [bearings;p(j,7)];
                freqs = [freqs;p(j,3)];
                snrs = [snrs;p(j,6)];
            end
        end

        if size(pts,1) >= 2 then
            [x,y,res] = irsa_triangulate_bearings(pts,bearings);
            confidence = 1.0/(1.0+res);
        else
            x = %nan;
            y = %nan;
            res = %inf;
            confidence = 0.0;
        end

        tracks(k,:) = [ ...
            id, ...
            size(pts,1), ...
            mean(freqs), ...
            mean(snrs), ...
            x, ...
            y, ...
            res, ...
            confidence, ...
            irsa_circular_mean_deg(bearings) ...
        ];
    end
endfunction
