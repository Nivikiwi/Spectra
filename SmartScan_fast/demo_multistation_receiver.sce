clear;
clc;

exec("irsa_receiver_config.sci",-1);
exec("irsa_station.sci",-1);
exec("irsa_geolocation.sci",-1);
exec("irsa_observation.sci",-1);

// ============================================================
// IRSA MULTI-STATION SDR RECEIVER DEMO
// ============================================================

// Three receiving stations.
rx1=irsa_create_receiver();
rx2=irsa_create_receiver();
rx3=irsa_create_receiver();

st1=irsa_create_station(1,0,0,rx1);
st2=irsa_create_station(2,100,0,rx2);
st3=irsa_create_station(3,50,80,rx3);

// ------------------------------------------------------------
// Four simultaneous transmitters
// ------------------------------------------------------------

txpos=[35 25;
       65 35;
       55 55;
       82 18];

txfreq=[210;225;238;245];
txamp=[-65;-68;-62;-70];

// ------------------------------------------------------------
// IMPORTANT:
// Create a separate event set for EACH receiving station.
//
// Each station gets the TRUE bearing from that station
// to each transmitter.
//
// Columns:
// time_s, frequency_mhz, PW_us, true_AoA_deg,
// amplitude_dbm, sim_id
// ------------------------------------------------------------

events1=[];
events2=[];
events3=[];

for k=1:4

    x=txpos(k,1);
    y=txpos(k,2);

    // Bearing from each receiving station to this transmitter.
    b1=irsa_bearing_deg(st1.x_km,st1.y_km,x,y);
    b2=irsa_bearing_deg(st2.x_km,st2.y_km,x,y);
    b3=irsa_bearing_deg(st3.x_km,st3.y_km,x,y);

    // RS1 observations
    events1=[events1;
             0.001 txfreq(k) 20 b1 txamp(k)   k;
             0.002 txfreq(k) 21 b1 txamp(k)-1 k;
             0.003 txfreq(k) 20 b1 txamp(k)-2 k];

    // RS2 observations
    events2=[events2;
             0.001 txfreq(k) 20 b2 txamp(k)   k;
             0.002 txfreq(k) 21 b2 txamp(k)-1 k;
             0.003 txfreq(k) 20 b2 txamp(k)-2 k];

    // RS3 observations
    events3=[events3;
             0.001 txfreq(k) 20 b3 txamp(k)   k;
             0.002 txfreq(k) 21 b3 txamp(k)-1 k;
             0.003 txfreq(k) 20 b3 txamp(k)-2 k];

end

// ------------------------------------------------------------
// Receiver scan
// ------------------------------------------------------------

scan_start=0.0;
center=227.5;

// Each receiver observes the SAME RF environment,
// but from its own physical location.

st1=irsa_station_scan(st1,events1,scan_start,center);
st2=irsa_station_scan(st2,events2,scan_start,center);
st3=irsa_station_scan(st3,events3,scan_start,center);

// ------------------------------------------------------------
// Multi-station fusion
// ------------------------------------------------------------

station_pdws=list(st1.last_pdws, ...
                  st2.last_pdws, ...
                  st3.last_pdws);

stations=[st1.x_km st1.y_km;
          st2.x_km st2.y_km;
          st3.x_km st3.y_km];

tracks=irsa_fuse_stations(stations,station_pdws);

obs=irsa_build_observation( ...
    stations,station_pdws,tracks,scan_start);

// ------------------------------------------------------------
// Display
// ------------------------------------------------------------

disp("==============================================");
disp(" IRSA MULTI-STATION SDR RECEIVER DEMO");
disp("==============================================");

disp("Profile: HETEROGENEOUS SDR");
disp("Receiving stations: 3");
disp("Simultaneous emitters: 4");

disp(" ");
disp("Station positions (km):");
disp(stations);

disp(" ");
disp("Scan center MHz:");
disp(center);

disp(" ");
disp("Scan window MHz:");
disp([center-25 center+25]);

disp(" ");
disp("Tracks:");
disp("Columns: ID, stations, freq_MHz, mean_SNR_dB, X_km, Y_km, residual_km, confidence, mean_bearing");
disp(tracks);

disp(" ");
disp("Ground truth positions (evaluation only):");
disp(txpos);

disp(" ");
disp("Receiver-estimated positions:");
disp(tracks(:,5:6));

disp(" ");
disp("Position errors (km):");

for k=1:size(txpos,1)

    dx=tracks(k,5)-txpos(k,1);
    dy=tracks(k,6)-txpos(k,2);

    err=sqrt(dx^2+dy^2);

    disp([k err]);

end
