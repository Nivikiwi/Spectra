// V2 RF environment helper.
//
// The pulse dataset already contains realized pulse frequency, PW, ToA,
// AoA and amplitude. V2 therefore treats each row as an observed RF pulse
// event and derives the per-pulse receiver SNR from signal level minus
// configured noise floor.
//
// More detailed propagation, beam, fading and scan-angle physics can be
// added after the pulse-to-pulse replay is validated.
