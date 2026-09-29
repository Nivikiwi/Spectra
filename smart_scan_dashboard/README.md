# Smart Scan Strategy - Frontend Prototype

This version preserves the existing dark technical UI and adds the requested software behavior.

## Changes

- Simulation Control now contains a synthetic MAB (Multi-Armed Bandit) simulator panel.
- Strategy selector: LightGBM + DBN, Whittle Index, Round Robin.
- Added synthetic MAB reward, arm-selection, band-score and selected-band radar charts.
- Line charts use sharp/linear segments rather than spline curves.
- Dashboard has an **AUDIT REPORT** button showing all main metrics and allowing CSV export.
- Spectrum Monitor has a filter for All, Idle, Occupied, Reserved and Selected.
- Added synthetic Reserved bands (5 and 15) for UI demonstration.
- Strategy Comparison now shows Round Robin, Whittle Index and LightGBM + DBN.
- Existing UI styling/layout is retained; new controls use the same visual language.

## Important

All simulation values and band activity are synthetic demo data. The frontend is not a real-world signal/radar control system.

## Backend integration

The existing `js/api.js` remains the integration boundary. When the FastAPI backend is ready, the mock state updates can be replaced by the backend JSON/WebSocket payloads while keeping the dashboard components.
