# Smart Scan Strategy for Electronic Warfare
## AI-Driven Spectrum Scan Simulation — High-Fidelity Frontend Prototype

> **SIH / DRDO Technical Demonstration Prototype**  
> **Notice**: This project is exclusively a frontend synthetic demonstration dashboard created for Smart India Hackathon judging. It uses completely synthetic mock data generated in JavaScript to compare an AI-based scanning strategy (e.g., PPO Reinforcement Learning) against a conventional baseline strategy. It does **not** contain or execute real-world electronic warfare functionality, real radar operating parameters, or signal interception routines.

---

## 📁 File Structure

```text
smart-scan-dashboard/
│
├── index.html          # Main HTML5 Dashboard Single-Page Structure
│
├── css/
│   └── style.css       # Complete Dark Defense/Engineering Design System & CSS Variables
│
├── js/
│   ├── app.js          # UI Initialization, Event Controllers & DOM Renderers
│   ├── simulation.js   # Central Mock Simulation Engine & Synthetic State Generator
│   ├── charts.js       # Dark-Themed Plotly.js Telemetry Charts Wrapper
│   └── api.js          # Backend Connector (Mock Mode + FastAPI / WebSocket Integration Hooks)
│
└── README.md           # Architecture & Setup Documentation
```

---

## 🚀 Quick Start (Running Locally)

Because this application is built with standard HTML5, CSS3, Vanilla JavaScript, and Plotly.js (via CDN), no heavy build tools or Node package installs are required.

### Method 1: Using Python built-in HTTP server
Open your terminal in the `smart-scan-dashboard` directory:
```bash
python -m http.server 8000
```
Then open your web browser to: [http://localhost:8000](http://localhost:8000)

### Method 2: Using Node `serve`
```bash
npx serve .
```

### Method 3: VS Code Live Server
Right-click `index.html` in VS Code and choose **Open with Live Server**.

---

## ⚙️ How the Synthetic Simulation Works

1. **Central State (`simulationState` in `js/simulation.js`)**:  
   All metrics, episode/step counts, spectrum band states, and telemetry history are managed in a single central JavaScript state object.

2. **Simulation Control Tick (`stepMockSimulation()`)**:  
   When **START** is clicked, a interval timer ticks every 500ms (at 2x speed):
   - Selects a synthetic band (prioritizing simulated active channels).
   - Dynamically calculates smooth learning curves for **Detection Rate**, **Detection Time**, **Energy Consumption**, and **Cumulative Reward**.
   - Updates visual states for all 20 spectrum bands (`IDLE`, `SELECTED`, `OBSERVED`, `OCCUPIED`).
   - Appends entries to the scan decision history table and console event log.
   - Pushes time-series data to 3 live Plotly line charts (`Plotly.react`).

3. **Control Buttons**:
   - **START**: Launches or resumes the mock tick loop.
   - **PAUSE**: Halts updates instantly.
   - **RESET**: Restores step 0, clears chart histories, and resets metrics.
   - **SPEED**: Allows toggling simulation rate between 1x, 2x, and 5x.

---

## 🔌 Connecting to FastAPI / WebSocket Backend

When your backend teammate is ready to connect the Python FastAPI backend, follow these steps:

### 1. Enable Backend Connection Mode in `js/api.js`
Open `js/api.js` and change the configuration flag:
```javascript
const API_CONFIG = {
    USE_MOCK_BACKEND: false, // Set to false to disable local mock simulation
    BASE_URL: 'http://localhost:8000/api/v1',
    WS_URL: 'ws://localhost:8000/ws/simulation'
};
```

### 2. Required FastAPI REST Endpoints

- **GET `/api/v1/simulation/state`**  
  Returns the current simulation state dictionary.

- **POST `/api/v1/simulation/control`**  
  Receives command payload:
  ```json
  {
    "command": "START" | "PAUSE" | "RESET" | "SPEED",
    "speed": 2
  }
  ```

### 3. Required WebSocket Telemetry Stream (`ws://localhost:8000/ws/simulation`)
The backend should stream JSON frames matching the state structure:
```json
{
  "step": 247,
  "episode": 12,
  "currentBand": 13,
  "aiMetrics": {
    "detectionRate": 87.4,
    "detectionTime": 2.84,
    "energyConsumption": 64.2,
    "cumulativeReward": 742.8
  },
  "baseMetrics": {
    "detectionRate": 72.0,
    "detectionTime": 4.80,
    "energyConsumption": 100.0,
    "cumulativeReward": 510.0
  },
  "bands": [
    { "id": 1, "state": "IDLE" },
    { "id": 13, "state": "SELECTED" }
  ]
}
```

When a frame is received, `connectWebSocket()` in `js/api.js` passes the payload to `window.onSimulationStepUpdate`, which seamlessly updates all KPI cards, Plotly charts, spectrum channels, history tables, and telemetry logs!

---

## 🎨 Design Tokens & Customization
All UI colors, panel backgrounds, borders, and typography tokens are defined at the top of `css/style.css` using CSS custom properties (`:root`). Colors can be adjusted globally by modifying these variables.
