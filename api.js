/**
 * Smart Scan Strategy - FastAPI & WebSocket Connection Interface
 * 
 * IMPORTANT ARCHITECTURE NOTE FOR BACKEND TEAM:
 * This module acts as the abstraction layer between the UI dashboard and the backend.
 * Currently, it operates in MOCK MODE (using simulation.js).
 * To switch to real FastAPI backend integration:
 *   1. Change USE_MOCK_BACKEND to false below.
 *   2. Implement backend API endpoints specified in the TODO comments.
 */

const API_CONFIG = {
    USE_MOCK_BACKEND: true, // Set to false when connecting to real FastAPI server
    BASE_URL: 'http://localhost:8000/api/v1',
    WS_URL: 'ws://localhost:8000/ws/simulation'
};

let webSocketInstance = null;

/**
 * PLACEHOLDER: Fetch simulation state via HTTP REST
 * Expected Endpoint: GET /api/v1/simulation/state
 */
async function fetchSimulationState() {
    if (API_CONFIG.USE_MOCK_BACKEND) {
        // Mock fallback: Return local JS state
        return simulationState;
    }

    try {
        // TODO: Backend teammate to implement FastAPI endpoint GET /api/v1/simulation/state
        const response = await fetch(`${API_CONFIG.BASE_URL}/simulation/state`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.warn("[API] Failed to fetch backend simulation state. Falling back to mock state.", error);
        return simulationState;
    }
}

/**
 * PLACEHOLDER: Send simulation command (START, PAUSE, RESET, SET_SPEED) via HTTP REST
 * Expected Endpoint: POST /api/v1/simulation/control
 */
async function sendSimulationCommand(command, payload = {}) {
    console.log(`[API Command] Sent command '${command}'`, payload);

    if (API_CONFIG.USE_MOCK_BACKEND) {
        // Local execution handler
        switch(command) {
            case 'START':
                startMockSimulation(window.onSimulationStepUpdate);
                break;
            case 'PAUSE':
                pauseMockSimulation();
                break;
            case 'RESET':
                resetMockSimulation(window.onSimulationStepUpdate);
                break;
            case 'SPEED':
                setSimulationSpeed(payload.speed);
                break;
        }
        return { success: true, mode: 'mock' };
    }

    try {
        // TODO: Backend teammate to implement FastAPI endpoint POST /api/v1/simulation/control
        const response = await fetch(`${API_CONFIG.BASE_URL}/simulation/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command, ...payload })
        });
        return await response.json();
    } catch (error) {
        console.error("[API] Error sending command to FastAPI backend:", error);
        return { success: false, error: error.message };
    }
}

/**
 * PLACEHOLDER: Connect to FastAPI WebSocket stream
 * Expected Endpoint: ws://localhost:8000/ws/simulation
 */
function connectWebSocket(onMessageCallback) {
    if (API_CONFIG.USE_MOCK_BACKEND) {
        console.log("[WebSocket] MOCK MODE ACTIVE. Internal timer tick drives updates.");
        return;
    }

    // TODO: Backend teammate to implement FastAPI WebSocket broadcast loop
    console.log(`[WebSocket] Connecting to ${API_CONFIG.WS_URL}...`);
    webSocketInstance = new WebSocket(API_CONFIG.WS_URL);

    webSocketInstance.onopen = () => {
        console.log("[WebSocket] Connected to FastAPI backend successfully.");
        document.getElementById('backend-status').classList.remove('status-backend');
        document.getElementById('backend-status').classList.add('status-online');
        document.getElementById('backend-status').querySelector('.status-text').textContent = "BACKEND: FASTAPI CONNECTED";
    };

    webSocketInstance.onmessage = (event) => {
        try {
            const telemetryPayload = JSON.parse(event.data);
            // TODO: Update global dashboard state from real-time telemetry frame
            if (typeof onMessageCallback === 'function') {
                onMessageCallback(telemetryPayload);
            }
        } catch (e) {
            console.error("[WebSocket] Failed to parse telemetry frame:", e);
        }
    };

    webSocketInstance.onerror = (error) => {
        console.error("[WebSocket] Connection error:", error);
    };

    webSocketInstance.onclose = () => {
        console.warn("[WebSocket] Connection closed. Attempting reconnect in 5s...");
        setTimeout(() => connectWebSocket(onMessageCallback), 5000);
    };
}
