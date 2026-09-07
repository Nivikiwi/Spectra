/**
 * Smart Scan Strategy - Main Application Controller
 * Wires UI controls, navigation, and renders real-time telemetry DOM updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Simulation & Plotly Charts
    initSimulation();
    initCharts();
    renderSpectrumGrid(simulationState.bands);
    updateDashboard(simulationState);

    // Register global simulation step callback
    window.onSimulationStepUpdate = (state) => {
        updateDashboard(state);
    };

    // 2. Setup Event Listeners
    setupControlPanel();
    setupNavigation();
    setupSidebarToggle();

    // 3. Connect API WebSocket hook (will stay in MOCK mode unless toggled in api.js)
    connectWebSocket(window.onSimulationStepUpdate);
});

/**
 * Master UI update function triggered on every simulation step
 */
function updateDashboard(state) {
    updateHeaderStatus(state);
    updateMetrics(state);
    updateComparisonTable(state);
    updateSpectrum(state.bands);
    updateHistoryTable(state.history);
    updateActivityLog(state.events);
    updateCharts(state.chartHistory);

    // Update telemetry step counters
    document.getElementById('val-step').textContent = state.step;
    document.getElementById('val-episode').textContent = state.episode;
}

/**
 * Header status badge indicator state updates
 */
function updateHeaderStatus(state) {
    const badge = document.getElementById('sim-status-badge');
    const text = document.getElementById('sim-status-text');

    if (state.running) {
        badge.className = 'status-badge status-running';
        text.textContent = 'SIMULATION: RUNNING';
    } else if (state.step > 0) {
        badge.className = 'status-badge status-paused';
        text.textContent = 'SIMULATION: PAUSED';
    } else {
        badge.className = 'status-badge status-ready';
        text.textContent = 'SIMULATION: READY';
    }
}

/**
 * KPI Metric Card Updates
 */
function updateMetrics(state) {
    const { aiMetrics, baseMetrics, currentBand, bands } = state;

    // Detection Rate KPI
    document.getElementById('kpi-detection-rate').textContent = `${aiMetrics.detectionRate.toFixed(1)}%`;
    const diffDetection = (aiMetrics.detectionRate - baseMetrics.detectionRate).toFixed(1);
    const diffElem = document.getElementById('kpi-detection-diff');
    diffElem.textContent = `${diffDetection >= 0 ? '+' : ''}${diffDetection}% vs Baseline`;
    diffElem.className = diffDetection >= 0 ? 'kpi-trend trend-up' : 'kpi-trend trend-neutral';

    // Avg Detection Time KPI
    document.getElementById('kpi-detection-time').textContent = `${aiMetrics.detectionTime.toFixed(2)} s`;

    // Energy KPI
    document.getElementById('kpi-energy').textContent = `${aiMetrics.energyConsumption.toFixed(1)} units`;

    // Cumulative Reward KPI
    document.getElementById('kpi-reward').textContent = `${aiMetrics.cumulativeReward.toFixed(1)}`;

    // Current Band KPI
    document.getElementById('kpi-current-band').textContent = `Band ${currentBand < 10 ? '0' + currentBand : currentBand}`;
    const activeBandObj = bands.find(b => b.id === currentBand);
    if (activeBandObj) {
        document.getElementById('kpi-band-status').textContent = `State: ${activeBandObj.state}`;
    }
}

/**
 * Live updates for Strategy Comparison Table
 */
function updateComparisonTable(state) {
    const { aiMetrics, baseMetrics } = state;

    document.getElementById('comp-base-detection').textContent = `${baseMetrics.detectionRate.toFixed(1)}%`;
    document.getElementById('comp-ai-detection').textContent = `${aiMetrics.detectionRate.toFixed(1)}%`;
    const diffDetection = (aiMetrics.detectionRate - baseMetrics.detectionRate).toFixed(1);
    document.getElementById('comp-diff-detection').textContent = `${diffDetection >= 0 ? '+' : ''}${diffDetection}%`;

    document.getElementById('comp-base-time').textContent = `${baseMetrics.detectionTime.toFixed(2)} s`;
    document.getElementById('comp-ai-time').textContent = `${aiMetrics.detectionTime.toFixed(2)} s`;
    const diffTime = (baseMetrics.detectionTime - aiMetrics.detectionTime).toFixed(2);
    document.getElementById('comp-diff-time').textContent = `-${diffTime} s (${((diffTime / baseMetrics.detectionTime) * 100 || 0).toFixed(1)}% faster)`;

    document.getElementById('comp-base-energy').textContent = `${baseMetrics.energyConsumption.toFixed(1)} units`;
    document.getElementById('comp-ai-energy').textContent = `${aiMetrics.energyConsumption.toFixed(1)} units`;
    const energySavings = (baseMetrics.energyConsumption - aiMetrics.energyConsumption).toFixed(1);
    document.getElementById('comp-diff-energy').textContent = `-${energySavings} units saved`;

    document.getElementById('comp-base-reward').textContent = `${baseMetrics.cumulativeReward.toFixed(1)}`;
    document.getElementById('comp-ai-reward').textContent = `${aiMetrics.cumulativeReward.toFixed(1)}`;
    const rewardGain = (aiMetrics.cumulativeReward - baseMetrics.cumulativeReward).toFixed(1);
    document.getElementById('comp-diff-reward').textContent = `+${rewardGain} pts`;
}

/**
 * Initial creation of 20 Spectrum Band cards in DOM
 */
function renderSpectrumGrid(bands) {
    const container = document.getElementById('spectrum-grid');
    container.innerHTML = '';

    bands.forEach(band => {
        const card = document.createElement('div');
        card.className = `band-card is-${band.state.toLowerCase()}`;
        card.id = `band-card-${band.id}`;

        card.innerHTML = `
            <div class="band-number">${band.label}</div>
            <div class="band-status-tag">${band.state}</div>
        `;
        container.appendChild(card);
    });
}

/**
 * Fast visual DOM update for Spectrum Band cards
 */
function updateSpectrum(bands) {
    bands.forEach(band => {
        const card = document.getElementById(`band-card-${band.id}`);
        if (card) {
            card.className = `band-card is-${band.state.toLowerCase()}`;
            const tag = card.querySelector('.band-status-tag');
            if (tag) tag.textContent = band.state;
        }
    });
}

/**
 * Render Scan History Table rows
 */
function updateHistoryTable(history) {
    const tbody = document.getElementById('history-table-body');
    if (!history || history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">Simulation not started. Click START to run simulation.</td></tr>`;
        return;
    }

    tbody.innerHTML = history.map(row => `
        <tr>
            <td><strong>#${row.step}</strong></td>
            <td><span class="highlight-cyan">${row.band}</span></td>
            <td>${row.strategy}</td>
            <td><span class="badge ${row.result === 'DETECTED' ? 'badge-success' : 'badge-warning'}">${row.result}</span></td>
            <td class="${parseFloat(row.reward) >= 0 ? 'trend-up' : 'trend-neutral'}">${row.reward}</td>
            <td>${row.energy} u</td>
        </tr>
    `).join('');
}

/**
 * Update Activity Console Log
 */
function updateActivityLog(events) {
    const consoleBox = document.getElementById('activity-log');
    if (!events || events.length === 0) return;

    consoleBox.innerHTML = events.map(evt => {
        let typeClass = 'line-info';
        if (evt.includes('DETECTED')) typeClass = 'line-event';
        if (evt.includes('STARTED') || evt.includes('selected')) typeClass = 'line-scan';
        if (evt.includes('speed') || evt.includes('RESET')) typeClass = 'line-reward';

        return `<div class="console-line ${typeClass}">${evt}</div>`;
    }).join('');
}

/**
 * Bind Control Panel button handlers
 */
function setupControlPanel() {
    const btnStart = document.getElementById('btn-start');
    const btnPause = document.getElementById('btn-pause');
    const btnReset = document.getElementById('btn-reset');
    const speedSelect = document.getElementById('speed-select');

    btnStart.addEventListener('click', () => {
        sendSimulationCommand('START');
        btnStart.disabled = true;
        btnPause.disabled = false;
    });

    btnPause.addEventListener('click', () => {
        sendSimulationCommand('PAUSE');
        btnStart.disabled = false;
        btnPause.disabled = true;
    });

    btnReset.addEventListener('click', () => {
        sendSimulationCommand('RESET');
        btnStart.disabled = false;
        btnPause.disabled = true;
    });

    speedSelect.addEventListener('change', (e) => {
        sendSimulationCommand('SPEED', { speed: parseInt(e.target.value) });
    });
}

/**
 * Smooth scrolling and sidebar navigation active link handler
 */
function setupNavigation() {
    const navItems = document.querySelectorAll('.sidebar-nav .nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

/**
 * Sidebar collapse/expand toggle
 */
function setupSidebarToggle() {
    const sidebar = document.getElementById('sidebar');
    const toggle = document.getElementById('sidebar-toggle');

    toggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });
}
