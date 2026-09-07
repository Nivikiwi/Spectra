/**
 * Smart Scan Strategy - Plotly.js Visualization Manager
 * Handles initializing and updating dark-themed telemetry charts.
 */

// Common Plotly dark layout template
const chartLayoutDefaults = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 45, r: 25, t: 20, b: 40 },
    font: {
        family: 'Inter, sans-serif',
        color: '#94a3b8',
        size: 11
    },
    xaxis: {
        gridcolor: '#243049',
        zerolinecolor: '#243049',
        tickfont: { color: '#64748b' },
        title: { text: 'Simulation Step', font: { size: 11 } }
    },
    yaxis: {
        gridcolor: '#243049',
        zerolinecolor: '#243049',
        tickfont: { color: '#64748b' }
    },
    legend: {
        orientation: 'h',
        x: 0,
        y: 1.15,
        font: { color: '#f1f5f9' }
    },
    hovermode: 'x unified'
};

const chartConfig = {
    responsive: true,
    displayModeBar: false
};

/**
 * Initialize all 3 Plotly charts with empty/initial datasets
 */
function initCharts() {
    // Chart 1: Detection Rate Trajectory
    const traceAIDetection = {
        x: [0],
        y: [0],
        mode: 'lines',
        name: 'AI Smart Scan (PPO)',
        line: { color: '#38bdf8', width: 2.5, shape: 'spline' }
    };
    const traceBaseDetection = {
        x: [0],
        y: [0],
        mode: 'lines',
        name: 'Conventional Baseline',
        line: { color: '#fbbf24', width: 2, dash: 'dot', shape: 'spline' }
    };

    const layoutDetection = JSON.parse(JSON.stringify(chartLayoutDefaults));
    layoutDetection.yaxis.title = { text: 'Detection Rate (%)', font: { size: 11 } };
    layoutDetection.yaxis.range = [0, 100];

    Plotly.newPlot('chart-detection', [traceAIDetection, traceBaseDetection], layoutDetection, chartConfig);

    // Chart 2: Cumulative Energy Consumption
    const traceAIEnergy = {
        x: [0],
        y: [0],
        mode: 'lines',
        name: 'AI Smart Scan (PPO)',
        line: { color: '#38bdf8', width: 2.5 }
    };
    const traceBaseEnergy = {
        x: [0],
        y: [0],
        mode: 'lines',
        name: 'Conventional Baseline',
        line: { color: '#fbbf24', width: 2, dash: 'dot' }
    };

    const layoutEnergy = JSON.parse(JSON.stringify(chartLayoutDefaults));
    layoutEnergy.yaxis.title = { text: 'Energy (Synthetic Units)', font: { size: 11 } };

    Plotly.newPlot('chart-energy', [traceAIEnergy, traceBaseEnergy], layoutEnergy, chartConfig);

    // Chart 3: Cumulative Reward Curve
    const traceAIReward = {
        x: [0],
        y: [0],
        mode: 'lines',
        name: 'AI Smart Scan (PPO)',
        line: { color: '#34d399', width: 2.5, shape: 'spline' }
    };
    const traceBaseReward = {
        x: [0],
        y: [0],
        mode: 'lines',
        name: 'Conventional Baseline',
        line: { color: '#fbbf24', width: 2, dash: 'dot', shape: 'spline' }
    };

    const layoutReward = JSON.parse(JSON.stringify(chartLayoutDefaults));
    layoutReward.yaxis.title = { text: 'Cumulative RL Reward Score', font: { size: 11 } };

    Plotly.newPlot('chart-reward', [traceAIReward, traceBaseReward], layoutReward, chartConfig);
}

/**
 * Update all Plotly charts dynamically with simulation telemetry history
 */
function updateCharts(chartHistory) {
    if (!chartHistory || !chartHistory.steps) return;

    // Update Detection Chart
    Plotly.react('chart-detection', [
        {
            x: chartHistory.steps,
            y: chartHistory.aiDetection,
            mode: 'lines',
            name: 'AI Smart Scan (PPO)',
            line: { color: '#38bdf8', width: 2.5, shape: 'spline' }
        },
        {
            x: chartHistory.steps,
            y: chartHistory.baseDetection,
            mode: 'lines',
            name: 'Conventional Baseline',
            line: { color: '#fbbf24', width: 2, dash: 'dot', shape: 'spline' }
        }
    ], {
        ...chartLayoutDefaults,
        yaxis: { gridcolor: '#243049', tickfont: { color: '#64748b' }, range: [0, 100], title: { text: 'Detection Rate (%)', font: { size: 11 } } }
    }, chartConfig);

    // Update Energy Chart
    Plotly.react('chart-energy', [
        {
            x: chartHistory.steps,
            y: chartHistory.aiEnergy,
            mode: 'lines',
            name: 'AI Smart Scan (PPO)',
            line: { color: '#38bdf8', width: 2.5 }
        },
        {
            x: chartHistory.steps,
            y: chartHistory.baseEnergy,
            mode: 'lines',
            name: 'Conventional Baseline',
            line: { color: '#fbbf24', width: 2, dash: 'dot' }
        }
    ], {
        ...chartLayoutDefaults,
        yaxis: { gridcolor: '#243049', tickfont: { color: '#64748b' }, title: { text: 'Energy (Synthetic Units)', font: { size: 11 } } }
    }, chartConfig);

    // Update Reward Chart
    Plotly.react('chart-reward', [
        {
            x: chartHistory.steps,
            y: chartHistory.aiReward,
            mode: 'lines',
            name: 'AI Smart Scan (PPO)',
            line: { color: '#34d399', width: 2.5, shape: 'spline' }
        },
        {
            x: chartHistory.steps,
            y: chartHistory.baseReward,
            mode: 'lines',
            name: 'Conventional Baseline',
            line: { color: '#fbbf24', width: 2, dash: 'dot', shape: 'spline' }
        }
    ], {
        ...chartLayoutDefaults,
        yaxis: { gridcolor: '#243049', tickfont: { color: '#64748b' }, title: { text: 'Cumulative RL Reward Score', font: { size: 11 } } }
    }, chartConfig);
}

/**
 * Reset all charts to step 0
 */
function resetCharts() {
    initCharts();
}
