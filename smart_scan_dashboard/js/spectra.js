/* Spectra dashboard. Every element is filled from GET /api/v1/simulation/state.
   Single mode: only the selected POLICY runs. COMPARE ALL: all three run side by side. */
'use strict';

const API = '/api/v1';
const POLL_MS = 500;
const KEYS = ['ai', 'whittle', 'base'];
const NAMES = { ai: 'Spectra', whittle: 'Whittle', base: 'Round Robin' };
const C = { accent: '#5e87b5', alert: '#c98a4b', ok: '#6e9e7a', dim: '#4c535d', text: '#d3d7dd' };
const PLOT_CFG = { responsive: true, displayModeBar: false };
const BASE_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'ui-monospace, Consolas, monospace', size: 10, color: '#9aa1ab' },
    margin: { l: 52, r: 10, t: 6, b: 32 },
    xaxis: { gridcolor: '#1c2027', zeroline: false, linecolor: '#262b33' },
    yaxis: { gridcolor: '#1c2027', zeroline: false, linecolor: '#262b33' },
    showlegend: false
};

const $ = id => document.getElementById(id);
const fin = v => typeof v === 'number' && isFinite(v);
const f = (v, d = 2, suf = '') => fin(v) ? v.toFixed(d) + suf : '—';
const int = v => fin(v) ? Math.round(v).toLocaleString('en-US') : '—';
const esc = s => String(s ?? '—').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
let S = null, busy = false, linkOk = null, tab = 'live';

// ------------------------------------------------------------------ transport
async function poll() {
    if (busy) return;
    busy = true;
    try {
        const r = await fetch(`${API}/simulation/state`, { cache: 'no-store' });
        if (!r.ok) throw new Error(r.status);
        S = await r.json();
        setLink(true);
        render(S);
    } catch (e) {
        setLink(false);
    } finally {
        busy = false;
    }
}

async function send(command, payload = {}) {
    try {
        const r = await fetch(`${API}/simulation/control`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command, ...payload })
        });
        const j = await r.json();
        if (!j.success && j.error) { $('st-message').textContent = 'ERROR: ' + j.error; $('st-message').classList.add('alert'); }
    } catch (e) { setLink(false); }
    poll();
}

function setLink(ok) {
    if (ok === linkOk) return;
    linkOk = ok;
    $('st-link').textContent = ok ? 'LINK OK' : 'LINK LOST';
    $('st-link').classList.toggle('alert', !ok);
}

// ------------------------------------------------------------------ render
function render(s) {
    renderBar(s);
    const tel = s.telemetry || {};
    const failN = ((tel.tierCounts || {})['2']) || 0;
    document.querySelector('.tab[data-tab="health"]').classList.toggle('has-alert', failN > 0 || s.phase === 'error');
    $('tab-info').textContent = s.runMode === 'compare' ? 'COMPARE ALL: ' + (s.runLabels || []).join(' + ')
                                                        : 'SINGLE POLICY: ' + ((s.runLabels || [])[0] || '—');
    if (tab === 'live') { renderHeatmap(s.heatmap, s.configStarts); renderDecision(s); renderKpis(s); renderReceiver(s); }
    if (tab === 'log') { renderLog(s.decisionLog || [], s); renderHitMiss(s.hitMiss || [], s); }
    if (tab === 'compare') renderComparison(s.comparison);
    if (tab === 'health') renderHealth(s);
}

function renderBar(s) {
    const running = !!s.active;
    $('btn-start').disabled = !!s.running || s.phase === 'starting';
    $('btn-pause').disabled = !s.running;
    const sel = $('sel-config');
    const want = (s.availableConfigs || []).join(',');
    if (sel.dataset.opts !== want) {
        const a = s.availableConfigs || [];
        sel.innerHTML = `<option value="all">ALL ${a[0]}-${a[a.length - 1]}</option>` +
            a.map(c => `<option value="${c}">${c} ${(s.testConfigs || []).includes(c) ? '(unseen)' : '(training)'}</option>`).join('');
        sel.dataset.opts = want;
    }
    if (document.activeElement !== sel) sel.value = String(s.selection);
    sel.disabled = running;
    const forced = !!(s.ppoForced && s.ppoForced.enabled);
    const fb = $('btn-force');
    fb.setAttribute('aria-pressed', forced ? 'true' : 'false');
    fb.textContent = forced ? 'PPO FORCED OFF - CLICK TO RESTORE' : 'FORCE PPO FAILURE';
    $('force-banner').hidden = !forced;
    const compare = s.runMode === 'compare';
    const cb = $('btn-compare');
    cb.setAttribute('aria-pressed', compare ? 'true' : 'false');
    cb.textContent = compare ? 'COMPARE ALL: ON' : 'COMPARE ALL: OFF';
    cb.disabled = running;
    $('lbl-policy').textContent = compare ? 'SHOW' : 'POLICY';
    const ss = $('sel-strategy');
    const val = s.strategyKey === 'ai' ? 'Spectra' : s.strategy;
    if (document.activeElement !== ss) ss.value = val;
    ss.disabled = running && !compare;                        // single mode: the running policy is fixed
    const p = s.progress || {};
    $('st-progress').textContent = (s.runKeys || []).map(k => `${NAMES[k]} ${p[k] && p[k].config !== null ? 'C' + p[k].config + ' ' + f(p[k].time_s, 1) + 's' : '—'}`).join(' · ') || '—';
    const n = (s.runKeys || []).length;
    $('st-source').textContent = 'RECEIVERS: ' + (s.mode === 'scilab' ? `${n} x LIVE SCILAB` : 'RECORDED SWEEP');
    let text = s.message || (compare
        ? 'COMPARE ALL is on: START runs Spectra, Whittle and Round Robin side by side.'
        : `Single policy: START runs only ${NAMES[s.strategyKey]}.`);
    if (s.lastAudit && s.phase === 'finished') text += `  |  audit: ${s.lastAudit}`;
    $('st-message').textContent = text;
    $('st-message').classList.toggle('alert', s.phase === 'error');
}

function cfgLines(starts, x0, x1) {
    return (starts || []).filter(c => x0 === undefined || (c.t > x0 && c.t < x1)).map(c => ({
        type: 'line', x0: c.t, x1: c.t, yref: 'paper', y0: 0, y1: 1, line: { color: '#2b313a', width: 1, dash: 'dot' } }));
}

function renderHeatmap(h, starts) {
    if (!h || !window.Plotly) return;
    const picks = h.picks || [];
    const hitP = picks.filter(p => p[2] === 1), missP = picks.filter(p => p[2] !== 1);
    const traces = [];
    if (h.hasModel && h.z.length) traces.push({
        type: 'heatmap', x: h.t, y: h.centres, z: h.z, zmin: 0, zmax: 1,
        colorscale: [[0, '#0c0e11'], [0.1, '#1d2735'], [0.5, '#4a6f98'], [1, '#c9d7e8']],
        colorbar: { thickness: 8, outlinewidth: 0, tickfont: { size: 9 } },
        hovertemplate: '%{x:.3f} s<br>%{y} MHz<br>P %{z:.2f}<extra></extra>'
    });
    traces.push({ type: 'scatter', mode: 'markers', x: hitP.map(p => p[0]), y: hitP.map(p => p[1]),
                  marker: { symbol: 'square', size: 5, color: C.ok }, hovertemplate: 'HIT %{x:.3f} s · %{y} MHz<extra></extra>' });
    traces.push({ type: 'scatter', mode: 'markers', x: missP.map(p => p[0]), y: missP.map(p => p[1]),
                  marker: { symbol: 'square-open', size: 5, color: '#8a919c' }, hovertemplate: 'MISS %{x:.3f} s · %{y} MHz<extra></extra>' });
    const t = h.t || [];
    const x0 = t.length ? t[0] : 0, x1 = t.length ? t[t.length - 1] + 0.01 : 1;
    Plotly.react('heatmap', traces, {
        ...BASE_LAYOUT, shapes: cfgLines(starts, x0, x1),
        xaxis: { ...BASE_LAYOUT.xaxis, range: [x0, x1], title: { text: 'run time (s)', font: { size: 10 } } },
        yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'band centre (MHz)', font: { size: 10 } }, range: [0, 16100] }
    }, PLOT_CFG);
    $('hm-sub').textContent = picks.length ? `${h.label} · last ${picks.length} scans · green = HIT, open = MISS` : 'no scans yet';
}

function resultCell(hit) { return hit ? '<td class="ok">HIT</td>' : '<td>MISS</td>'; }

function renderLog(rows, s) {
    $('log-sub').textContent = `${NAMES[s.strategyKey]}, newest first (last ${rows.length})`;
    const tb = $('tbl-log').tBodies[0];
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="10" class="empty">no scans yet</td></tr>'; return; }
    tb.innerHTML = rows.map(r => `<tr class="${r.tier === 2 ? 'alert-row' : ''}">
        <td>${r.config_id}</td><td>${int(r.dwell)}</td><td>${f(r.time_s, 3)}</td>
        <td>${fin(r.dwell_s) ? (r.dwell_s * 1000).toFixed(2) : '—'}</td><td>${int(r.centre_mhz)}</td>
        <td class="${r.tier === 2 ? 'alert' : ''}" title="${esc(r.decision)}">${esc(r.decision_label || r.decision)}</td>
        <td>${esc(r.freq_mode || '—')}</td><td>${f(r.p_target, 3)}</td><td>${f(r.confidence, 2)}</td>
        ${resultCell(r.receiver_hit)}<td>${int(r.pdws)}</td></tr>`).join('');
}

function renderDecision(s) {
    const L = (s.telemetry || {}).last;
    $('dec-strategy').textContent = NAMES[s.strategyKey] || '—';
    const set = (id, v, cls = '') => { $(id).textContent = v; $(id).classList.remove('alert', 'ok'); if (cls) $(id).classList.add(cls); };
    if (!L) { ['d-pos', 'd-time', 'd-band', 'd-p', 'd-conf', 'd-dwell', 'd-decision', 'd-result', 'd-pdws', 'd-source'].forEach(i => set(i, '—')); return; }
    const noModel = s.strategyKey === 'base';
    set('d-pos', `${L.config_id} / ${int(L.dwell)}`);
    set('d-time', `${f(L.time_s, 3)} s`);
    set('d-band', `${int(L.centre_mhz)} MHz`);
    set('d-p', noModel ? 'no model' : f(L.p_target, 3));
    set('d-conf', noModel ? 'no model' : f(L.confidence, 3));
    set('d-dwell', fin(L.dwell_s) ? `${(L.dwell_s * 1000).toFixed(2)} ms` : '—');
    set('d-decision', (L.decision_label || L.decision || '—') + (L.tierName ? ` · ${L.tierName}` : ''), L.tier === 2 ? 'alert' : '');
    $('d-decision').title = L.decision || '';
    set('d-why', L.why || (L.tier === 2 ? 'PPO unavailable' : '—'));
    set('d-fmode', L.freq_mode ? `${L.freq_mode} (${f(L.freq_mode_confidence, 2)})` : '—');
    set('d-result', (L.receiver_hit ? 'HIT' : 'MISS') + (fin(L.snr_db) ? ` · SNR ${L.snr_db.toFixed(1)} dB` : ''), L.receiver_hit ? 'ok' : '');
    set('d-pdws', int(L.pdws));
    set('d-source', s.receiverSource);
}

function spark(id, x, series, refX, ref) {
    const el = $(id);
    if (!el) return;
    if (!x || !series || series.length < 2) { el.innerHTML = ''; return; }
    const xs = x.concat(refX || []), ys = series.concat(ref || []);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const sx = v => (100 * (v - x0) / ((x1 - x0) || 1)).toFixed(2), sy = v => (23 - 22 * (v - y0) / ((y1 - y0) || 1)).toFixed(2);
    const pts = (a, b) => a.map((v, i) => `${sx(v)},${sy(b[i])}`).join(' ');
    el.innerHTML = (ref && ref.length > 1 ? `<polyline class="ref" points="${pts(refX, ref)}"/>` : '') + `<polyline points="${pts(x, series)}"/>`;
}

function renderKpis(s) {
    const K = s.kpi || {}, a = s.strategyKey, m = K[a];
    const ids = ['k-found', 'k-found-s', 'k-ttfd', 'k-ttfd-s', 'k-det', 'k-det-s', 'k-rate', 'k-rate-s', 'k-pd', 'k-pd-s', 'k-pdws', 'k-pdws-s'];
    if (!m) { ids.forEach(i => $(i).textContent = '—'); ['sp-found', 'sp-det', 'sp-rate'].forEach(i => $(i).innerHTML = ''); return; }
    const others = KEYS.filter(k => k !== a && K[k]);
    const sub = fn => others.length ? others.map(k => `${NAMES[k]} ${fn(K[k])}`).join(' · ') : 'single policy run';
    const SR = s.series || {}, ref = others.length ? (a === 'ai' ? 'whittle' : 'ai') : null;
    const sa = SR[a] || { t: [] }, sr = ref && SR[ref] ? SR[ref] : { t: [] };
    $('k-found').textContent = `${int(m.found)} / ${int(m.detectable)}`;
    $('k-found-s').textContent = sub(x => int(x.found));
    spark('sp-found', sa.t, sa.found, sr.t, sr.found);
    $('k-ttfd').textContent = `${f(m.delay_s, 2)} s`;
    $('k-ttfd-s').textContent = sub(x => f(x.delay_s, 2) + ' s');
    $('k-det').textContent = `${int(m.hits)} · ${f(m.per_s, 2)}/s`;
    $('k-det-s').textContent = sub(x => f(x.per_s, 2) + '/s');
    spark('sp-det', sa.t, sa.detections, sr.t, sr.detections);
    const r = sa.rate || [];
    $('k-rate').textContent = r.length ? `${f(r[r.length - 1], 1)}%` : '—';
    $('k-rate-s').textContent = `run ${f(m.rate, 1)}%` + (others.length ? ' · ' + sub(x => f(x.rate, 1) + '%') : '');
    spark('sp-rate', sa.t, sa.rate, sr.t, sr.rate);
    $('k-pd').textContent = `${f(m.pd, 1)}% / ${f(m.pfa, 2)}%`;
    $('k-pd-s').textContent = sub(x => `${f(x.pd, 1)}/${f(x.pfa, 2)}%`);
    $('k-pdws').textContent = `${int(m.pdws)} · ${f(m.mean_dwell_ms, 1)} ms`;
    $('k-pdws-s').textContent = sub(x => `${int(x.pdws)} · ${f(x.mean_dwell_ms, 1)} ms`);
}

function renderReceiver(s) {
    if (!window.Plotly) return;
    const rx = s.receiver || {};
    $('rx-ft-h').textContent = `${NAMES[s.strategyKey]} · ${s.receiverSource}`;
    const d = rx.detections || [];
    Plotly.react('rx-ft', [{
        type: 'scattergl', mode: 'markers', x: d.map(p => p[0]), y: d.map(p => p[1]),
        marker: { size: 4, color: d.map(p => p[2]), colorscale: [[0, '#2c3a4c'], [1, '#c9d7e8']],
                  colorbar: { thickness: 8, outlinewidth: 0, title: { text: 'SNR dB', side: 'right' } } },
        hovertemplate: '%{x:.3f} s<br>%{y} MHz<extra></extra>'
    }], { ...BASE_LAYOUT, shapes: cfgLines(s.configStarts), xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'run time (s)' } },
          yaxis: { ...BASE_LAYOUT.yaxis, range: [0, 16100], title: { text: 'MHz' } } }, PLOT_CFG);
    const g = rx.geo || {};
    const est = g.estimates || [], em = g.emitters || [], st = g.stations || [];
    const tr = [];
    if (est.length) tr.push({ type: 'scattergl', mode: 'markers', x: est.map(e => e[0]), y: est.map(e => e[1]), name: 'estimate',
                              marker: { size: 4, color: C.accent, opacity: 0.7 } });
    if (em.length) tr.push({ type: 'scatter', mode: 'markers', x: em.map(e => e[1]), y: em.map(e => e[2]), name: 'true start',
                             marker: { symbol: 'x', size: 7, color: '#7f8792' } });
    if (st.length) tr.push({ type: 'scatter', mode: 'markers+text', x: st.map(e => e[1]), y: st.map(e => e[2]), text: st.map(e => 'RS' + e[0]),
                             name: 'station', textposition: 'top center', marker: { symbol: 'triangle-up', size: 10, color: C.text } });
    Plotly.react('rx-geo', tr, { ...BASE_LAYOUT, showlegend: tr.length > 0, legend: { orientation: 'h', y: 1.08, font: { size: 9 } },
        xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'x (km)' } },
        yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'y (km)' }, scaleanchor: 'x' },
        annotations: tr.length ? [] : [{ text: 'geolocation comes from the live Scilab receiver', showarrow: false, xref: 'paper', yref: 'paper', x: 0.5, y: 0.5 }]
    }, PLOT_CFG);
}

function renderHitMiss(rows, s) {
    const tb = $('tbl-hm').tBodies[0];
    tb.innerHTML = rows.length ? rows.map(r => `<tr class="${r.strategy === (s.strategyLabel) ? 'me' : ''}"><td class="lbl">${esc(r.strategy)}</td><td>${int(r.scans)}</td>
        <td class="ok">${int(r.hits)}</td><td>${int(r.misses)}</td><td>${f(r.hitPct, 1)}%</td><td>${int(r.tp)}</td><td>${int(r.fp)}</td>
        <td>${int(r.tn)}</td><td>${int(r.fn)}</td><td>${esc(r.found)}</td><td>${f(r.meanDwellMs, 1)} ms</td></tr>`).join('')
        : '<tr><td colspan="11" class="empty">no scans yet</td></tr>';
}

function cmpTable(id, block, emptyText) {
    const t = $(id);
    const head = '<thead><tr><th>METRIC</th><th>ROUND ROBIN</th><th>WHITTLE INDEX</th><th>SPECTRA</th></tr></thead>';
    if (!block || !block.rows || !block.rows.length) {
        t.innerHTML = head + `<tbody><tr><td colspan="4" class="empty">${emptyText}</td></tr></tbody>`;
        return;
    }
    t.innerHTML = head + '<tbody>' + block.rows.map(r => {
        const v = ['base', 'whittle', 'ai'].map(k => r.values[k]);
        const n = v.map(x => (x && fin(x.n)) ? x.n : null);
        let best = -1;
        if (r.better && n.every(x => x !== null)) {
            const tgt = r.better === 'higher' ? Math.max(...n) : Math.min(...n);
            if (n.filter(x => x === tgt).length === 1) best = n.indexOf(tgt);
        }
        return `<tr><td class="lbl">${esc(r.metric)}<small>${esc(r.sub)}</small></td>` +
            v.map((x, i) => `<td class="${i === best ? 'best' : ''}">${x ? esc(x.text) : '—'}</td>`).join('') + '</tr>';
    }).join('') + '</tbody>' + (block.caption ? `<caption class="note" style="caption-side:bottom;text-align:left">${esc(block.caption)}</caption>` : '');
}

function renderComparison(c) {
    if (!c) return;
    $('cmp-note').textContent = c.note || '';
    const emptyRun = c.compareMode ? 'no configuration of this type run yet' : 'switch COMPARE ALL on and press START';
    cmpTable('cmp-unseen', c.unseen, emptyRun);
    cmpTable('cmp-train', c.train, emptyRun);
    const rows = c.perConfig || [];
    $('cmp-per').innerHTML = '<thead><tr><th>CONFIG</th><th>TYPE</th><th>ROUND ROBIN</th><th>WHITTLE</th><th>SPECTRA</th><th>MOST TRANSMITTERS FOUND</th></tr></thead><tbody>' +
        (rows.length ? rows.map(r => `<tr><td>${r.config}</td><td>${esc(r.split)}</td><td>${esc(r.base)}</td><td>${esc(r.whittle)}</td><td>${esc(r.ai)}</td><td>${esc(r.winner)}</td></tr>`).join('')
            : `<tr><td colspan="6" class="empty">${emptyRun}</td></tr>`) + '</tbody>';
    const A = c.algorithms, t = $('cmp-algos');
    if (A && A.rows && A.rows.length) {
        t.innerHTML = '<thead><tr>' + A.columns.map(x => `<th>${esc(x)}</th>`).join('') + '</tr></thead><tbody>' +
            A.rows.map(r => `<tr class="${String(r[0]).startsWith('Spectra') || String(r[0]).startsWith('SPECTRA') ? 'me' : ''}">` +
                r.map((x, i) => `<td class="${i === 0 ? 'lbl' : ''}">${esc(x)}</td>`).join('') + '</tr>').join('') + '</tbody>';
    } else {
        t.innerHTML = '<tbody><tr><td class="empty">run step7_compare_algorithms.py (part of run_all_training.py) to fill this table</td></tr></tbody>';
    }
}

function renderHealth(s) {
    const tel = s.telemetry || {};
    const tb = $('tbl-tiers').tBodies[0];
    if (!tel.spectraRunning) {
        tb.innerHTML = '<tr><td colspan="4" class="empty">Spectra is not running in this session (tiers apply to Spectra only)</td></tr>';
        $('tier-note').textContent = `Running: ${(s.runLabels || []).join(', ') || '—'}`;
        $('tier-note').classList.remove('alert');
    } else {
        const c = tel.tierCounts || {};
        const tot = (c['1'] || 0) + (c['2'] || 0);
        tb.innerHTML = [['1', 'PPO_CUSTOM (Spectra)'], ['2', 'WHITTLE (failover)']].map(([k, m]) => {
            const n = c[k] || 0, bad = k === '2' && n > 0;
            return `<tr><td class="${bad ? 'alert' : ''}">${k}</td><td class="${bad ? 'alert' : ''}">${m}</td><td class="${bad ? 'alert' : ''}">${int(n)}</td>
                    <td>${tot ? (100 * n / tot).toFixed(1) + '%' : '—'}</td></tr>`;
        }).join('');
        const parts = [`PPO ${tel.ppoLoaded ? 'loaded' : 'NOT loaded'}`, `timeout ${f(tel.ppoTimeoutMs, tel.ppoTimeoutMs < 10 ? 2 : 0)} ms`];
        if (tel.forcedTier) parts.push(`FORCED TIER ${tel.forcedTier} (SPECTRA_FORCE_TIER)`);
        $('tier-note').textContent = parts.join(' · ');
        $('tier-note').classList.toggle('alert', !!tel.forcedTier || !tel.ppoLoaded);
    }
    const L = tel.latencyMs;
    $('h-n').textContent = L ? int(L.n) : '—';
    $('h-mean').textContent = L ? `${f(L.mean, 2)} ms` : '—';
    $('h-p95').textContent = L ? `${f(L.p95, 2)} ms` : '—';
    $('h-max').textContent = L ? `${f(L.max, 2)} ms` : '—';
    $('h-max').classList.toggle('alert', !!(L && fin(tel.ppoTimeoutMs) && L.max > tel.ppoTimeoutMs));
    $('h-timeout').textContent = fin(tel.ppoTimeoutMs) ? `${f(tel.ppoTimeoutMs, tel.ppoTimeoutMs < 10 ? 2 : 0)} ms` : '—';
    $('h-rx').textContent = `${(s.runKeys || []).length} x ${s.mode === 'scilab' ? 'live Scilab' : 'recorded sweep'} · ${(s.runLabels || []).join(', ')}`;
    const fails = tel.failovers || [];
    $('tbl-fail').tBodies[0].innerHTML = fails.length ? fails.map(e => `<tr class="alert-row"><td>${esc(e.time)}</td><td>${e.config}</td><td>${int(e.scan)}</td>
        <td>${f(e.sim_time_s, 3)}</td><td>${esc(e.reason_code || '—')}</td>
        <td class="alert" title="${esc(e.decision)}">${esc(e.label || e.decision)}</td></tr>`).join('')
        : `<tr><td colspan="6" class="empty">${tel.spectraRunning ? 'no failover so far - every decision came from PPO' : 'Spectra is not running in this session'}</td></tr>`;
    const l = s.lpi || {};
    const cv = l.cv || {};
    let note = l.available ? `${l.classified} classified · LPI correct ${l.lpiCorrect} · LPD correct ${l.lpdCorrect}` : 'LPI/LPD models missing - run the training (step 6)';
    if (cv.lpi && cv.lpd) note += ` · cross-validated AUC LPI ${cv.lpi.roc_auc} / LPD ${cv.lpd.roc_auc}`;
    $('lpi-note').textContent = note;
    const yn = v => v === null || v === undefined ? '—' : (v ? 'yes' : 'no');
    const rows = l.rows || [];
    $('tbl-lpi').tBodies[0].innerHTML = rows.length ? rows.map(r => {
        const bi = r.trueLPI !== null && r.predLPI !== r.trueLPI, bd = r.trueLPD !== null && r.predLPD !== r.trueLPD;
        return `<tr><td>${esc(r.tx)}</td><td>${int(r.pulses)}</td><td title="confidence ${f(r.freqModeConf,2)}">${esc(r.freqMode || '—')}</td><td>${f(r.pLPI, 2)}</td><td class="${bi ? 'alert' : ''}">${yn(r.predLPI)}</td>
            <td>${yn(r.trueLPI)}</td><td>${f(r.pLPD, 2)}</td><td class="${bd ? 'alert' : ''}">${yn(r.predLPD)}</td><td>${yn(r.trueLPD)}</td></tr>`;
    }).join('') : '<tr><td colspan="9" class="empty">no transmitter with 3 intercepted pulses yet</td></tr>';
    $('events').textContent = (s.events || []).join('\n') || '—';
}

// ------------------------------------------------------------------ wiring
function showTab(name) {
    tab = name;
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === 'page-' + name));
    if (S) render(S);
    requestAnimationFrame(() => document.querySelectorAll('#page-' + name + ' .chart').forEach(c => {
        if (window.Plotly && c.data) Plotly.Plots.resize(c);
    }));
}

function init() {
    $('btn-start').addEventListener('click', () => send('START'));
    $('btn-pause').addEventListener('click', () => send('PAUSE'));
    $('btn-reset').addEventListener('click', () => send('RESET'));
    $('btn-force').addEventListener('click', () => {
        const on = $('btn-force').getAttribute('aria-pressed') === 'true';
        send('FORCE_PPO_FAILURE', { enabled: !on, source: 'dashboard' });
    });
    $('btn-compare').addEventListener('click', () => {
        const on = $('btn-compare').getAttribute('aria-pressed') === 'true';
        send('RUN_MODE', { mode: on ? 'single' : 'compare' });
    });
    $('sel-config').addEventListener('change', e => send('CONFIG', { config: e.target.value === 'all' ? 'all' : parseInt(e.target.value, 10) }));
    $('sel-strategy').addEventListener('change', e => send('STRATEGY', { strategy: e.target.value }));
    document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => showTab(t.dataset.tab)));
    poll();
    setInterval(poll, POLL_MS);
}
document.addEventListener('DOMContentLoaded', init);
