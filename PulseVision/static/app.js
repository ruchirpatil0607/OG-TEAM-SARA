/**
 * OmniVitals — Interactive Client Application
 * High-performance UI rendering, telemetry polling, canvas waveform charting, and REST connectivity.
 */

// Global State
let currentPage = 'home';
let telemetryInterval = null;
let activeSessionState = 'IDLE';
let lastTelemetryData = null;

// ============================================================================
// Page Navigation
// ============================================================================
function navigateTo(pageId) {
  currentPage = pageId;

  // Update navigation sidebar active state
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.getAttribute('data-page') === pageId);
  });

  // Switch visible page view
  document.querySelectorAll('.page-view').forEach(view => {
    view.classList.remove('active');
  });

  const targetView = document.getElementById(`view-${pageId}`);
  if (targetView) {
    targetView.classList.add('active');
  }

  // Trigger page-specific initializers
  if (pageId === 'monitor') {
    startTelemetryPolling();
    resizeCanvases();
  } else if (pageId === 'analytics') {
    loadAnalyticsData('7days');
  } else if (pageId === 'history') {
    loadSessionsList();
  } else if (pageId === 'settings') {
    loadSettings();
  }
}

// ============================================================================
// Home Page: 3D Face Wireframe Animation
// ============================================================================
function initWireframeAnimation() {
  const canvas = document.getElementById('wireframeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // Stylized 3D Face Landmark Nodes
  const faceVertices = [
    // Forehead
    { x: 0, y: -0.75, z: 0.1 }, { x: -0.3, y: -0.65, z: 0.2 }, { x: 0.3, y: -0.65, z: 0.2 },
    { x: -0.55, y: -0.5, z: 0.35 }, { x: 0.55, y: -0.5, z: 0.35 },
    // Eyes & Brow
    { x: -0.35, y: -0.3, z: 0.4 }, { x: -0.15, y: -0.3, z: 0.45 },
    { x: 0.15, y: -0.3, z: 0.45 }, { x: 0.35, y: -0.3, z: 0.4 },
    // Nose bridge & tip
    { x: 0, y: -0.3, z: 0.5 }, { x: 0, y: -0.1, z: 0.6 }, { x: 0, y: 0.1, z: 0.7 },
    { x: -0.12, y: 0.12, z: 0.55 }, { x: 0.12, y: 0.12, z: 0.55 },
    // Cheeks
    { x: -0.45, y: 0.05, z: 0.4 }, { x: 0.45, y: 0.05, z: 0.4 },
    { x: -0.35, y: 0.25, z: 0.45 }, { x: 0.35, y: 0.25, z: 0.45 },
    // Mouth
    { x: -0.22, y: 0.4, z: 0.5 }, { x: 0.22, y: 0.4, z: 0.5 },
    { x: 0, y: 0.35, z: 0.55 }, { x: 0, y: 0.48, z: 0.52 },
    // Jaw & Chin
    { x: -0.5, y: 0.25, z: 0.25 }, { x: 0.5, y: 0.25, z: 0.25 },
    { x: -0.35, y: 0.55, z: 0.3 }, { x: 0.35, y: 0.55, z: 0.3 },
    { x: 0, y: 0.75, z: 0.45 }
  ];

  // Meshed polygon edge indices
  const edges = [
    [0, 1], [0, 2], [1, 3], [2, 4], [1, 5], [2, 8],
    [5, 6], [7, 8], [6, 9], [7, 9], [9, 10], [10, 11],
    [11, 12], [11, 13], [12, 14], [13, 15], [14, 16], [15, 17],
    [16, 18], [17, 19], [18, 20], [19, 20], [18, 21], [19, 21],
    [3, 22], [4, 23], [22, 24], [23, 25], [24, 26], [25, 26],
    [21, 26], [16, 24], [17, 25], [6, 10], [7, 10]
  ];

  // Floating background space particles
  const particles = Array.from({ length: 40 }, () => ({
    x: (Math.random() - 0.5) * 2.2,
    y: (Math.random() - 0.5) * 2.2,
    z: Math.random() * 0.8,
    speed: 0.002 + Math.random() * 0.004
  }));

  let angleY = 0;

  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const scale = Math.min(canvas.width, canvas.height) * 0.46;

    angleY += 0.008;

    // Render floating background particles
    ctx.fillStyle = 'rgba(0, 242, 254, 0.3)';
    particles.forEach(p => {
      p.y -= p.speed;
      if (p.y < -1.1) p.y = 1.1;
      const px = cx + p.x * scale;
      const py = cy + p.y * scale;
      ctx.beginPath();
      ctx.arc(px, py, 1.2, 0, Math.PI * 2);
      ctx.fill();
    });

    // 3D rotation transform
    const cosY = Math.cos(angleY * 0.8);
    const sinY = Math.sin(angleY * 0.8);

    const projected = faceVertices.map(v => {
      // Rotate around Y axis
      const rx = v.x * cosY + v.z * sinY;
      const rz = -v.x * sinY + v.z * cosY + 1.8;
      const fov = 1.6 / rz;
      return {
        x: cx + rx * fov * scale,
        y: cy + v.y * fov * scale,
        z: rz
      };
    });

    // Draw connecting mesh lines with neon cyan glow
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.35)';
    ctx.lineWidth = 1.2;
    edges.forEach(([i, j]) => {
      const p1 = projected[i];
      const p2 = projected[j];
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    });

    // Draw landmark vertices
    projected.forEach(p => {
      ctx.fillStyle = '#00F2FE';
      ctx.shadowColor = '#00F2FE';
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    requestAnimationFrame(render);
  }

  render();
}

// ============================================================================
// Live Monitor: Telemetry Polling & Chart Renderers
// ============================================================================
function startTelemetryPolling() {
  if (telemetryInterval) return;
  fetchTelemetry();
  telemetryInterval = setInterval(fetchTelemetry, 250);
}

async function fetchTelemetry() {
  try {
    const res = await fetch('/api/telemetry');
    if (!res.ok) return;
    const data = await res.json();
    lastTelemetryData = data;
    updateMonitorUI(data);
  } catch (err) {
    console.error('Telemetry error:', err);
  }
}

function updateMonitorUI(data) {
  activeSessionState = data.session_state;

  // Session Time
  const timerEl = document.getElementById('sessionTimerText');
  if (timerEl) timerEl.innerText = data.formatted_time || '00:00:00';

  // Current BPM
  const bpmEl = document.getElementById('liveBpmValue');
  if (bpmEl) {
    if (data.bpm && data.bpm > 0) {
      bpmEl.innerText = Math.round(data.bpm);
    } else {
      bpmEl.innerText = '--';
    }
  }

  // Signal Quality
  const qText = document.getElementById('liveQualityText');
  const qBar = document.getElementById('liveQualityProgress');
  const qPct = document.getElementById('liveQualityPct');
  const radPct = document.getElementById('radialCenterPct');

  const sqiScore = data.sqi || 78;
  const sqiCat = data.sqi_status || 'Good';

  if (qText) qText.innerText = sqiCat;
  if (qBar) qBar.style.width = `${sqiScore}%`;
  if (qPct) qPct.innerText = `${Math.round(sqiScore)}%`;
  if (radPct) radPct.innerText = `${Math.round(sqiScore)}%`;

  // Status Checklist Indicators
  const diag = data.diagnostics || {};
  const elTrack = document.getElementById('statFaceTracking');
  if (elTrack) {
    elTrack.className = `status-metric-val ${diag.is_tracking ? 'val-green' : 'val-warning'}`;
    elTrack.innerHTML = `<span class="check-dot"></span> ${diag.face_tracking}`;
  }

  const elPos = document.getElementById('statFacePosition');
  if (elPos) {
    elPos.className = `status-metric-val ${diag.is_centered ? 'val-blue' : 'val-warning'}`;
    elPos.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg> ${diag.face_position}`;
  }

  const elLight = document.getElementById('statLighting');
  if (elLight) {
    elLight.className = `status-metric-val ${diag.is_lighting_good ? 'val-green' : 'val-warning'}`;
    elLight.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/></svg> ${diag.lighting}`;
  }

  const elLms = document.getElementById('statLandmarks');
  if (elLms) {
    elLms.className = `status-metric-val ${diag.is_landmarks_detected ? 'val-green' : 'val-warning'}`;
    elLms.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> ${diag.landmarks}`;
  }

  // Draw Charts
  drawRawSignals(data.raw_signals);
  drawPulseWaveform(data.pulse_waveform);
  drawBpmTrend(data.bpm_trend);
  drawSpectrum(data.spectrum);
  drawRadialQuality(sqiScore);
}

// ----------------------------------------------------------------------------
// Canvas Waveform Chart Drawers
// ----------------------------------------------------------------------------
function resizeCanvases() {
  ['rawSignalsCanvas', 'pulseSignalCanvas', 'bpmTrendCanvas', 'spectrumCanvas', 'trendChartCanvas', 'bpmDistCanvas', 'sessCompCanvas'].forEach(id => {
    const el = document.getElementById(id);
    if (el && el.parentElement) {
      el.width = el.parentElement.clientWidth - 44;
      el.height = el.clientHeight || 130;
    }
  });
}

window.addEventListener('resize', resizeCanvases);

function drawRawSignals(signals) {
  const canvas = document.getElementById('rawSignalsCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h, ['0.6', '0.4', '0.2'], ['0', '5', '10', '15', '20']);

  if (!signals || !signals.r || signals.r.length < 2) return;

  function plotLine(arr, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    const len = arr.length;
    for (let i = 0; i < len; i++) {
      const x = 32 + (i / (len - 1)) * (w - 44);
      // Map 0.2 - 0.6 to canvas height
      const normY = (arr[i] - 0.2) / 0.4;
      const y = h - 22 - normY * (h - 36);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  plotLine(signals.r, '#FF5A87');
  plotLine(signals.g, '#00F2FE');
  plotLine(signals.b, '#3B82F6');
}

function drawPulseWaveform(pulse) {
  const canvas = document.getElementById('pulseSignalCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h, ['1.0', '0.0', '-1.0'], ['0', '5', '10', '15', '20']);

  if (!pulse || pulse.length < 2) return;

  ctx.strokeStyle = '#00F2FE';
  ctx.lineWidth = 2.0;
  ctx.shadowColor = 'rgba(0, 242, 254, 0.5)';
  ctx.shadowBlur = 8;
  ctx.beginPath();

  const len = pulse.length;
  for (let i = 0; i < len; i++) {
    const x = 32 + (i / (len - 1)) * (w - 44);
    // Map -1.0 to +1.0
    const normY = (pulse[i] + 1.0) / 2.0;
    const y = h - 22 - normY * (h - 36);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function drawBpmTrend(trend) {
  const canvas = document.getElementById('bpmTrendCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h, ['120', '100', '80', '60', '40'], ['0', '5', '10', '15', '20']);

  // Draw points & line
  const pts = (trend && trend.length > 0) ? trend : [
    { time: 0, bpm: 70 }, { time: 5, bpm: 72 }, { time: 10, bpm: 71 }, { time: 15, bpm: 73 }, { time: 20, bpm: 72 }
  ];

  ctx.strokeStyle = '#00F2FE';
  ctx.lineWidth = 1.8;
  ctx.beginPath();

  pts.forEach((pt, i) => {
    const x = 32 + (i / (pts.length - 1)) * (w - 44);
    const normY = Math.max(0, Math.min(1, (pt.bpm - 40) / 80));
    const y = h - 22 - normY * (h - 36);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dots
  pts.forEach((pt, i) => {
    const x = 32 + (i / (pts.length - 1)) * (w - 44);
    const normY = Math.max(0, Math.min(1, (pt.bpm - 40) / 80));
    const y = h - 22 - normY * (h - 36);
    ctx.fillStyle = '#00F2FE';
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawSpectrum(spec) {
  const canvas = document.getElementById('spectrumCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h, ['1.0', '0.5', '0.0'], ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0']);

  const powers = (spec && spec.powers && spec.powers.length > 0) ? spec.powers : [
    0.05, 0.1, 0.2, 0.4, 0.85, 1.0, 0.7, 0.35, 0.2, 0.1, 0.05
  ];

  const barCount = powers.length;
  const totalW = w - 44;
  const barWidth = Math.max(4, totalW / (barCount * 1.5));

  powers.forEach((p, i) => {
    const x = 32 + (i / (barCount - 1)) * (totalW - barWidth);
    const barH = p * (h - 40);
    const y = h - 22 - barH;

    const isPeak = (p >= 0.95);
    ctx.fillStyle = isPeak ? '#00F2FE' : 'rgba(0, 242, 254, 0.45)';
    ctx.fillRect(x, y, barWidth, barH);
  });
}

function drawRadialQuality(pct) {
  const canvas = document.getElementById('radialQualityCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const r = 36;

  ctx.clearRect(0, 0, w, h);

  // Background ring track
  ctx.strokeStyle = 'rgba(30, 41, 59, 0.8)';
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  // Active progress arc
  const startAngle = -Math.PI / 2;
  const endAngle = startAngle + (Math.PI * 2 * (pct / 100));

  ctx.strokeStyle = '#00F2FE';
  ctx.lineWidth = 7;
  ctx.lineCap = 'round';
  ctx.shadowColor = 'rgba(0, 242, 254, 0.6)';
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function drawGrid(ctx, w, h, yLabels, xLabels) {
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;
  ctx.font = '9px JetBrains Mono';
  ctx.fillStyle = '#64748B';

  // Horizontal Grid Lines & Y labels
  const yCount = yLabels.length;
  for (let i = 0; i < yCount; i++) {
    const y = 14 + (i / (yCount - 1)) * (h - 36);
    ctx.beginPath();
    ctx.moveTo(28, y);
    ctx.lineTo(w - 12, y);
    ctx.stroke();
    ctx.fillText(yLabels[i], 4, y + 3);
  }

  // X labels along bottom
  const xCount = xLabels.length;
  for (let j = 0; j < xCount; j++) {
    const x = 32 + (j / (xCount - 1)) * (w - 44);
    ctx.fillText(xLabels[j], x - 4, h - 8);
  }
}

// ============================================================================
// Monitoring Engine Control Buttons
// ============================================================================
async function startMonitoring() {
  await fetch('/api/session/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
  startTelemetryPolling();
}

async function pauseMonitoring() {
  await fetch('/api/session/pause', { method: 'POST' });
}

async function resumeMonitoring() {
  await fetch('/api/session/resume', { method: 'POST' });
}

async function stopMonitoring() {
  const res = await fetch('/api/session/stop', { method: 'POST' });
  const data = await res.json();
  if (data && data.stats) {
    alert(`Session completed and saved! Average Heart Rate: ${data.stats.avg_bpm || '--'} BPM`);
  }
}

async function restartMonitoring() {
  await fetch('/api/session/restart', { method: 'POST' });
}

// ============================================================================
// Page 3: Analytics View
// ============================================================================
async function loadAnalyticsData(rangeFilter) {
  try {
    const res = await fetch(`/api/analytics?range_filter=${rangeFilter}`);
    if (!res.ok) return;
    const data = await res.json();

    // KPI cards
    document.getElementById('kpiAvgBpm').innerText = data.kpi.avg_bpm;
    document.getElementById('kpiMinBpm').innerText = data.kpi.min_bpm;
    document.getElementById('kpiMaxBpm').innerText = data.kpi.max_bpm;
    document.getElementById('kpiDuration').innerText = data.kpi.duration;

    // Draw Heart Rate Trend Canvas
    drawAnalyticsTrend(data.heart_rate_trend);

    // Draw Signal Quality Donut
    drawAnalyticsDonut(data.signal_quality);

    // Draw BPM Distribution Histogram
    drawBpmDistribution(data.bpm_distribution);

    // Draw Session Comparison
    drawSessionComparison(data.session_comparison);

  } catch (err) {
    console.error('Error loading analytics:', err);
  }
}

function drawAnalyticsTrend(trend) {
  const canvas = document.getElementById('trendChartCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h, ['120', '100', '80', '60', '40'], trend.labels);

  const vals = trend.values;
  ctx.strokeStyle = '#00F2FE';
  ctx.lineWidth = 2.2;
  ctx.beginPath();

  vals.forEach((v, i) => {
    const x = 32 + (i / (vals.length - 1)) * (w - 44);
    const normY = (v - 40) / 80;
    const y = h - 22 - normY * (h - 36);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dots
  vals.forEach((v, i) => {
    const x = 32 + (i / (vals.length - 1)) * (w - 44);
    const normY = (v - 40) / 80;
    const y = h - 22 - normY * (h - 36);
    ctx.fillStyle = '#00F2FE';
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawAnalyticsDonut(sq) {
  const canvas = document.getElementById('analyticsDonutCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const r = 48;
  const lw = 14;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Angles: Good (78%), Moderate (15%), Poor (7%)
  const segments = [
    { pct: sq.good_pct, color: '#10B981' },
    { pct: sq.moderate_pct, color: '#F59E0B' },
    { pct: sq.poor_pct, color: '#EF4444' }
  ];

  let currentAngle = -Math.PI / 2;
  ctx.lineWidth = lw;

  segments.forEach(seg => {
    const arcAngle = (seg.pct / 100) * (Math.PI * 2);
    ctx.strokeStyle = seg.color;
    ctx.beginPath();
    ctx.arc(cx, cy, r, currentAngle, currentAngle + arcAngle);
    ctx.stroke();
    currentAngle += arcAngle;
  });
}

function drawBpmDistribution(dist) {
  const canvas = document.getElementById('bpmDistCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h, ['15', '10', '5', '0'], dist.bins);

  const counts = dist.counts;
  const barCount = counts.length;
  const totalW = w - 44;
  const barWidth = totalW / (barCount * 1.6);

  counts.forEach((c, i) => {
    const x = 32 + (i / (barCount - 1)) * (totalW - barWidth);
    const barH = (c / 15) * (h - 40);
    const y = h - 22 - barH;
    ctx.fillStyle = '#00F2FE';
    ctx.fillRect(x, y, barWidth, barH);
  });
}

function drawSessionComparison(comp) {
  const canvas = document.getElementById('sessCompCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h, ['120', '100', '80', '60', '40'], ['0', '2', '4', '6', '8', '10', '12']);

  function plotComp(arr, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = 32 + (i / (arr.length - 1)) * (w - 44);
      const normY = (v - 40) / 80;
      const y = h - 22 - normY * (h - 36);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    arr.forEach((v, i) => {
      const x = 32 + (i / (arr.length - 1)) * (w - 44);
      const normY = (v - 40) / 80;
      const y = h - 22 - normY * (h - 36);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  plotComp(comp.session_1, '#00F2FE');
  plotComp(comp.session_2, '#C084FC');
}

// ============================================================================
// Page 4: Session History & Export
// ============================================================================
async function loadSessionsList() {
  try {
    const res = await fetch('/api/sessions');
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById('sessionsTableBody');
    if (!tbody) return;

    tbody.innerHTML = '';
    (data.sessions || []).forEach(s => {
      const tr = document.createElement('tr');
      const durSec = s.duration_sec || 0;
      const mins = Math.floor(durSec / 60);
      const secs = Math.floor(durSec % 60);
      const durStr = `${mins}:${secs < 10 ? '0' : ''}${secs}`;

      tr.innerHTML = `
        <td>${s.start_time || 'Recent'}</td>
        <td>${durStr}</td>
        <td><b>${s.avg_bpm || '--'}</b></td>
        <td><span class="pill-badge badge-valid"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg> Valid</span></td>
        <td>
          <div class="table-action-btns">
            <button class="icon-btn" title="View Details" onclick="viewSessionDetails('${s.session_id}')">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
            </button>
            <button class="icon-btn" title="Download CSV" onclick="downloadSessionCsv('${s.session_id}')">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            </button>
            <button class="icon-btn delete-btn" title="Delete Session" onclick="deleteSessionRecord('${s.session_id}')">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error('Error loading sessions:', err);
  }
}

function viewSessionDetails(sid) {
  fetch(`/api/sessions/${sid}`)
    .then(r => r.json())
    .then(data => {
      const s = data.session;
      alert(`Session ID: ${s.session_id}\nParticipant: ${s.user_label}\nAverage Heart Rate: ${s.avg_bpm} BPM\nObserved Range: ${s.min_bpm} - ${s.max_bpm} BPM\nUsable Signal: ${s.usable_pct}%`);
    });
}

function downloadSessionCsv(sid) {
  window.open(`/api/reports/csv?session_id=${sid}`, '_blank');
}

async function deleteSessionRecord(sid) {
  if (confirm(`Delete session ${sid}?`)) {
    await fetch(`/api/sessions/${sid}`, { method: 'DELETE' });
    loadSessionsList();
  }
}

async function clearAllSessions() {
  if (confirm('Clear all session records from database?')) {
    await fetch('/api/sessions/clear_all', { method: 'POST' });
    loadSessionsList();
  }
}

function exportData(format) {
  if (format === 'csv') {
    window.open('/api/reports/csv', '_blank');
  } else if (format === 'pdf') {
    window.open('/api/reports/pdf', '_blank');
  }
}

// ============================================================================
// Page 5: Settings Management
// ============================================================================
async function loadSettings() {
  try {
    const res = await fetch('/api/settings');
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('settingAlgorithmSelect').value = data.algorithm;
    document.getElementById('settingDurationSelect').value = data.duration_sec;
    document.getElementById('settingWindowSelect').value = data.signal_window_sec;
    document.getElementById('settingRefreshSelect').value = data.refresh_interval_sec;
    document.getElementById('toggleLandmarkMesh').checked = data.show_landmark_mesh;
    document.getElementById('toggleSignalGraphs').checked = data.show_signal_graphs;
    document.getElementById('toggleSaveSessions').checked = data.save_sessions;
    document.getElementById('toggleAutoExport').checked = data.auto_export_csv;

    // Load available hardware camera devices
    const camRes = await fetch('/api/cameras');
    if (camRes.ok) {
      const camData = await camRes.json();
      const sel = document.getElementById('cameraSelect');
      const setSel = document.getElementById('settingCameraSelect');
      if (sel && setSel) {
        sel.innerHTML = '';
        setSel.innerHTML = '';
        camData.cameras.forEach(idx => {
          const opt1 = document.createElement('option');
          opt1.value = idx;
          opt1.innerText = `Webcam ${idx} ${idx === 0 ? '(Integrated)' : ''}`;
          sel.appendChild(opt1);

          const opt2 = document.createElement('option');
          opt2.value = idx;
          opt2.innerText = `Webcam ${idx} ${idx === 0 ? '(Default)' : ''}`;
          setSel.appendChild(opt2);
        });
        sel.value = data.camera_index;
        setSel.value = data.camera_index;
      }
    }

  } catch (err) {
    console.error('Error loading settings:', err);
  }
}

async function updateSettingsField(field, value) {
  const payload = {};
  payload[field] = value;
  await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}

function onCameraChange(camIdx) {
  updateSettingsField('camera_index', camIdx);
}

// ============================================================================
// Application Startup
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initWireframeAnimation();
  navigateTo('home');
});
