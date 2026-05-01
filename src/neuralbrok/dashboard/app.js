/* ═══════════════════════════════════════════════════════════════════════════
   NeuralBroker v2.0 — Dashboard SPA Logic
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  const POLL_MS = 800;
  const MAX_FEED = 25;
  const CIRC = 2 * Math.PI * 70;

  // ── State ─────────────────────────────────────────────────────────────────
  let currentView = 'overview';
  let cachedAgents = [];
  let cachedFit = null;
  let cachedProviders = [];

  // ── Navigation ────────────────────────────────────────────────────────────
  function navigate(view) {
    currentView = view;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const el = document.getElementById('view-' + view);
    if (el) el.classList.add('active');
    const nav = document.querySelector('[data-view="' + view + '"]');
    if (nav) nav.classList.add('active');
    // Lazy-load data for specific views
    if (view === 'agents' && cachedAgents.length === 0) fetchAgents();
    if (view === 'models') fetchFit();
    if (view === 'providers') fetchDetectedProviders();
  }

  document.addEventListener('click', function (e) {
    const nav = e.target.closest('.nav-item');
    if (nav && nav.dataset.view) {
      e.preventDefault();
      navigate(nav.dataset.view);
    }
    const modeBtn = e.target.closest('.mode-btn');
    if (modeBtn && modeBtn.dataset.mode) setMode(modeBtn.dataset.mode);
  });

  // ── VRAM Gauge ────────────────────────────────────────────────────────────
  function updateGauge(util) {
    const arc = document.getElementById('gauge-arc');
    const pct = document.getElementById('gauge-pct');
    if (!arc || !pct) return;
    const p = Math.round(util * 100);
    arc.style.strokeDashoffset = CIRC * (1 - util);
    pct.textContent = p + '%';
    if (util > 0.9) arc.style.stroke = 'var(--hot)';
    else if (util > 0.7) arc.style.stroke = 'var(--accent)';
    else arc.style.stroke = 'var(--ok)';
  }

  // ── API Helpers ───────────────────────────────────────────────────────────
  async function api(path) {
    try { const r = await fetch(path); return await r.json(); }
    catch (e) { return null; }
  }

  async function apiPost(path, body) {
    try {
      const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      return await r.json();
    } catch (e) { return null; }
  }

  // ── Poll: VRAM ────────────────────────────────────────────────────────────
  async function fetchVram() {
    const d = await api('/nb/vram');
    if (d && d.gpu_0) updateGauge(d.gpu_0.utilization);
  }

  // ── Poll: Stats ───────────────────────────────────────────────────────────
  async function fetchStats() {
    const s = await api('/nb/stats');
    if (!s) return;
    setText('stat-total', (s.total_requests || 0).toLocaleString());
    setText('stat-local', (s.local_pct || 0) + '%');
    setText('stat-saved', '$' + (s.total_saved || 0).toFixed(2));
    setText('stat-cloud', '$' + (s.total_cost_cloud || 0).toFixed(2));
    if (s.routing_mode) updateModeBadge(s.routing_mode);
  }

  // ── Poll: Feed ────────────────────────────────────────────────────────────
  async function fetchFeed() {
    const d = await api('/nb/routing-log');
    const feed = document.getElementById('routing-feed');
    if (!feed || !d) return;
    const decisions = (d.decisions || []).slice(-MAX_FEED).reverse();
    if (decisions.length === 0) {
      feed.innerHTML = '<div class="empty-state">Waiting for routing decisions…</div>';
      return;
    }
    feed.innerHTML = decisions.map(function (d) {
      const cls = (d.backend || '').includes('ollama') || (d.mode === 'smart' && !d.is_cloud) ? 'local' : 'cloud';
      const barPct = Math.min(100, Math.round((d.latency_ms || 0) / 200 * 100));
      return '<div class="feed-item"><div class="feed-row"><div>' +
        '<span class="feed-backend ' + cls + '">' + (d.backend || '—') + '</span>' +
        '<div class="feed-meta">' + (d.mode || '') + ' · ' + (d.latency_ms || 0) + 'ms</div></div>' +
        '<span class="feed-reason">' + (d.reason || '').substring(0, 40) + '</span></div>' +
        '<div class="waterfall-wrap"><div class="waterfall-bar ' + cls + '" style="width:' + barPct + '%"></div></div></div>';
    }).join('');
  }

  // ── Poll: Providers ───────────────────────────────────────────────────────
  async function fetchProviders() {
    const d = await api('/nb/providers');
    const list = document.getElementById('providers-sidebar');
    if (!list || !d) return;
    cachedProviders = d.providers || [];
    list.innerHTML = cachedProviders.map(function (p) {
      const led = p.healthy ? 'ok' : 'fail';
      return '<div class="provider-card"><span class="provider-led ' + led + '"></span>' +
        '<div><div class="provider-name">' + p.name + '</div>' +
        '<div class="provider-type">' + p.type + '</div></div></div>';
    }).join('');
  }

  // ── Fetch: Agents ─────────────────────────────────────────────────────────
  async function fetchAgents() {
    const d = await api('/nb/agents');
    if (!d) return;
    cachedAgents = d.agents || [];
    const grid = document.getElementById('agents-grid');
    if (!grid) return;
    grid.innerHTML = cachedAgents.map(function (a) {
      const caps = (a.capabilities || []).map(c => '<span class="cap-tag">' + c + '</span>').join('');
      return '<div class="agent-card" style="border-left:3px solid ' + (a.color || 'var(--accent)') + '">' +
        '<div class="agent-header"><span class="agent-icon">' + (a.icon || '🤖') + '</span>' +
        '<span class="agent-name">' + a.name + '</span></div>' +
        '<div class="agent-role">' + (a.role || '') + '</div>' +
        '<div class="agent-caps">' + caps + '</div></div>';
    }).join('');
  }

  // ── Fetch: NeuralFit Model Scoring ───────────────────────────────────────────
  async function fetchFit() {
    const d = await api('/nb/fit?max_results=20');
    if (!d) return;
    cachedFit = d;
    const hw = document.getElementById('fit-hardware');
    if (hw && d.hardware) {
      hw.innerHTML = '<span class="hw-badge-sidebar"><span class="hw-dot"></span>' +
        (d.hardware.gpu || 'Unknown') + ' · ' + (d.hardware.vram_gb || 0) + ' GB VRAM</span>';
    }
    const tbody = document.getElementById('fit-tbody');
    if (!tbody) return;
    tbody.innerHTML = (d.models || []).map(function (m) {
      const sc = m.scores || {};
      return '<tr>' +
        '<td class="model-name">' + m.name + '</td>' +
        '<td>' + m.params_b + 'B</td>' +
        '<td>' + m.best_quant + '</td>' +
        '<td><span class="fit-badge ' + m.fit_level + '">' + m.fit_level + '</span></td>' +
        '<td>' + scoreCell(sc.quality, 'quality') + '</td>' +
        '<td>' + scoreCell(sc.speed, 'speed') + '</td>' +
        '<td>' + scoreCell(sc.fit, 'fit') + '</td>' +
        '<td>' + scoreCell(sc.context, 'context') + '</td>' +
        '<td style="font-weight:600;color:var(--accent)">' + (sc.composite || 0).toFixed(1) + '</td>' +
        '<td>' + (m.estimated_tok_s || 0) + ' t/s</td>' +
        '<td>' + (m.vram_needed_gb || 0) + ' GB</td>' +
        '<td>' + (m.is_installed ? '<span style="color:var(--ok)">✓</span>' : '') + '</td>' +
        '</tr>';
    }).join('');
  }

  function scoreCell(val, cls) {
    const v = val || 0;
    return v.toFixed(0) + '<span class="score-bar-wrap"><span class="score-bar ' + cls + '" style="width:' + v + '%"></span></span>';
  }

  // ── Fetch: Detected Providers ─────────────────────────────────────────────
  async function fetchDetectedProviders() {
    const d = await api('/nb/providers/detect');
    const grid = document.getElementById('detected-providers');
    if (!grid || !d) return;
    const entries = Object.entries(d.providers || {});
    grid.innerHTML = entries.map(function (e) {
      const name = e[0], p = e[1];
      const ledCls = p.available ? 'ok' : 'unknown';
      const status = p.available ? 'Available' : 'Not detected';
      return '<div class="provider-card"><span class="provider-led ' + ledCls + '"></span>' +
        '<div><div class="provider-name">' + name + '</div>' +
        '<div class="provider-type">' + p.type + ' · ' + status + '</div></div>' +
        (p.cost_per_1k ? '<span class="provider-cost">$' + p.cost_per_1k.toFixed(5) + '/1k</span>' : '') +
        '</div>';
    }).join('');
  }

  // ── Fetch: Hardware ───────────────────────────────────────────────────────
  async function fetchHardware() {
    const d = await api('/nb/hardware');
    const el = document.getElementById('hw-sidebar');
    if (!el || !d) return;
    el.innerHTML = '<span class="hw-dot"></span>' + (d.model || 'Unknown GPU') +
      '<br><span style="color:var(--ink-4)">' + (d.vram_gb || 0) + ' GB · ' +
      (d.bandwidth_gbps || '?') + ' GB/s</span>';
  }

  // ── Mode Switcher ─────────────────────────────────────────────────────────
  async function setMode(mode) {
    await apiPost('/nb/mode', { mode: mode });
    updateModeBadge(mode);
  }

  function updateModeBadge(mode) {
    const badge = document.getElementById('mode-badge');
    if (badge) badge.textContent = mode + '-mode';
    document.querySelectorAll('.mode-btn').forEach(function (b) {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  // ── Poll Loop ─────────────────────────────────────────────────────────────
  function poll() {
    fetchVram();
    fetchStats();
    if (currentView === 'overview' || currentView === 'routing') fetchFeed();
    if (currentView === 'overview') fetchProviders();
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  fetchHardware();
  fetchAgents();
  poll();
  setInterval(poll, POLL_MS);

  // Start on overview
  navigate('overview');
})();
