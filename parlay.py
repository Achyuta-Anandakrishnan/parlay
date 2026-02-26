import datetime as dt
import math
import os

import requests
from flask import Flask, jsonify, render_template_string
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# CONFIG
# =========================

DAYS_AHEAD = 3
MAX_LEGS = 8
MIN_EDGE = 0.03
TIMEOUT = 15
MAX_RETRIES = 2

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
MARKET_SOURCE = "espn_moneyline_draftkings"
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()

LEAGUES = [
    ("Basketball", "NBA", "basketball", "nba"),
    ("Basketball", "WNBA", "basketball", "wnba"),
    ("Basketball", "NCAAM", "basketball", "mens-college-basketball"),
    ("Basketball", "NCAAW", "basketball", "womens-college-basketball"),
    ("Football", "NFL", "football", "nfl"),
    ("Football", "NCAAF", "football", "college-football"),
    ("Hockey", "NHL", "hockey", "nhl"),
    ("Baseball", "MLB", "baseball", "mlb"),
    ("Soccer", "EPL", "soccer", "eng.1"),
    ("Soccer", "La Liga", "soccer", "esp.1"),
    ("Soccer", "Bundesliga", "soccer", "ger.1"),
    ("Soccer", "Serie A", "soccer", "ita.1"),
    ("Soccer", "Ligue 1", "soccer", "fra.1"),
    ("Soccer", "MLS", "soccer", "usa.1"),
    ("Soccer", "UCL", "soccer", "uefa.champions"),
]

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Parlay Builder Pro</title>
  <style>
    :root {
      --bg: #090f16;
      --panel: #111a25;
      --panel-2: #162231;
      --line: #25374a;
      --text: #edf3fb;
      --muted: #8ea4bc;
      --green: #84ff43;
      --teal: #35d9b5;
      --blue: #55a9ff;
      --red: #ff6c85;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Bahnschrift", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(900px 450px at 88% -15%, rgba(53, 217, 181, .15), transparent 60%),
        radial-gradient(760px 380px at -5% 12%, rgba(85, 169, 255, .13), transparent 58%),
        linear-gradient(180deg, #08111d, #090f16 40%, #070c12);
      min-height: 100vh;
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 25;
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
      background: rgba(6, 10, 16, .82);
    }
    .topbar-inner {
      max-width: 1320px;
      margin: 0 auto;
      padding: 12px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .brand {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-weight: 800;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    .brand-dot {
      width: 10px;
      height: 10px;
      border-radius: 99px;
      background: linear-gradient(180deg, var(--green), #4fdb14);
      box-shadow: 0 0 16px rgba(132, 255, 67, .7);
    }
    .top-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .refresh-btn {
      border: 0;
      border-radius: 9px;
      padding: 10px 13px;
      font-weight: 800;
      letter-spacing: .05em;
      text-transform: uppercase;
      font-size: 12px;
      color: #051507;
      background: linear-gradient(180deg, #a4ff76, #72f83b);
      cursor: pointer;
    }
    .refresh-btn:disabled { opacity: .6; cursor: not-allowed; }

    .shell {
      max-width: 1320px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 370px;
      gap: 18px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 15px;
      background: linear-gradient(180deg, rgba(22, 34, 49, .97), rgba(16, 26, 37, .97));
      box-shadow: 0 14px 34px rgba(0, 0, 0, .33);
    }

    .hero {
      padding: 16px;
      margin-bottom: 12px;
    }
    .kicker {
      color: var(--teal);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .09em;
      text-transform: uppercase;
    }
    h1 {
      margin: 7px 0;
      font-size: 28px;
      line-height: 1.13;
      letter-spacing: .01em;
    }
    .hero p { margin: 0; color: var(--muted); }
    .meta {
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .meta-box {
      border: 1px solid var(--line);
      border-radius: 11px;
      padding: 10px;
      background: rgba(9, 15, 22, .5);
    }
    .meta-label {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .07em;
      font-size: 10px;
      margin-bottom: 4px;
    }
    .meta-value { font-size: 17px; font-weight: 700; }

    .section {
      padding: 11px;
      margin-bottom: 12px;
    }
    .section-title {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin: 0 0 9px;
    }

    .sport-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }
    .sport-tab {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 11px;
      font-size: 12px;
      background: rgba(9, 15, 22, .55);
      color: #d0deec;
      cursor: pointer;
    }
    .sport-tab.active {
      border-color: #4cbcfc;
      color: #dff0ff;
      background: linear-gradient(180deg, rgba(85, 169, 255, .24), rgba(85, 169, 255, .08));
    }

    .picks, .games { display: grid; gap: 10px; }
    .pick, .game {
      border: 1px solid #2b3e54;
      border-radius: 11px;
      background: #1a2736;
      padding: 11px;
    }
    .pick {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
    }
    .leg {
      display: inline-block;
      color: var(--teal);
      text-transform: uppercase;
      letter-spacing: .07em;
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .match {
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 3px;
    }
    .market {
      font-size: 13px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .edge-box {
      min-width: 114px;
      border-radius: 10px;
      border: 1px solid #375a2c;
      background: linear-gradient(180deg, rgba(132, 255, 67, .2), rgba(132, 255, 67, .07));
      text-align: center;
      padding: 8px;
      align-self: center;
    }
    .edge {
      color: var(--green);
      font-size: 15px;
      font-weight: 800;
    }
    .prob {
      color: #c0d0df;
      font-size: 11px;
      margin-top: 4px;
    }

    .sports-columns {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 11px;
    }
    .game-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }
    .league {
      color: var(--teal);
      text-transform: uppercase;
      letter-spacing: .06em;
      font-size: 11px;
      font-weight: 700;
    }
    .time { color: var(--muted); font-size: 12px; }

    .moneyline {
      margin-top: 8px;
      border-top: 1px dashed #334a63;
      padding-top: 8px;
    }
    .moneyline-title {
      color: #a6bdd4;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-bottom: 6px;
    }
    .odds-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .odds-btn {
      border: 1px solid var(--line);
      border-radius: 9px;
      background: rgba(9, 15, 22, .7);
      color: #deebf8;
      text-align: center;
      font-size: 12px;
      padding: 8px 6px;
      cursor: pointer;
      transition: border-color .15s ease, transform .15s ease;
    }
    .odds-btn .team { display: block; font-weight: 700; margin-bottom: 3px; }
    .odds-btn .line { display: block; color: #b8c8d8; }
    .odds-btn .imp { display: block; color: #8ea7be; font-size: 11px; margin-top: 2px; }
    .odds-btn:hover { transform: translateY(-1px); border-color: #4f6985; }
    .odds-btn.active {
      border-color: #76ff3e;
      background: linear-gradient(180deg, rgba(132, 255, 67, .22), rgba(132, 255, 67, .08));
    }
    .odds-btn[disabled] {
      cursor: not-allowed;
      opacity: .46;
      transform: none;
    }

    .slip {
      position: sticky;
      top: 74px;
      padding: 14px;
      height: fit-content;
    }
    .slip-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }
    .slip h2 {
      margin: 0;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: .05em;
    }
    .status-line { color: var(--muted); font-size: 13px; margin-bottom: 10px; }
    .badge {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 11px;
      color: #bfd1e4;
      letter-spacing: .07em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .stake-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      margin-bottom: 10px;
      align-items: center;
    }
    .stake-row input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: rgba(9, 15, 22, .85);
      color: var(--text);
      padding: 9px;
    }

    .slip-list {
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }
    .slip-leg {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(9, 15, 22, .8);
      padding: 9px;
    }
    .slip-leg-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 3px;
      font-size: 12px;
      color: var(--muted);
    }
    .slip-leg-main { font-weight: 700; font-size: 13px; }

    .mini-btn {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(9, 15, 22, .8);
      color: #d2e2f3;
      font-size: 11px;
      cursor: pointer;
      padding: 6px 9px;
    }
    .remove-leg {
      border-color: #603442;
      background: rgba(255, 108, 133, .1);
      color: #ffc2cd;
    }
    .slip-controls {
      display: flex;
      justify-content: flex-end;
      margin-bottom: 10px;
    }

    .tickets { display: grid; gap: 8px; margin-bottom: 10px; }
    .ticket-row {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: rgba(9, 15, 22, .75);
      font-size: 13px;
    }
    .t-label { color: var(--muted); }
    .t-value { font-weight: 700; }

    .google-wrap {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(9, 15, 22, .72);
      padding: 10px;
      margin-bottom: 10px;
    }
    .user-card {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .user-meta {
      min-width: 0;
      overflow: hidden;
    }
    .user-name { font-size: 13px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .user-email { font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .google-note { color: var(--muted); font-size: 12px; line-height: 1.4; }

    .empty {
      border: 1px dashed var(--line);
      border-radius: 11px;
      padding: 16px 12px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
    }
    .error { color: var(--red); }

    @media (max-width: 1140px) {
      .shell { grid-template-columns: 1fr; }
      .slip { position: static; }
      .sports-columns { grid-template-columns: 1fr; }
    }
    @media (max-width: 780px) {
      .meta { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand"><span class="brand-dot"></span>Parlay Builder Pro</div>
      <div class="top-actions">
        <button id="runBtn" class="refresh-btn">Refresh Board</button>
      </div>
    </div>
  </header>

  <main class="shell">
    <section>
      <article class="hero card">
        <div class="kicker">Market Board</div>
        <h1>Clean board + manual slip, Kalshi-style layout</h1>
        <p>Pick a sport tab, choose moneyline legs, and build your own parlay live.</p>
        <div class="meta">
          <div class="meta-box"><div class="meta-label">Games</div><div id="mGames" class="meta-value">-</div></div>
          <div class="meta-box"><div class="meta-label">Sports</div><div id="mSports" class="meta-value">-</div></div>
          <div class="meta-box"><div class="meta-label">Markets</div><div id="mMarkets" class="meta-value">-</div></div>
          <div class="meta-box"><div class="meta-label">Auto Picks</div><div id="mPicks" class="meta-value">-</div></div>
        </div>
      </article>

      <article class="section card">
        <div class="section-title">Sports</div>
        <div id="sportTabs" class="sport-tabs"></div>
      </article>

      <article class="section card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:9px;">
          <div class="section-title" style="margin:0;">Suggested Picks</div>
          <button id="addAllProjectedBtn" class="mini-btn">Add All Projected</button>
        </div>
        <div id="picks" class="empty">No picks loaded yet.</div>
      </article>

      <article class="section card">
        <div class="section-title">Games (Moneyline)</div>
        <div id="gamesBySport" class="empty">No games loaded yet.</div>
      </article>
    </section>

    <aside class="slip card">
      <div class="slip-head">
        <h2>Bet Slip</h2>
      </div>
      <div id="status" class="status-line">Ready to build.</div>
      <div id="sourceBadge" class="badge">Source: -</div>

      <div class="google-wrap">
        <div id="googleAuthArea"></div>
      </div>

      <div class="stake-row">
        <label for="stakeInput" class="t-label">Stake ($)</label>
        <input id="stakeInput" type="number" min="1" step="1" value="10" />
      </div>

      <div class="slip-controls">
        <button id="clearSlipBtn" class="mini-btn">Clear Slip</button>
      </div>

      <div id="manualSlip" class="slip-list"></div>

      <div class="tickets">
        <div class="ticket-row"><span class="t-label">Legs</span><span id="tLegs" class="t-value">0</span></div>
        <div class="ticket-row"><span class="t-label">Parlay Odds</span><span id="tParlayOdds" class="t-value">N/A</span></div>
        <div class="ticket-row"><span class="t-label">To Win</span><span id="tToWin" class="t-value">$0.00</span></div>
      </div>
      <div class="empty">Select ML buttons in the game board to build your custom parlay.</div>
    </aside>
  </main>

  <script src="https://accounts.google.com/gsi/client" async defer></script>
  <script>
    const GOOGLE_CLIENT_ID = "{{ google_client_id|e }}";

    const btn = document.getElementById('runBtn');
    const picksEl = document.getElementById('picks');
    const gamesEl = document.getElementById('gamesBySport');
    const tabsEl = document.getElementById('sportTabs');
    const statusEl = document.getElementById('status');
    const sourceBadge = document.getElementById('sourceBadge');
    const googleAuthArea = document.getElementById('googleAuthArea');

    const mGames = document.getElementById('mGames');
    const mSports = document.getElementById('mSports');
    const mMarkets = document.getElementById('mMarkets');
    const mPicks = document.getElementById('mPicks');

    const tLegs = document.getElementById('tLegs');
    const tParlayOdds = document.getElementById('tParlayOdds');
    const tToWin = document.getElementById('tToWin');
    const stakeInput = document.getElementById('stakeInput');
    const manualSlipEl = document.getElementById('manualSlip');
    const clearSlipBtn = document.getElementById('clearSlipBtn');
    const addAllProjectedBtn = document.getElementById('addAllProjectedBtn');

    let manualLegs = [];
    let selectedSport = 'All';
    let latestData = null;
    let googleInitialized = false;
    let googleTokenClient = null;
    let googleTokenClientId = '';

    function esc(text) {
      return String(text ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    function fmtTime(iso) {
      if (!iso) return '-';
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return '-';
      return d.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
    }

    function oddsToImpliedPct(oddsText) {
      if (!oddsText) return null;
      const raw = String(oddsText).replace('−', '-').replace('+', '').trim();
      const val = Number(raw);
      if (!Number.isFinite(val) || val === 0) return null;
      if (val > 0) return (100 / (val + 100)) * 100;
      return (Math.abs(val) / (Math.abs(val) + 100)) * 100;
    }

    function parseAmericanOdds(oddsText) {
      if (!oddsText) return null;
      const val = Number(String(oddsText).replace('−', '-').trim());
      if (!Number.isFinite(val) || val === 0) return null;
      if (val > 0) return 1 + (val / 100);
      return 1 + (100 / Math.abs(val));
    }

    function displayOdds(oddsText) {
      return oddsText ? oddsText : 'Not out yet';
    }

    function decimalToAmerican(decimalOdds) {
      if (!Number.isFinite(decimalOdds) || decimalOdds <= 1) return 'N/A';
      if (decimalOdds >= 2) return `+${Math.round((decimalOdds - 1) * 100)}`;
      return `-${Math.round(100 / (decimalOdds - 1))}`;
    }

    function decodeJwtPayload(token) {
      try {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const json = decodeURIComponent(atob(base64).split('').map((c) =>
          '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)
        ).join(''));
        return JSON.parse(json);
      } catch {
        return null;
      }
    }

    function loadUser() {
      try {
        const raw = localStorage.getItem('pb_user');
        return raw ? JSON.parse(raw) : null;
      } catch {
        return null;
      }
    }

    function saveUser(user) {
      localStorage.setItem('pb_user', JSON.stringify(user));
    }

    function clearUser() {
      localStorage.removeItem('pb_user');
    }

    function getEffectiveGoogleClientId() {
      const localId = (localStorage.getItem('pb_google_client_id') || '').trim();
      return localId || GOOGLE_CLIENT_ID;
    }

    function saveGoogleClientId(clientId) {
      localStorage.setItem('pb_google_client_id', clientId.trim());
    }

    function initGoogleIdentity() {
      if (googleInitialized) return true;
      if (!(window.google && google.accounts && google.accounts.id)) return false;

      google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: (response) => {
          const payload = decodeJwtPayload(response.credential);
          if (payload) {
            saveUser({
              name: payload.name || '',
              email: payload.email || '',
              picture: payload.picture || '',
            });
            renderAuth();
          }
        }
      });

      googleInitialized = true;
      return true;
    }

    async function handleGoogleSignInClick() {
      if (!(window.google && google.accounts && google.accounts.oauth2)) {
        statusEl.classList.add('error');
        statusEl.textContent = 'Google script is still loading. Try again.';
        return;
      }

      let clientId = getEffectiveGoogleClientId();
      if (!clientId) {
        const entered = window.prompt('Enter your Google OAuth Web Client ID:');
        if (!entered || !entered.trim()) return;
        saveGoogleClientId(entered.trim());
        clientId = entered.trim();
      }

      if (!googleTokenClient || googleTokenClientId !== clientId) {
        googleTokenClientId = clientId;
        googleTokenClient = google.accounts.oauth2.initTokenClient({
          client_id: clientId,
          scope: 'openid email profile',
          callback: async (tokenResponse) => {
            if (!tokenResponse || tokenResponse.error || !tokenResponse.access_token) {
              statusEl.classList.add('error');
              statusEl.textContent = 'Google sign-in failed.';
              return;
            }
            try {
              const userRes = await fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
                headers: { Authorization: `Bearer ${tokenResponse.access_token}` },
              });
              const user = await userRes.json();
              if (!user || !user.email) {
                throw new Error('Missing profile info');
              }
              saveUser({
                name: user.name || '',
                email: user.email || '',
                picture: user.picture || '',
              });
              statusEl.classList.remove('error');
              statusEl.textContent = `Signed in as ${user.email}`;
              renderAuth();
            } catch {
              statusEl.classList.add('error');
              statusEl.textContent = 'Google profile fetch failed.';
            }
          },
        });
      }

      googleTokenClient.requestAccessToken({ prompt: 'consent' });
    }

    function renderAuth() {
      const user = loadUser();
      if (user) {
        googleAuthArea.innerHTML = `
          <div class="user-card">
            <div class="user-meta">
              <div class="user-name">${esc(user.name || 'Signed In')}</div>
              <div class="user-email">${esc(user.email || '')}</div>
            </div>
            <button id="signOutBtn" class="mini-btn">Sign Out</button>
          </div>
        `;
        const outBtn = document.getElementById('signOutBtn');
        outBtn.addEventListener('click', () => {
          clearUser();
          if (window.google && google.accounts && google.accounts.id) {
            google.accounts.id.disableAutoSelect();
          }
          renderAuth();
        });
        return;
      }

      googleAuthArea.innerHTML = `
        <div style="display:grid; gap:8px;">
          <button id="signInGoogleBtn" class="mini-btn" style="display:flex;align-items:center;justify-content:center;gap:8px;">
            <svg width="16" height="16" viewBox="0 0 18 18" aria-hidden="true">
              <path fill="#EA4335" d="M9 7.36v3.54h4.95c-.2 1.14-.86 2.1-1.8 2.75l2.9 2.25c1.7-1.57 2.68-3.88 2.68-6.63 0-.64-.06-1.26-.17-1.85H9z"/>
              <path fill="#34A853" d="M9 18c2.43 0 4.46-.8 5.95-2.17l-2.9-2.25c-.8.54-1.83.87-3.05.87-2.35 0-4.34-1.58-5.06-3.71H.94v2.33A9 9 0 0 0 9 18z"/>
              <path fill="#4A90E2" d="M3.94 10.74A5.4 5.4 0 0 1 3.66 9c0-.6.1-1.17.28-1.74V4.93H.94A9 9 0 0 0 0 9c0 1.45.35 2.82.94 4.07l3-2.33z"/>
              <path fill="#FBBC05" d="M9 3.58c1.32 0 2.5.45 3.43 1.34l2.58-2.58C13.45.9 11.43 0 9 0A9 9 0 0 0 .94 4.93l3 2.33C4.66 5.16 6.65 3.58 9 3.58z"/>
            </svg>
            Sign In With Google
          </button>
        </div>
      `;
      const signInBtn = document.getElementById('signInGoogleBtn');
      signInBtn.addEventListener('click', handleGoogleSignInClick);
    }

    function isSelected(eventId, side) {
      return manualLegs.some((l) => l.eventId === eventId && l.side === side);
    }

    function syncOddsButtons() {
      document.querySelectorAll('.odds-btn').forEach((el) => {
        const eventId = el.dataset.eventId;
        const side = el.dataset.side;
        el.classList.toggle('active', isSelected(eventId, side));
      });
    }

    function addOrReplaceLeg(leg) {
      const idx = manualLegs.findIndex((l) => l.eventId === leg.eventId);
      if (idx >= 0) {
        if (manualLegs[idx].side === leg.side) {
          manualLegs.splice(idx, 1);
        } else {
          manualLegs[idx] = leg;
        }
      } else {
        manualLegs.push(leg);
      }
      renderManualSlip();
      syncOddsButtons();
    }

    function removeLeg(eventId, side) {
      manualLegs = manualLegs.filter((l) => !(l.eventId === eventId && l.side === side));
      renderManualSlip();
      syncOddsButtons();
    }

    function renderManualSlip() {
      if (!manualLegs.length) {
        manualSlipEl.innerHTML = '<div class="empty">No legs selected yet.</div>';
      } else {
        manualSlipEl.innerHTML = manualLegs.map((leg, idx) => `
          <article class="slip-leg">
            <div class="slip-leg-top">
              <span>Leg ${idx + 1} - ${esc(leg.sport)} / ${esc(leg.league)}</span>
              <button class="mini-btn remove-leg" data-remove-key="${esc(`${leg.eventId}:${leg.side}`)}">Remove</button>
            </div>
            <div class="slip-leg-main">${esc(leg.away)} @ ${esc(leg.home)}</div>
            <div class="market" style="margin-top:4px;">${esc(leg.team)} ML (${esc(leg.odds)})</div>
          </article>
        `).join('');
      }

      const stake = Math.max(1, Number(stakeInput.value) || 10);
      const decimals = manualLegs.map((l) => parseAmericanOdds(l.odds)).filter((d) => d && d > 1);

      tLegs.textContent = manualLegs.length;

      if (!decimals.length || decimals.length !== manualLegs.length) {
        tParlayOdds.textContent = 'N/A';
        tToWin.textContent = '$0.00';
      } else {
        const combo = decimals.reduce((a, b) => a * b, 1);
        const payout = stake * combo;
        const toWin = Math.max(0, payout - stake);
        tParlayOdds.textContent = decimalToAmerican(combo);
        tToWin.textContent = `$${toWin.toFixed(2)}`;
      }

      document.querySelectorAll('.remove-leg').forEach((el) => {
        if (el.dataset.bound === '1') return;
        el.dataset.bound = '1';
        el.addEventListener('click', () => {
          const [eventId, side] = String(el.dataset.removeKey || '').split(':');
          removeLeg(eventId, side);
        });
      });
    }

    function renderSportTabs(groups) {
      const sports = Object.keys(groups || {}).filter((s) => (groups[s] || []).length);
      const items = ['All', ...sports];
      if (!items.includes(selectedSport)) {
        selectedSport = 'All';
      }
      tabsEl.innerHTML = items.map((sport) => {
        const count = sport === 'All'
          ? sports.reduce((sum, s) => sum + (groups[s] || []).length, 0)
          : (groups[sport] || []).length;
        const active = selectedSport === sport ? 'active' : '';
        return `<button class="sport-tab ${active}" data-sport="${esc(sport)}">${esc(sport)} (${count})</button>`;
      }).join('');

      document.querySelectorAll('.sport-tab').forEach((el) => {
        el.addEventListener('click', () => {
          selectedSport = el.dataset.sport;
          renderBoardFromCache();
        });
      });
    }

    function renderSuggestedPicks(items) {
      const filtered = selectedSport === 'All'
        ? items
        : items.filter((p) => p.game.sport === selectedSport);

      if (!filtered.length) {
        return '<div class="empty">No suggested picks for this sport.</div>';
      }

      return `<div class="picks">` + filtered.map((pick, idx) => {
        const side = pick.market.side;
        const team = side === 'home' ? pick.game.home : pick.game.away;
        const odds = pick.market.odds || 'Not out yet';
        const alreadyAdded = isSelected(pick.game.event_id, side);
        return `
          <article class="pick">
            <div>
              <span class="leg">Leg ${idx + 1}</span>
              <div class="match">${esc(pick.game.away)} @ ${esc(pick.game.home)} <span style="color:var(--muted);font-size:12px;">${esc(pick.game.sport)}</span></div>
              <div class="market">${esc(pick.market.title)}</div>
              <div style="margin-top:7px;">
                <button
                  class="mini-btn projected-add-btn"
                  data-event-id="${esc(pick.game.event_id)}"
                  data-side="${esc(side)}"
                  data-team="${esc(team)}"
                  data-odds="${esc(odds)}"
                  data-home="${esc(pick.game.home)}"
                  data-away="${esc(pick.game.away)}"
                  data-sport="${esc(pick.game.sport)}"
                  data-league="${esc(pick.game.league)}"
                >${alreadyAdded ? 'Added' : 'Add To Slip'}</button>
              </div>
            </div>
            <div class="edge-box">
              <div class="edge">+${pick.edge.toFixed(3)}</div>
              <div class="prob">M ${pick.model_prob.toFixed(3)} | Mk ${pick.market_prob.toFixed(3)}</div>
            </div>
          </article>
        `;
      }).join('') + `</div>`;
    }

    function getProjectedPicksFiltered() {
      const picks = latestData?.projected_picks || latestData?.parlay || [];
      if (selectedSport === 'All') return picks;
      return picks.filter((p) => p.game.sport === selectedSport);
    }

    function addProjectedPickToSlip(pick) {
      const side = pick.market.side;
      const team = side === 'home' ? pick.game.home : pick.game.away;
      addOrReplaceLeg({
        eventId: pick.game.event_id,
        side,
        team,
        odds: pick.market.odds || '',
        home: pick.game.home,
        away: pick.game.away,
        sport: pick.game.sport,
        league: pick.game.league,
      });
    }

    function addAllProjectedToSlip() {
      const picks = getProjectedPicksFiltered();
      if (!picks.length) return;

      picks.forEach((pick) => addProjectedPickToSlip(pick));
      statusEl.classList.remove('error');
      statusEl.textContent = `Added ${picks.length} projected picks to slip.`;
      renderBoardFromCache();
    }

    function renderGamesBySport(groups) {
      const sports = selectedSport === 'All'
        ? Object.keys(groups || {}).filter((s) => (groups[s] || []).length)
        : [selectedSport].filter((s) => (groups[s] || []).length);

      if (!sports.length) {
        return '<div class="empty">No games for this sport currently.</div>';
      }

      const sections = sports.map((sport) => {
        const cards = (groups[sport] || []).map((g) => {
          const awayPct = oddsToImpliedPct(g.away_odds);
          const homePct = oddsToImpliedPct(g.home_odds);
          const awayDisabled = !g.away_odds ? 'disabled' : '';
          const homeDisabled = !g.home_odds ? 'disabled' : '';
          return `
            <article class="game">
              <div class="game-top">
                <div class="league">${esc(g.league)}</div>
                <div class="time">${fmtTime(g.start)}</div>
              </div>
              <div class="match">${esc(g.away)} @ ${esc(g.home)}</div>

              <div class="moneyline">
                <div class="moneyline-title">Moneyline</div>
                <div class="odds-row">
                  <button
                    class="odds-btn"
                    ${awayDisabled}
                    data-event-id="${esc(g.event_id)}"
                    data-side="away"
                    data-team="${esc(g.away)}"
                    data-odds="${esc(g.away_odds || '')}"
                    data-home="${esc(g.home)}"
                    data-away="${esc(g.away)}"
                    data-sport="${esc(g.sport)}"
                    data-league="${esc(g.league)}"
                  >
                    <span class="team">${esc(g.away)}</span>
                    <span class="line">${esc(displayOdds(g.away_odds))}</span>
                    <span class="imp">${awayPct ? awayPct.toFixed(1) + '% implied' : 'Not out yet'}</span>
                  </button>

                  <button
                    class="odds-btn"
                    ${homeDisabled}
                    data-event-id="${esc(g.event_id)}"
                    data-side="home"
                    data-team="${esc(g.home)}"
                    data-odds="${esc(g.home_odds || '')}"
                    data-home="${esc(g.home)}"
                    data-away="${esc(g.away)}"
                    data-sport="${esc(g.sport)}"
                    data-league="${esc(g.league)}"
                  >
                    <span class="team">${esc(g.home)}</span>
                    <span class="line">${esc(displayOdds(g.home_odds))}</span>
                    <span class="imp">${homePct ? homePct.toFixed(1) + '% implied' : 'Not out yet'}</span>
                  </button>
                </div>
              </div>
            </article>
          `;
        }).join('');

        return `
          <section class="card section" style="margin:0;padding:10px;">
            <div class="section-title" style="margin-bottom:8px;">${esc(sport)}</div>
            <div class="games">${cards}</div>
          </section>
        `;
      }).join('');

      return `<div class="sports-columns">${sections}</div>`;
    }

    function bindOddsHandlers() {
      document.querySelectorAll('.odds-btn').forEach((el) => {
        if (el.dataset.bound === '1') return;
        el.dataset.bound = '1';
        el.addEventListener('click', () => {
          if (el.disabled) return;
          addOrReplaceLeg({
            eventId: el.dataset.eventId,
            side: el.dataset.side,
            team: el.dataset.team,
            odds: el.dataset.odds,
            home: el.dataset.home,
            away: el.dataset.away,
            sport: el.dataset.sport,
            league: el.dataset.league,
          });
        });
      });

      document.querySelectorAll('.projected-add-btn').forEach((el) => {
        if (el.dataset.bound === '1') return;
        el.dataset.bound = '1';
        el.addEventListener('click', () => {
          addOrReplaceLeg({
            eventId: el.dataset.eventId,
            side: el.dataset.side,
            team: el.dataset.team,
            odds: el.dataset.odds,
            home: el.dataset.home,
            away: el.dataset.away,
            sport: el.dataset.sport,
            league: el.dataset.league,
          });
          renderBoardFromCache();
        });
      });
    }

    function renderBoardFromCache() {
      if (!latestData) return;
      renderSportTabs(latestData.games_by_sport || {});
      picksEl.innerHTML = renderSuggestedPicks(latestData.projected_picks || latestData.parlay || []);
      gamesEl.innerHTML = renderGamesBySport(latestData.games_by_sport || {});
      bindOddsHandlers();
      syncOddsButtons();
      renderManualSlip();
    }

    async function run() {
      btn.disabled = true;
      statusEl.classList.remove('error');
      statusEl.textContent = 'Refreshing board...';

      try {
        const res = await fetch('/api/parlay');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Request failed');

        latestData = data;

        mGames.textContent = data.games_count;
        mSports.textContent = data.sports_count;
        mMarkets.textContent = data.markets_count;
        mPicks.textContent = (data.projected_picks || data.parlay || []).length;
        sourceBadge.textContent = `Source: ${data.market_source}`;
        statusEl.textContent = `Updated: ${new Date().toLocaleTimeString()}`;

        renderBoardFromCache();
      } catch (err) {
        statusEl.classList.add('error');
        statusEl.textContent = 'Unable to fetch board.';
        picksEl.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
        gamesEl.innerHTML = '<div class="empty">No games available.</div>';
      } finally {
        btn.disabled = false;
      }
    }

    clearSlipBtn.addEventListener('click', () => {
      manualLegs = [];
      renderManualSlip();
      syncOddsButtons();
      renderBoardFromCache();
    });

    stakeInput.addEventListener('input', renderManualSlip);
    addAllProjectedBtn.addEventListener('click', addAllProjectedToSlip);
    btn.addEventListener('click', run);

    renderAuth();
    run();
  </script>
</body>
</html>
"""


# =========================
# UTIL
# =========================


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def clamp(prob):
    return max(0.01, min(0.99, prob))


def normalize(text):
    return str(text or "").strip().lower()


def get_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def american_to_prob(odds_text):
    if not odds_text:
        return None

    text = str(odds_text).replace("-", "-").replace("+", "").strip()
    try:
        val = int(text)
    except ValueError:
        return None

    if val > 0:
        return clamp(100.0 / (val + 100.0))
    if val < 0:
        n = abs(val)
        return clamp(n / (n + 100.0))
    return None


def team_strength(name):
    return (sum(ord(ch) for ch in normalize(name)) % 200) - 100


def model_probability_for_side(home, away, side):
    diff = team_strength(home) - team_strength(away)
    home_prob = clamp(0.5 + (0.002 * diff) + 0.03)
    if side == "home":
        return home_prob
    return clamp(1.0 - home_prob)


# =========================
# DATA FETCH
# =========================


def fetch_board_data():
    today = utcnow().date()
    games = []
    markets = []
    games_by_sport = {}
    seen = set()

    with get_session() as session:
        for sport_bucket, league_label, sport, league in LEAGUES:
            for i in range(DAYS_AHEAD):
                date = today + dt.timedelta(days=i)
                url = f"{ESPN_BASE}/{sport}/{league}/scoreboard"

                try:
                    res = session.get(url, params={"dates": date.strftime("%Y%m%d")}, timeout=TIMEOUT)
                    res.raise_for_status()
                    data = res.json()
                except (requests.exceptions.RequestException, ValueError):
                    continue

                for event in data.get("events", []):
                    event_id = event.get("id")
                    if not event_id or event_id in seen:
                        continue
                    seen.add(event_id)

                    competitions = event.get("competitions") or []
                    if not competitions:
                        continue

                    comp = competitions[0]
                    competitors = comp.get("competitors") or []

                    try:
                        home_c = next(c for c in competitors if c.get("homeAway") == "home")
                        away_c = next(c for c in competitors if c.get("homeAway") == "away")
                        home = home_c["team"]["displayName"]
                        away = away_c["team"]["displayName"]
                    except (StopIteration, KeyError, TypeError):
                        continue

                    odds_entry = (comp.get("odds") or [{}])[0]
                    moneyline = odds_entry.get("moneyline") or {}
                    home_close = ((moneyline.get("home") or {}).get("close") or {})
                    away_close = ((moneyline.get("away") or {}).get("close") or {})
                    home_open = ((moneyline.get("home") or {}).get("open") or {})
                    away_open = ((moneyline.get("away") or {}).get("open") or {})

                    home_odds = home_close.get("odds") or home_open.get("odds")
                    away_odds = away_close.get("odds") or away_open.get("odds")

                    game = {
                        "event_id": event_id,
                        "sport": sport_bucket,
                        "league": league_label,
                        "home": home,
                        "away": away,
                        "start": comp.get("date"),
                        "home_odds": home_odds,
                        "away_odds": away_odds,
                    }
                    games.append(game)
                    games_by_sport.setdefault(sport_bucket, []).append(game)

                    home_prob = american_to_prob(home_odds)
                    away_prob = american_to_prob(away_odds)

                    if home_prob is not None:
                        markets.append(
                            {
                                "event_id": event_id,
                                "source": MARKET_SOURCE,
                                "title": f"{away} @ {home} - {home} ML",
                                "side": "home",
                                "odds": home_odds,
                                "implied_prob": home_prob,
                            }
                        )

                    if away_prob is not None:
                        markets.append(
                            {
                                "event_id": event_id,
                                "source": MARKET_SOURCE,
                                "title": f"{away} @ {home} - {away} ML",
                                "side": "away",
                                "odds": away_odds,
                                "implied_prob": away_prob,
                            }
                        )

    for sport in games_by_sport:
        games_by_sport[sport].sort(key=lambda g: g.get("start") or "")

    return games, markets, games_by_sport


# =========================
# BUILD PARLAY
# =========================


def build_parlay():
    games, markets, games_by_sport = fetch_board_data()
    game_index = {g["event_id"]: g for g in games}

    opportunities = []
    for market in markets:
        game = game_index.get(market["event_id"])
        if not game:
            continue

        market_prob = market.get("implied_prob")
        if market_prob is None:
            continue

        model_prob = model_probability_for_side(game["home"], game["away"], market["side"])
        edge = model_prob - market_prob

        if edge > MIN_EDGE:
            opportunities.append(
                {
                    "game": game,
                    "market": market,
                    "market_prob": market_prob,
                    "model_prob": model_prob,
                    "edge": edge,
                }
            )

    opportunities.sort(key=lambda x: x["edge"], reverse=True)

    active_sports = {k: v for k, v in games_by_sport.items() if v}
    return {
        "parlay": opportunities[:MAX_LEGS],
        "projected_picks": opportunities,
        "games_count": len(games),
        "markets_count": len(markets),
        "sports_count": len(active_sports),
        "games_by_sport": active_sports,
        "market_source": MARKET_SOURCE,
    }


# =========================
# WEB APP
# =========================

app = Flask(__name__)


@app.get("/")
def home():
    return render_template_string(PAGE, google_client_id=GOOGLE_CLIENT_ID)


@app.get("/api/parlay")
def api_parlay():
    try:
        return jsonify(build_parlay())
    except Exception as err:
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

