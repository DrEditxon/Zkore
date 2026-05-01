/* ══════════════════════════════════════════════
   ZKORE DASHBOARD — app.js
   ══════════════════════════════════════════════ */

let leaguesData       = [];
let matchesData       = [];
let historyData       = null;
let currentLeague     = null;
let currentController = null;
let trainingTimer     = null;

// ── Boot ──────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    setTopbarDate();
    loadLeagues();
});

function setTopbarDate() {
    const el = document.getElementById("topbarDate");
    if (el) el.textContent = new Date().toLocaleDateString("es-ES", { weekday:"short", day:"numeric", month:"short", year:"numeric" });
}

// ── Sidebar ───────────────────────────────────
function toggleSidebar() {
    const sb = document.getElementById("sidebar");
    const mc = document.getElementById("mainContent");
    if (window.innerWidth <= 768) {
        sb.classList.toggle("open");
    } else {
        sb.classList.toggle("collapsed");
        mc.classList.toggle("expanded");
    }
}

// ── View switching ────────────────────────────
function switchView(view, el) {
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    const vEl = document.getElementById(`view-${view}`);
    if (vEl) vEl.classList.add("active");
    if (el) el.classList.add("active");

    // Lazy-load data when switching to views
    if (view === "partidos") renderPartidosView();
    if (view === "predicciones") renderAllPredictionsView();
    if (view === "historial") renderHistorialView();
    if (view === "valuebets") renderValueBetsView();
    if (view === "estadisticas") renderEstadisticasView();
    return false;
}

// ══════════════════════════════════════════════
// LEAGUES
// ══════════════════════════════════════════════
async function loadLeagues() {
    const sel = document.getElementById("leagueSelect");
    try {
        const r = await fetch("/leagues");
        if (!r.ok) throw new Error(`${r.status}`);
        const data = await r.json();
        leaguesData = Array.isArray(data) ? data : Object.values(data);

        sel.innerHTML = '<option value="" disabled>Elige una liga</option>';
        leaguesData.forEach(l => {
            const o = document.createElement("option");
            o.value = l.code; o.textContent = l.name;
            sel.appendChild(o);
        });

        if (leaguesData.length > 0) {
            sel.value = leaguesData[0].code;
            onLeagueChange();
        }
    } catch(e) {
        sel.innerHTML = '<option value="">Error al cargar ligas</option>';
        showStatus("error");
    }
}

async function onLeagueChange() {
    if (trainingTimer) { clearTimeout(trainingTimer); trainingTimer = null; }
    currentLeague = document.getElementById("leagueSelect").value;
    if (!currentLeague) return;

    // Reset all panels to loading state
    resetDashboard();
    matchesData = [];
    historyData = null;

    // Load in parallel
    loadMatchesAndPredictions();
    loadHistory();
}

// ══════════════════════════════════════════════
// MATCHES + PREDICTIONS (Dashboard + Partidos)
// ══════════════════════════════════════════════
function resetDashboard() {
    document.getElementById("predictionsBody").innerHTML =
        `<tr><td colspan="6" class="table-loading"><div class="table-spinner"></div> Cargando predicciones...</td></tr>`;
    document.getElementById("highlightsList").innerHTML =
        `<div class="skeleton-highlight"></div><div class="skeleton-highlight"></div><div class="skeleton-highlight"></div>`;
    const mg = document.getElementById("matchGridFull"); if (mg) mg.innerHTML = "";
    const allPredBody = document.getElementById("allPredictionsBody"); 
    if (allPredBody) allPredBody.innerHTML = `<tr><td colspan="8" class="table-loading"><div class="table-spinner"></div> Cargando jornada...</td></tr>`;
    document.getElementById("trainingBannerDash").style.display = "none";
    document.getElementById("matchdayTag").textContent = "Jornada —";
    document.getElementById("kpiMatches").textContent = "—";
    document.getElementById("kpiConfidence").textContent = "—";
}

async function loadMatchesAndPredictions() {
    if (currentController) currentController.abort();
    currentController = new AbortController();

    try {
        const r = await fetch(`/upcoming/${currentLeague}`, { signal: currentController.signal });

        if (r.status === 202) {
            const d = await r.json().catch(() => ({}));
            showTrainingState(d?.detail?.message || "Entrenando modelo IA...");
            showStatus("warning");
            scheduleRetry();
            return;
        }
        if (!r.ok) throw new Error(`${r.status}`);

        const data = await r.json();
        matchesData = data.matches || [];

        if (data.training_in_progress) {
            showTrainingState(data.training_message || "Entrenando...");
            showStatus("warning");
            scheduleRetry();
            return;
        }

        showStatus("ok");
        document.getElementById("matchdayTag").textContent = `Jornada ${data.matchday || "—"}`;
        document.getElementById("kpiMatches").textContent = matchesData.length;

        renderPredictionsTable(matchesData);
        renderHighlights(matchesData);
        renderStatCards(matchesData);

        // If partidos/predicciones view is active, also update it
        if (document.getElementById("view-partidos").classList.contains("active")) {
            renderPartidosView();
        }
        if (document.getElementById("view-predicciones").classList.contains("active")) {
            renderAllPredictionsView();
        }

    } catch(e) {
        if (e.name === "AbortError") return;
        document.getElementById("predictionsBody").innerHTML =
            `<tr><td colspan="6" class="table-loading" style="color:var(--danger)">⚠️ ${e.message}</td></tr>`;
        showStatus("error");
    }
}

function showTrainingState(msg) {
    document.getElementById("predictionsBody").innerHTML =
        `<tr><td colspan="6" class="table-loading" style="color:var(--blue)">
            <div class="training-dot-anim"></div> ${msg}
        </td></tr>`;
    const banner = document.getElementById("trainingBannerDash");
    document.getElementById("trainingMsg").textContent = msg;
    banner.style.display = "block";
}

function scheduleRetry() {
    if (trainingTimer) clearTimeout(trainingTimer);
    trainingTimer = setTimeout(() => {
        if (document.getElementById("leagueSelect").value === currentLeague) {
            loadMatchesAndPredictions();
        }
    }, 30000);
}

function showStatus(type) {
    const dot = document.querySelector(".status-dot");
    const txt = document.querySelector(".topbar-model-status span:last-child");
    dot.className = `status-dot status-${type}`;
    txt.textContent = type === "ok" ? "Modelo activo" : type === "warning" ? "Entrenando..." : "Error";
}

// ── Predictions Table ─────────────────────────
function renderPredictionsTable(matches) {
    const tbody = document.getElementById("predictionsBody");
    if (!matches.length) {
        tbody.innerHTML = `<tr><td colspan="6" class="table-loading">No hay partidos próximos</td></tr>`;
        return;
    }

    // Top 5 mejores predicciones: orden descendente por probabilidad máxima
    const top5 = [...matches]
        .filter(m => m.prediction && !m.training)
        .sort((a, b) =>
            Math.max(b.prediction.local, b.prediction.empate, b.prediction.visitante) -
            Math.max(a.prediction.local, a.prediction.empate, a.prediction.visitante)
        )
        .slice(0, 5);

    const displayMatches = top5.length ? top5 : matches.slice(0, 5);

    tbody.innerHTML = displayMatches.map((m, idx) => {
        const pred = m.prediction || { local: 33.3, empate: 33.3, visitante: 33.3 };
        const isTraining = m.training;
        const maxProb = Math.max(pred.local, pred.empate, pred.visitante);
        let predTeam, predType, conf;

        if (pred.local >= pred.visitante && pred.local >= pred.empate) {
            predTeam = m.homeTeam.name; predType = "Gana Local";
        } else if (pred.visitante >= pred.local && pred.visitante >= pred.empate) {
            predTeam = m.awayTeam.name; predType = "Gana Visitante";
        } else {
            predTeam = "Empate"; predType = "X";
        }

        if (maxProb >= 60) conf = "Alta";
        else if (maxProb >= 50) conf = "Media";
        else conf = "Baja";

        const xgLocal = pred._xg_local || "—";
        const xgVisit = pred._xg_visitante || "—";
        const date = new Date(m.utcDate).toLocaleTimeString("es-ES", { hour:"2-digit", minute:"2-digit" });

        // crests helpers
        const hCrestUrl = m.homeTeam.crest || "";
        const aCrestUrl = m.awayTeam.crest || "";
        const hImg = hCrestUrl ? `<img src="${hCrestUrl}" style="width:22px;height:22px;object-fit:contain;vertical-align:middle;margin-right:5px" onerror="this.style.display='none'" alt="">` : "";
        const aImg = aCrestUrl ? `<img src="${aCrestUrl}" style="width:22px;height:22px;object-fit:contain;vertical-align:middle;margin-right:5px" onerror="this.style.display='none'" alt="">` : "";

        // winner crest
        let winnerCrestUrl = "";
        if (!isTraining) {
            if (pred.local >= pred.visitante && pred.local >= pred.empate)  winnerCrestUrl = hCrestUrl;
            else if (pred.visitante > pred.local && pred.visitante > pred.empate) winnerCrestUrl = aCrestUrl;
        }
        const winnerImg = winnerCrestUrl
            ? `<img src="${winnerCrestUrl}" style="width:26px;height:26px;object-fit:contain;display:block;margin:0 auto 3px" onerror="this.style.display='none'" alt="">`
            : "";

        return `<tr onclick="openPredModal(${m.id}, ${m.homeTeam.id}, ${m.awayTeam.id}, '${esc(m.homeTeam.name)}', '${esc(m.awayTeam.name)}', '${esc(m.homeTeam.crest||"")}', '${esc(m.awayTeam.crest||"")}', '${m.utcDate}')">
            <td><span style="font-family:var(--mono);font-size:.78rem;font-weight:700;color:var(--accent);background:var(--accent-dim);border-radius:99px;padding:2px 9px">#${idx+1}</span></td>
            <td>
                <div class="match-cell">
                    <span class="match-home" style="display:flex;align-items:center">${hImg}${m.homeTeam.name}</span>
                    <span class="match-away" style="display:flex;align-items:center">${aImg}vs ${m.awayTeam.name}</span>
                    <span class="match-time">${date}</span>
                </div>
            </td>
            <td>
                <div class="pred-cell" style="align-items:center">
                    ${winnerImg}
                    <span class="pred-team">${isTraining ? "Calculando..." : predTeam}</span>
                    <span class="pred-type">${isTraining ? "" : predType}</span>
                </div>
            </td>
            <td>
                <div class="prob-cell">
                    <span class="prob-val">${maxProb.toFixed(1)}%</span>
                    <div class="prob-bar"><div class="prob-bar-fill" style="width:${maxProb}%"></div></div>
                </div>
            </td>
            <td class="xg-cell">
                <div class="xg-pair">
                    <span>${xgLocal}</span>
                    <span class="xg-sep">–</span>
                    <span>${xgVisit}</span>
                </div>
            </td>
            <td><span class="conf-badge conf-${conf.toLowerCase()}">${conf}</span></td>
            <td><button class="detail-btn" onclick="event.stopPropagation();openPredModal(${m.id}, ${m.homeTeam.id}, ${m.awayTeam.id}, '${esc(m.homeTeam.name)}', '${esc(m.awayTeam.name)}', '${esc(m.homeTeam.crest||"")}', '${esc(m.awayTeam.crest||"")}', '${m.utcDate}')">Ver →</button></td>
        </tr>`;

    }).join("");
}

// ── Highlights Panel ──────────────────────────
function renderHighlights(matches) {
    const el = document.getElementById("highlightsList");
    const top3 = [...matches]
        .filter(m => m.prediction)
        .sort((a,b) => Math.max(b.prediction.local, b.prediction.empate, b.prediction.visitante) -
                       Math.max(a.prediction.local, a.prediction.empate, a.prediction.visitante))
        .slice(0, 3);

    if (!top3.length) { el.innerHTML = `<div class="highlight-card" style="color:var(--text-muted);padding:20px;text-align:center">Sin datos disponibles</div>`; return; }

    el.innerHTML = top3.map(m => {
        const pred = m.prediction;
        const maxP = Math.max(pred.local, pred.empate, pred.visitante);
        let predLabel = pred.local >= pred.visitante && pred.local >= pred.empate
            ? `${m.homeTeam.name} Gana`
            : pred.visitante > pred.local && pred.visitante > pred.empate
            ? `${m.awayTeam.name} Gana`
            : "Empate";

        // winner crest for highlights
        const hlWinnerCrest = pred.local >= pred.visitante && pred.local >= pred.empate
            ? m.homeTeam.crest
            : pred.visitante > pred.local && pred.visitante > pred.empate
            ? m.awayTeam.crest
            : null;
        const hlWinImg = hlWinnerCrest
            ? `<img src="${hlWinnerCrest}" style="width:28px;height:28px;object-fit:contain;border-radius:50%;flex-shrink:0" onerror="this.style.display='none'" alt="">`
            : `<span style="font-size:1.2rem">⚽</span>`;

        return `<div class="highlight-card" onclick="openPredModal(${m.id}, ${m.homeTeam.id}, ${m.awayTeam.id}, '${esc(m.homeTeam.name)}', '${esc(m.awayTeam.name)}', '${esc(m.homeTeam.crest||"")}', '${esc(m.awayTeam.crest||"")}', '${m.utcDate}')">
            <div class="hl-teams">
                ${hlWinImg}
                <div style="display:flex;flex-direction:column;gap:2px;min-width:0">
                    <span class="hl-home">${m.homeTeam.name}</span>
                    <span class="hl-away" style="font-size:.7rem">vs ${m.awayTeam.name}</span>
                </div>
            </div>
            <div class="hl-pred">${predLabel}</div>
            <div class="hl-stats">
                <div class="hl-prob-bar"><div class="hl-prob-fill" style="width:${maxP}%"></div></div>
                <span class="hl-prob-val">${maxP.toFixed(1)}%</span>
            </div>
        </div>`;
    }).join("");
}

// ── Stat Cards ────────────────────────────────
function renderStatCards(matches) {
    const withPred = matches.filter(m => m.prediction && !m.training);
    const avgConf = withPred.length
        ? withPred.reduce((s,m) => s + Math.max(m.prediction.local, m.prediction.empate, m.prediction.visitante), 0) / withPred.length
        : 0;

    const highConf = withPred.filter(m => Math.max(m.prediction.local, m.prediction.empate, m.prediction.visitante) >= 60).length;
    const localFav = withPred.filter(m => m.prediction.local > m.prediction.visitante && m.prediction.local > m.prediction.empate).length;
    const avgXgTotal = withPred.length ? withPred.reduce((s,m) => {
        return s + ((m.prediction._xg_local||1.2) + (m.prediction._xg_visitante||1.0));
    }, 0) / withPred.length : 2.2;

    document.getElementById("kpiConfidence").textContent = avgConf ? `${avgConf.toFixed(0)}%` : "—";

    document.getElementById("statCards").innerHTML = `
        <div class="stat-card accent-green">
            <div class="sc-icon green">⚽</div>
            <div class="sc-value">${matches.length}</div>
            <div class="sc-label">Partidos analizados</div>
            <div class="sc-delta neutral">Liga activa</div>
        </div>
        <div class="stat-card accent-blue">
            <div class="sc-icon blue">🎯</div>
            <div class="sc-value">${avgConf.toFixed(0)}%</div>
            <div class="sc-label">Confianza promedio</div>
            <div class="sc-delta ${avgConf>=55?'up':'neutral'}">${avgConf>=55?"Alta":"Media"}</div>
        </div>
        <div class="stat-card accent-warn">
            <div class="sc-icon warn">⭐</div>
            <div class="sc-value">${highConf}</div>
            <div class="sc-label">Predicciones Alta Conf.</div>
            <div class="sc-delta up">≥ 60% prob.</div>
        </div>
        <div class="stat-card accent-red">
            <div class="sc-icon green">🏠</div>
            <div class="sc-value">${localFav}</div>
            <div class="sc-label">Favoritos locales</div>
            <div class="sc-delta neutral">De ${withPred.length} partidos</div>
        </div>`;
}

// ══════════════════════════════════════════════
// HISTORY
// ══════════════════════════════════════════════
async function loadHistory() {
    if (!currentLeague) return;
    try {
        const r = await fetch(`/history/${currentLeague}`);
        if (!r.ok) return;
        historyData = await r.json();
        
        if (historyData.training_in_progress) {
            scheduleRetry();
        }
        
        renderDonut(historyData.summary);
        renderHistoryBars(historyData.history || []);
        renderHistoryKpis(historyData.summary);
    } catch(e) {
        console.warn("History load error:", e);
    }
}

function renderHistoryKpis(s) {
    document.getElementById("hstatAccuracy").textContent = `${s.accuracy}%`;
    document.getElementById("hstatHits").textContent    = s.hits;
    document.getElementById("hstatMisses").textContent  = s.misses;
    document.getElementById("hstatTotal").textContent   = s.total;
}

function renderDonut(s) {
    const circ = 289;
    const hitDash  = ((s.hits  / (s.total||1)) * circ).toFixed(1);
    const missDash = ((s.misses / (s.total||1)) * circ).toFixed(1);
    const donutHits   = document.getElementById("donutHits");
    const donutMisses = document.getElementById("donutMisses");
    const donutVal    = document.getElementById("donutCenterVal");

    requestAnimationFrame(() => {
        donutHits.setAttribute("stroke-dasharray",  `${hitDash} ${circ}`);
        donutMisses.setAttribute("stroke-dasharray", `${missDash} ${circ}`);
        // Offset misses to start after hits
        donutMisses.setAttribute("stroke-dashoffset", `-${hitDash}`);
    });
    if (donutVal) donutVal.textContent = `${s.accuracy}%`;
}

function renderHistoryBars(history) {
    const el = document.getElementById("historyBars");
    if (!history.length) { el.innerHTML = `<div style="color:var(--text-muted);font-size:.8rem;margin:auto">Sin historial</div>`; return; }

    const last8 = history.slice(0, 8).reverse();
    const maxH = 100;
    el.innerHTML = last8.map(h => {
        const isHit = h.is_hit;
        const dateStr = new Date(h.date).toLocaleDateString("es-ES", { day:"numeric", month:"numeric" });
        return `<div class="hbar-wrap">
            <div class="hbar-bar ${isHit?'hit':'miss'}" style="height:${isHit?70:35}px"></div>
            <div class="hbar-label">${dateStr}</div>
        </div>`;
    }).join("");
}

// ══════════════════════════════════════════════
// PARTIDOS VIEW
// ══════════════════════════════════════════════
function renderPartidosView() {
    const grid = document.getElementById("matchGridFull");
    if (!matchesData.length) {
        grid.innerHTML = Array(6).fill(0).map(() => `
            <div class="match-card-skeleton">
                <div class="skel-teams"><div class="skel-crest skel-crest"></div><div class="skel-name"></div><div class="skel-vs"></div><div class="skel-name"></div><div class="skel-crest skel-crest"></div></div>
                <div class="skel-bar"></div><div class="skel-date"></div>
            </div>`).join("");
        return;
    }

    grid.innerHTML = matchesData.map(m => {
        const pred = m.prediction || { local:33.3, empate:33.3, visitante:33.3 };
        const date = new Date(m.utcDate).toLocaleString("es-ES", { weekday:"short", day:"numeric", month:"short", hour:"2-digit", minute:"2-digit" });
        const verdict = m.verdict || "?";
        const isTraining = m.training;

        const pL = pred.local, pE = pred.empate, pV = pred.visitante;
        const lWin = pL > pE && pL > pV;
        const vWin = pV > pL && pV > pE;

        const hCrest = m.homeTeam.crest
            ? `<img src="${m.homeTeam.crest}" class="mcd-crest" onerror="this.style.display='none'" alt="">`
            : `<div class="mcd-crest-ph">⚽</div>`;
        const aCrest = m.awayTeam.crest
            ? `<img src="${m.awayTeam.crest}" class="mcd-crest" onerror="this.style.display='none'" alt="">`
            : `<div class="mcd-crest-ph">⚽</div>`;

        return `<div class="match-card-dash" onclick="openPredModal(${m.id}, ${m.homeTeam.id}, ${m.awayTeam.id}, '${esc(m.homeTeam.name)}', '${esc(m.awayTeam.name)}', '${esc(m.homeTeam.crest||"")}', '${esc(m.awayTeam.crest||"")}', '${m.utcDate}')">
            <div class="mcd-teams">
                <div class="mcd-team">${hCrest}<span class="mcd-name">${m.homeTeam.name}</span></div>
                <span class="mcd-vs">VS</span>
                <div class="mcd-team">${aCrest}<span class="mcd-name">${m.awayTeam.name}</span></div>
            </div>
            <div class="mcd-probs">
                <div class="mcd-prob-pill${lWin?' winner':''}">
                    <span class="label">Local</span>
                    <span class="val">${pL.toFixed(1)}%</span>
                </div>
                <div class="mcd-prob-pill${(!lWin&&!vWin)?' winner':''}">
                    <span class="label">Empate</span>
                    <span class="val">${pE.toFixed(1)}%</span>
                </div>
                <div class="mcd-prob-pill${vWin?' winner':''}">
                    <span class="label">Visit.</span>
                    <span class="val">${pV.toFixed(1)}%</span>
                </div>
            </div>
            <div class="mcd-footer">
                <span>${date}</span>
                <span class="mcd-verdict verdict-${isTraining?'Q':verdict}">${isTraining?'Calc.':verdict}</span>
            </div>
        </div>`;
    }).join("");
}

// ══════════════════════════════════════════════
// DATA GRID VIEW (Predicciones)
// ══════════════════════════════════════════════
function renderAllPredictionsView() {
    const tbody = document.getElementById("allPredictionsBody");
    if (!tbody) return;
    if (!matchesData.length) {
        tbody.innerHTML = `<tr><td colspan="8" class="table-loading">No hay partidos disponibles</td></tr>`;
        return;
    }

    tbody.innerHTML = matchesData.map(m => {
        const pred = m.prediction || { local: 33.3, empate: 33.3, visitante: 33.3 };
        const isTraining = m.training;
        
        const dateStr = new Date(m.utcDate).toLocaleString("es-ES", { weekday:"short", day:"numeric", hour:"2-digit", minute:"2-digit" });
        
        const maxProb = Math.max(pred.local, pred.empate, pred.visitante);
        let conf;
        if (maxProb >= 60) conf = "Alta";
        else if (maxProb >= 50) conf = "Media";
        else conf = "Baja";
        
        const confClass = conf === "Alta" ? "conf-alta" : conf === "Media" ? "conf-media" : "conf-baja";
        const xgLocal = pred._xg_local || "—";
        const xgVisit = pred._xg_visitante || "—";
        
        // Colors for 1X2 based on probability
        const getColor = (p) => p >= 50 ? 'var(--accent)' : p >= 35 ? 'var(--blue)' : 'var(--text-secondary)';
        
        const hCrestUrl = m.homeTeam.crest || "";
        const aCrestUrl = m.awayTeam.crest || "";
        const hImg = hCrestUrl ? `<img src="${hCrestUrl}" style="width:20px;height:20px;object-fit:contain;vertical-align:middle;margin-right:8px" onerror="this.style.display='none'" alt="">` : "";
        const aImg = aCrestUrl ? `<img src="${aCrestUrl}" style="width:20px;height:20px;object-fit:contain;vertical-align:middle;margin-right:8px" onerror="this.style.display='none'" alt="">` : "";

        return `<tr onclick="openPredModal(${m.id}, ${m.homeTeam.id}, ${m.awayTeam.id}, '${esc(m.homeTeam.name)}', '${esc(m.awayTeam.name)}', '${esc(m.homeTeam.crest||"")}', '${esc(m.awayTeam.crest||"")}', '${m.utcDate}')">
            <td style="font-size:0.75rem;color:var(--text-muted);font-family:var(--mono)">${dateStr}</td>
            <td>
                <div style="display:flex;flex-direction:column;gap:4px;font-weight:600">
                    <span style="display:flex;align-items:center">${hImg}${m.homeTeam.name}</span>
                    <span style="display:flex;align-items:center;color:var(--text-secondary);font-size:0.8rem;font-weight:400">${aImg}${m.awayTeam.name}</span>
                </div>
            </td>
            <td style="font-family:var(--mono);font-weight:700;color:${isTraining?'inherit':getColor(pred.local)}">${isTraining?'-':pred.local.toFixed(1)+'%'}</td>
            <td style="font-family:var(--mono);font-weight:700;color:${isTraining?'inherit':getColor(pred.empate)}">${isTraining?'-':pred.empate.toFixed(1)+'%'}</td>
            <td style="font-family:var(--mono);font-weight:700;color:${isTraining?'inherit':getColor(pred.visitante)}">${isTraining?'-':pred.visitante.toFixed(1)+'%'}</td>
            <td class="xg-cell">
                <div class="xg-pair">
                    <span>${isTraining?'-':xgLocal}</span>
                    <span class="xg-sep">–</span>
                    <span>${isTraining?'-':xgVisit}</span>
                </div>
            </td>
            <td>${isTraining?'-':`<span class="conf-badge ${confClass}">${conf}</span>`}</td>
            <td><button class="detail-btn" onclick="event.stopPropagation();openPredModal(${m.id}, ${m.homeTeam.id}, ${m.awayTeam.id}, '${esc(m.homeTeam.name)}', '${esc(m.awayTeam.name)}', '${esc(m.homeTeam.crest||"")}', '${esc(m.awayTeam.crest||"")}', '${m.utcDate}')">Ver →</button></td>
        </tr>`;
    }).join("");
}

// ══════════════════════════════════════════════
// HISTORIAL VIEW
// ══════════════════════════════════════════════
function renderHistorialView() {
    const el = document.getElementById("historialContent");
    if (!historyData) {
        el.innerHTML = `<div style="color:var(--text-muted);text-align:center;padding:40px">Cargando historial...</div>`;
        if (currentLeague) loadHistory().then(() => { if (historyData) renderHistorialView(); });
        return;
    }

    if (historyData.training_in_progress) {
        el.innerHTML = `<div style="text-align:center;padding:60px 20px;color:var(--blue)">
            <div class="training-dot-anim" style="margin:0 auto 16px;width:28px;height:28px"></div>
            <p style="font-weight:600">Calculando historial con el modelo en entrenamiento...</p>
        </div>`;
        return;
    }

    const history = historyData.history || [];
    if (!history.length) {
        el.innerHTML = `<div style="color:var(--text-muted);text-align:center;padding:40px">Sin historial disponible</div>`;
        return;
    }

    el.innerHTML = history.map(h => `
        <div class="hist-card">
            <span class="hist-badge ${h.is_hit?'hit':'miss'}">${h.is_hit?'ACIERTO':'FALLO'}</span>
            <div>
                <div class="hist-match">${h.match}</div>
                <div class="hist-date">${new Date(h.date).toLocaleDateString("es-ES",{day:"numeric",month:"short",year:"numeric"})}</div>
            </div>
            <div class="hist-score">${h.actual_score}</div>
            <div class="hist-details">
                <span class="hist-pill ${h.details.winner_hit?'ok':''}">🏆 ${h.details.winner_hit?'✓':'✗'}</span>
                <span class="hist-pill ${h.details.goals_hit?'ok':''}">⚽ ${h.details.goals_hit?'✓':'✗'}</span>
            </div>
        </div>`).join("");
}

// ══════════════════════════════════════════════
// VALUE BETS VIEW
// ══════════════════════════════════════════════
async function renderValueBetsView() {
    const el = document.getElementById("valueBetsContent");
    if (!el) return;
    if (!currentLeague) {
        el.innerHTML = `<div style="color:var(--text-muted);text-align:center;padding:60px 20px">
            <div style="font-size:2rem;margin-bottom:12px">⚽</div>
            <p>Selecciona una liga para ver las Value Bets.</p>
        </div>`;
        return;
    }

    // Loading state
    el.innerHTML = `<div style="text-align:center;padding:60px 20px;color:var(--text-muted)">
        <div class="table-spinner" style="margin:0 auto 16px;width:32px;height:32px;border-width:3px"></div>
        <p>Analizando oportunidades de valor...</p>
    </div>`;

    try {
        const r = await fetch(`/value-bets/${currentLeague}`);
        if (!r.ok) {
            const d = await r.json().catch(() => ({}));
            throw new Error(d?.detail || `Error ${r.status}`);
        }
        const data = await r.json();

        if (data.training_in_progress) {
            el.innerHTML = `<div style="text-align:center;padding:60px 20px;color:var(--blue)">
                <div class="training-dot-anim" style="margin:0 auto 16px;width:28px;height:28px"></div>
                <p style="font-weight:600">Modelo entrenándose</p>
                <p style="color:var(--text-muted);margin-top:8px;font-size:.85rem">${data.training_message || "Intenta en ~30 segundos"}</p>
            </div>`;
            return;
        }

        const bets   = data.value_bets  || [];
        const base   = data.baseline    || {};

        // ── Baseline panel ────────────────────────────────────────────────
        const baseHtml = `
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px 22px;margin-bottom:24px">
            <div style="font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:14px">
                📊 Baseline de Liga · ${base.total_matches?.toLocaleString() || "—"} partidos históricos · Margen simulado ${base.overround || "7%"} · Edge mínimo ${base.min_edge || "7%"}
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
                <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center">
                    <div style="font-size:.78rem;color:var(--text-muted);margin-bottom:4px">🤖 Inteligencia</div>
                    <div style="font-size:1.1rem;font-weight:700;color:var(--text-primary)">XGBoost + ELO</div>
                    <div style="font-size:.7rem;color:var(--blue);margin-top:2px">Contexto Avanzado</div>
                </div>
                <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center">
                    <div style="font-size:.78rem;color:var(--text-muted);margin-bottom:4px">🏦 Bookie Simulado</div>
                    <div style="font-size:1.1rem;font-weight:700;color:var(--text-primary)">Poisson Dinámico</div>
                    <div style="font-size:.7rem;color:var(--text-secondary);margin-top:2px">Con Margen ${base.overround || "7%"}</div>
                </div>
                <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center">
                    <div style="font-size:.78rem;color:var(--text-muted);margin-bottom:4px">⚡ Edge Mínimo</div>
                    <div style="font-size:1.1rem;font-weight:700;color:var(--text-primary)">+${base.min_edge || "5%"} EV</div>
                    <div style="font-size:.7rem;color:var(--accent);margin-top:2px">Ventaja Requerida</div>
                </div>
            </div>
        </div>`;

        // ── No bets found ─────────────────────────────────────────────────
        if (!bets.length) {
            el.innerHTML = baseHtml + `
            <div style="text-align:center;padding:48px 20px;color:var(--text-muted)">
                <div style="font-size:2.5rem;margin-bottom:14px">🔍</div>
                <p style="font-weight:600;color:var(--text-primary);margin-bottom:6px">Sin Value Bets detectadas</p>
                <p style="font-size:.85rem">El modelo no encontró diferencias ≥ ${base.min_edge || "7%"} entre sus probabilidades y el mercado simulado de esta jornada.</p>
            </div>`;
            return;
        }

        // ── Bet cards ─────────────────────────────────────────────────────
        const outcomeColors = {
            "Local Gana":     "var(--accent)",
            "Empate":         "#64748b",
            "Visitante Gana": "var(--blue)",
        };

        const cardsHtml = bets.map((vb, idx) => {
            const date = new Date(vb.utc_date).toLocaleString("es-ES", { weekday:"short", day:"numeric", month:"short", hour:"2-digit", minute:"2-digit" });
            const hCrest = vb.home_crest ? `<img src="${vb.home_crest}" style="width:28px;height:28px;object-fit:contain" onerror="this.style.display='none'" alt="">` : "⚽";
            const aCrest = vb.away_crest ? `<img src="${vb.away_crest}" style="width:28px;height:28px;object-fit:contain" onerror="this.style.display='none'" alt="">` : "⚽";

            const betsRows = vb.bets.map(b => {
                const edgeColor  = b.edge >= 15 ? "var(--success,#22c55e)" : b.edge >= 10 ? "var(--warn,#f59e0b)" : "var(--blue)";
                const evColor    = b.expected_value >= 10 ? "var(--success,#22c55e)" : "var(--warn,#f59e0b)";
                const outColor   = outcomeColors[b.outcome] || "var(--accent)";
                return `
                <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:13px 16px;margin-top:10px">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                        <div>
                            <span style="font-weight:700;color:${outColor};font-size:.9rem">${b.outcome}</span>
                            <span style="font-size:.78rem;color:var(--text-muted);margin-left:8px">${b.team_label}</span>
                        </div>
                        <div style="display:flex;gap:8px">
                            <span title="Diferencia directa de probabilidades (Modelo - Mercado)" style="cursor:help;background:${edgeColor}22;color:${edgeColor};border-radius:99px;padding:3px 10px;font-size:.75rem;font-weight:700">Edge +${b.edge}%</span>
                            <span title="Expected Value: (Prob. Modelo × Cuota) - 1. Mide tu retorno a largo plazo." style="cursor:help;background:${evColor}22;color:${evColor};border-radius:99px;padding:3px 10px;font-size:.75rem;font-weight:700">EV +${b.expected_value}%</span>
                            <span title="Kelly Fraccional (1/4): % recomendado de tu Bankroll total para esta apuesta." style="cursor:help;background:var(--accent-dim);color:var(--accent);border-radius:99px;padding:3px 10px;font-size:.75rem;font-weight:700">💰 Stake ${b.kelly_stake}%</span>
                        </div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:.78rem">
                        <div style="text-align:center">
                            <div style="color:var(--text-muted);margin-bottom:2px">Modelo</div>
                            <div style="font-size:1.05rem;font-weight:700;color:var(--accent)">${b.model_prob}%</div>
                        </div>
                        <div style="text-align:center">
                            <div style="color:var(--text-muted);margin-bottom:2px">Mercado</div>
                            <div style="font-size:1.05rem;font-weight:700;color:var(--text-primary)">${b.market_prob}%</div>
                        </div>
                        <div style="text-align:center">
                            <div style="color:var(--text-muted);margin-bottom:2px">Cuota</div>
                            <div style="font-size:1.05rem;font-weight:700;color:var(--text-primary)">${b.market_odds}</div>
                        </div>
                    </div>
                    <div style="margin-top:10px">
                        <div style="height:4px;border-radius:99px;background:var(--border);overflow:hidden">
                            <div style="height:100%;width:${Math.min(b.model_prob, 100)}%;background:${outColor};border-radius:99px;transition:width .6s ease"></div>
                        </div>
                    </div>
                </div>`;
            }).join("");

            return `
            <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px 22px;margin-bottom:16px;cursor:pointer"
                 onclick="openPredModal(${vb.match_id},${vb.home_id},${vb.away_id},'${esc(vb.home_team)}','${esc(vb.away_team)}','${esc(vb.home_crest||"")}','${esc(vb.away_crest||"")}','${vb.utc_date}')">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
                    <div style="display:flex;align-items:center;gap:10px">
                        <span style="font-family:var(--mono);font-size:.78rem;font-weight:700;color:var(--accent);background:var(--accent-dim);border-radius:99px;padding:2px 9px">#${idx+1}</span>
                        <div style="display:flex;align-items:center;gap:7px">
                            ${hCrest}
                            <span style="font-weight:600;font-size:.95rem">${vb.home_team}</span>
                            <span style="color:var(--text-muted);font-size:.8rem">vs</span>
                            <span style="font-weight:600;font-size:.95rem">${vb.away_team}</span>
                            ${aCrest}
                        </div>
                    </div>
                    <span style="font-size:.75rem;color:var(--text-muted)">${date}</span>
                </div>
                ${betsRows}
            </div>`;
        }).join("");

        el.innerHTML = baseHtml + `
        <div style="margin-bottom:14px;font-size:.85rem;color:var(--text-muted)">
            <strong style="color:var(--text-primary)">${bets.length}</strong> partido${bets.length!==1?"s":""} con oportunidad de valor detectada — ordenados por mayor edge
        </div>` + cardsHtml;

        // Animate bars after render
        setTimeout(() => {
            el.querySelectorAll("div[style*='transition:width']").forEach(bar => {
                const w = bar.style.width; bar.style.width = "0";
                requestAnimationFrame(() => { bar.style.width = w; });
            });
        }, 80);

    } catch(e) {
        el.innerHTML = `<div style="text-align:center;padding:60px 20px;color:var(--danger)">⚠️ ${e.message}</div>`;
    }
}

// ══════════════════════════════════════════════
// ESTADÍSTICAS VIEW
// ══════════════════════════════════════════════
function renderEstadisticasView() {
    const el = document.getElementById("statsContent");
    if (!historyData) {
        el.innerHTML = `<div style="color:var(--text-muted);text-align:center;padding:40px">Cargando estadísticas...</div>`;
        return;
    }
    
    if (historyData.training_in_progress) {
        el.innerHTML = `<div style="text-align:center;padding:60px 20px;color:var(--blue)">
            <div class="training-dot-anim" style="margin:0 auto 16px;width:28px;height:28px"></div>
            <p style="font-weight:600">Analizando estadísticas del modelo...</p>
        </div>`;
        return;
    }
    const s = historyData.summary;
    el.innerHTML = `
        <div class="stat-cards" style="max-width:800px">
            <div class="stat-card accent-green">
                <div class="sc-icon green">🎯</div>
                <div class="sc-value">${s.accuracy}%</div>
                <div class="sc-label">Precisión global</div>
                <div class="sc-delta ${s.accuracy>=60?'up':'neutral'}">Últimos ${s.total} partidos</div>
            </div>
            <div class="stat-card accent-blue">
                <div class="sc-icon blue">✅</div>
                <div class="sc-value">${s.hits}</div>
                <div class="sc-label">Aciertos totales</div>
                <div class="sc-delta up">Resultado correcto</div>
            </div>
            <div class="stat-card accent-red">
                <div class="sc-icon red">❌</div>
                <div class="sc-value">${s.misses}</div>
                <div class="sc-label">Fallos totales</div>
                <div class="sc-delta down">${(100-s.accuracy).toFixed(1)}% error</div>
            </div>
            <div class="stat-card accent-warn">
                <div class="sc-icon warn">📊</div>
                <div class="sc-value">${s.total}</div>
                <div class="sc-label">Total evaluados</div>
                <div class="sc-delta neutral">Liga actual</div>
            </div>
        </div>
        <div style="margin-top:16px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;max-width:800px">
            <p class="modal-section-title">Sobre el Modelo</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px;font-size:.85rem;color:var(--text-secondary)">
                <div><strong style="color:var(--text-primary)">Algoritmo</strong><br>XGBoost Regressor para Expected Goals (xG)</div>
                <div><strong style="color:var(--text-primary)">Probabilidades</strong><br>Distribución Poisson bivariada</div>
                <div><strong style="color:var(--text-primary)">Calibración</strong><br>Isotonic Regression</div>
                <div><strong style="color:var(--text-primary)">Features</strong><br>ELO, forma, H2H, xG proxy, rest days</div>
            </div>
        </div>`;
}

// ══════════════════════════════════════════════
// PREDICTION MODAL
// ══════════════════════════════════════════════
async function openPredModal(matchId, homeId, awayId, homeName, awayName, homeCrest, awayCrest, utcDate) {
    const modal   = document.getElementById("predModal");
    const body    = document.getElementById("predModalBody");

    modal.style.display = "flex";
    document.body.style.overflow = "hidden";

    const hImg = homeCrest ? `<img src="${homeCrest}" onerror="this.style.display='none'" alt="">` : `<div class="modal-crest-ph">⚽</div>`;
    const aImg = awayCrest ? `<img src="${awayCrest}" onerror="this.style.display='none'" alt="">` : `<div class="modal-crest-ph">⚽</div>`;

    body.innerHTML = `
        <div class="modal-match-hdr">
            <div class="modal-team">${hImg}<div class="modal-team-name">${homeName}</div></div>
            <div class="modal-vs-box">
                <div class="modal-vs-lbl">VS</div>
                <div class="modal-xg-badge"><div class="table-spinner" style="display:inline-block;margin-right:6px"></div>Cargando...</div>
            </div>
            <div class="modal-team">${aImg}<div class="modal-team-name">${awayName}</div></div>
        </div>
        <div class="modal-body">
            <div style="text-align:center;padding:20px;color:var(--text-muted)">
                <div class="table-spinner" style="margin:0 auto 12px;width:28px;height:28px;border-width:3px"></div>
                Generando análisis completo...
            </div>
        </div>`;

    try {
        const r = await fetch(`/predict?league_code=${currentLeague}&team_local=${homeId}&team_visitante=${awayId}&match_id=${matchId}&utc_date=${encodeURIComponent(utcDate)}`);

        if (r.status === 202) {
            const d = await r.json().catch(()=>({}));
            body.innerHTML = `<div class="modal-body" style="text-align:center;padding:40px">
                <div class="training-dot-anim" style="margin:0 auto 16px;width:24px;height:24px"></div>
                <p style="color:var(--blue);font-weight:600">Modelo entrenándose</p>
                <p style="color:var(--text-muted);margin-top:8px;font-size:.85rem">${d?.detail?.message||"Intenta en ~30 segundos"}</p>
            </div>`;
            return;
        }

        if (!r.ok) {
            const d = await r.json().catch(()=>({}));
            throw new Error(d?.detail || r.status);
        }

        const data = await r.json();
        renderModalContent(body, data, homeName, awayName, homeCrest, awayCrest, hImg, aImg);

    } catch(e) {
        body.innerHTML = `<div class="modal-body" style="text-align:center;padding:40px;color:var(--danger)">⚠️ ${e.message}</div>`;
    }
}

function renderModalContent(body, data, homeName, awayName, homeCrest, awayCrest, hImg, aImg) {
    const info    = data.modelo_info;
    const probs   = data.probabilidades;
    const markets = data.metricas_mercado;
    const xG      = data.expected_goals;
    const dist    = data.distribucion_goles;
    const scores  = data.marcadores_probables;

    const pL = probs.local, pE = probs.empate, pV = probs.visitante;
    const maxP = Math.max(pL, pE, pV);

    const confClass = info.confianza === "Alta" ? "conf-alta" : info.confianza === "Media" ? "conf-media" : "conf-baja";

    function probRow(label, val, colorH, golden) {
        return `<div class="prob-row-m ${golden?'golden':''}">
            <span class="prob-label-m">${label}${golden?' <span class="fav-star">★</span>':''}</span>
            <div class="prob-bar-m"><div class="prob-fill-m" style="width:${val}%;background:${colorH}"></div></div>
            <span class="prob-val-m">${val.toFixed(1)}%</span>
        </div>`;
    }

    function mktBox(name, val) {
        const cls = val >= 55 ? "high" : val <= 40 ? "low" : "";
        return `<div class="market-box ${cls}">
            <div class="market-name">${name}</div>
            <div class="market-val">${val}%</div>
            <div class="market-bar"><div class="market-bar-f" style="width:${val}%"></div></div>
        </div>`;
    }

    function gdRows(items, cls) {
        const maxP2 = Math.max(...items.map(d=>d.probabilidad));
        return items.map(d => `<div class="gd-row">
            <span class="gd-lbl">${d.goles}</span>
            <div class="gd-bar"><div class="gd-fill ${cls}" style="width:${maxP2>0?(d.probabilidad/maxP2*100):0}%"></div></div>
            <span class="gd-pct">${d.probabilidad}%</span>
        </div>`).join("");
    }

    // Stats section
    let statsHtml = "";
    if (data.estadisticas_esperadas) {
        const s = data.estadisticas_esperadas;
        const nota = data.nota ? `<div style="font-size:.72rem;color:var(--warn);margin-bottom:8px;font-style:italic">${data.nota}</div>` : "";
        statsHtml = `<div>${nota}<div class="advanced-grid">
            <div class="adv-box">
                <div class="adv-title">🟨 Amarillas Esperadas</div>
                <div class="adv-stats">
                    <div class="adv-stat"><div class="adv-val">${s.tarjetas_amarillas.local}</div><div class="adv-lbl">Local</div></div>
                    <div class="adv-stat"><div class="adv-val">${s.tarjetas_amarillas.visitante}</div><div class="adv-lbl">Visitante</div></div>
                </div>
            </div>
            <div class="adv-box">
                <div class="adv-title">🎯 Tiros a Puerta</div>
                <div class="adv-stats">
                    <div class="adv-stat"><div class="adv-val">${s.tiros_arco.local}</div><div class="adv-lbl">Local</div></div>
                    <div class="adv-stat"><div class="adv-val">${s.tiros_arco.visitante}</div><div class="adv-lbl">Visitante</div></div>
                </div>
            </div>
        </div></div>`;
    }

    body.innerHTML = `
        <div class="modal-match-hdr">
            <div class="modal-team">${hImg}<div class="modal-team-name">${homeName}</div></div>
            <div class="modal-vs-box">
                <div class="modal-vs-lbl">VS</div>
                <div class="modal-xg-badge">xG ${xG.local} – ${xG.visitante}</div>
            </div>
            <div class="modal-team">${aImg}<div class="modal-team-name">${awayName}</div></div>
        </div>
        <div class="modal-body">

            <div>
                <div class="modal-model-badge">🤖 ${info.tipo} &nbsp;·&nbsp; <span class="conf-badge ${confClass}">${info.confianza}</span></div>
            </div>

            <div>
                <div class="modal-section-title">📊 Análisis del Modelo</div>
                <div class="modal-explanation">${info.explicacion}</div>
            </div>

            <div>
                <div class="modal-section-title">Probabilidades de Resultado</div>
                <div class="prob-rows">
                    ${probRow("Local", pL, "var(--accent)", pL===maxP)}
                    ${probRow("Empate", pE, "#64748b", pE===maxP)}
                    ${probRow("Visitante", pV, "var(--blue)", pV===maxP)}
                </div>
            </div>

            ${statsHtml ? `<div><div class="modal-section-title">Métricas de Juego</div>${statsHtml}</div>` : ""}

            <div>
                <div class="modal-section-title">Mercados</div>
                <div class="markets-grid">
                    ${mktBox("Ambos Marcan", markets.btts)}
                    ${mktBox("Más de 2.5", markets.over_2_5)}
                    ${mktBox("Menos de 2.5", markets.under_2_5)}
                    ${mktBox(`CS ${homeName.substring(0,4)}.`, markets.clean_sheet_local)}
                    ${mktBox(`CS ${awayName.substring(0,4)}.`, markets.clean_sheet_visitante)}
                    ${mktBox("Prob. Local", markets.prob_home_win)}
                </div>
            </div>

            <div>
                <div class="modal-section-title">Distribución de Goles</div>
                <div class="goal-dist-grid">
                    <div><div class="gd-team-label">${homeName}</div>${gdRows(dist.local,"home")}</div>
                    <div><div class="gd-team-label">${awayName}</div>${gdRows(dist.visitante,"away")}</div>
                </div>
            </div>

            <div>
                <div class="modal-section-title">Marcadores Probables</div>
                <div class="scorelines">
                    ${scores.map((s,i) => `
                    <div class="scoreline-row ${i===0?'top':''}">
                        <span class="score-val">${s.local} – ${s.visitante}</span>
                        <div class="score-bar"><div class="score-fill" style="width:${Math.min(s.probabilidad*5,100)}%"></div></div>
                        <span class="score-pct">${s.probabilidad}%</span>
                    </div>`).join("")}
                </div>
            </div>

        </div>`;

    // Trigger bar animations
    setTimeout(() => {
        body.querySelectorAll(".prob-fill-m,.market-bar-f,.score-fill,.gd-fill,.hl-prob-fill").forEach(el => {
            const w = el.style.width; el.style.width = "0"; requestAnimationFrame(() => { el.style.width = w; });
        });
    }, 50);
}

function closeModal(e) {
    if (e && e.target !== document.getElementById("predModal")) return;
    closePredModal();
}
function closePredModal() {
    document.getElementById("predModal").style.display = "none";
    document.body.style.overflow = "auto";
}

// ── Utils ─────────────────────────────────────
function esc(s) { return String(s).replace(/'/g, "\\'"); }

function showToast(msg) {
    const c = document.getElementById("toastContainer");
    const t = document.createElement("div");
    t.className = "toast"; t.textContent = msg;
    c.appendChild(t);
    setTimeout(() => t.remove(), 4000);
}

// Close sidebar on mobile when clicking outside
document.addEventListener("click", e => {
    if (window.innerWidth <= 768) {
        const sb = document.getElementById("sidebar");
        if (sb.classList.contains("open") && !sb.contains(e.target) && !e.target.closest(".sidebar-toggle")) {
            sb.classList.remove("open");
        }
    }
});
