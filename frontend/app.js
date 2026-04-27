let leaguesData = [];
let currentController = null;
let trainingPollingTimer = null;

document.addEventListener("DOMContentLoaded", () => {
    loadLeagues();
});

// ─────────────────────────────────────────────
// LEAGUES
// ─────────────────────────────────────────────
async function loadLeagues() {
    const leagueSelect = document.getElementById("league");
    if (!leagueSelect) return;

    try {
        const response = await fetch("/leagues");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        leaguesData = Array.isArray(data) ? data : Object.values(data);

        if (leaguesData.length === 0) {
            leagueSelect.innerHTML = '<option value="">No hay ligas disponibles</option>';
            return;
        }

        leagueSelect.innerHTML = '<option value="" disabled selected>Elige liga para ver partidos próximos</option>';
        leaguesData.forEach(league => {
            const option = document.createElement("option");
            option.value = league.code;
            option.textContent = league.name;
            leagueSelect.appendChild(option);
        });

        if (leaguesData.length > 0) {
            leagueSelect.value = leaguesData[0].code;
            onLeagueChange();
        }
    } catch (error) {
        console.error("Error loading leagues:", error);
        leagueSelect.innerHTML = '<option value="">Error al conectar con el servidor</option>';
    }
}

async function onLeagueChange() {
    // Cancel any active training polling when user switches league
    if (trainingPollingTimer) {
        clearTimeout(trainingPollingTimer);
        trainingPollingTimer = null;
    }
    loadHistory();
    loadMatches();
}

// ─────────────────────────────────────────────
// HISTORY
// ─────────────────────────────────────────────
async function loadHistory() {
    try {
        const leagueCode = document.getElementById("league").value;
        const historySection = document.getElementById("zkore-prediction-history");
        const summaryText = document.getElementById("history-summary");
        const accuracyText = document.getElementById("history-accuracy");
        const hitCircle = document.getElementById("history-chart-hit");
        const missCircle = document.getElementById("history-chart-miss");
        const detailsContainer = document.getElementById("history-details");

        if (!historySection || !leagueCode) return;

        historySection.style.display = "block";
        summaryText.innerText = "Analizando últimos resultados...";
        detailsContainer.innerHTML = "";
        detailsContainer.style.display = "none";

        const response = await fetch(`/history/${leagueCode}`);
        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

        const data = await response.json();
        const { summary, history } = data;

        const total = summary.total || 1;
        const hitPercent = (summary.hits / total) * 100;
        const missPercent = (summary.misses / total) * 100;

        if (hitCircle && missCircle) {
            hitCircle.setAttribute("stroke-dasharray", `${hitPercent}, 100`);
            missCircle.setAttribute("stroke-dasharray", `${missPercent}, 100`);
            missCircle.setAttribute("stroke-dashoffset", `-${hitPercent}`);
        }

        if (accuracyText) accuracyText.textContent = `${summary.accuracy}%`;
        if (summaryText) summaryText.innerText = `Hemos acertado ${summary.hits} de los últimos ${summary.total} partidos.`;

        renderHistoryList(history);
    } catch (error) {
        console.error("Error in loadHistory:", error);
        const summaryText = document.getElementById("history-summary");
        if (summaryText) summaryText.innerText = "No se pudo cargar el historial.";
    }
}

function renderHistoryList(history) {
    const container = document.getElementById("history-details");
    if (!history || history.length === 0) {
        container.innerHTML = '<p style="color:#94a3b8;text-align:center;padding:1rem">Sin historial disponible aún.</p>';
        return;
    }
    container.innerHTML = history.map(h => `
        <div class="history-item-card">
            <div class="history-item-header">
                <span class="${h.is_hit ? 'hit-badge' : 'miss-badge'}">${h.is_hit ? 'ACIERTO' : 'FALLO'}</span>
                <span style="font-size: 0.65rem; color: #64748b;">${new Date(h.date).toLocaleDateString()}</span>
            </div>
            <div class="history-item-teams">${h.match}</div>
            <div class="history-item-score">Final: ${h.actual_score}</div>
            <div class="history-item-stats">
                <div class="stat-pill ${h.details.winner_hit ? 'success' : ''}">🏆 Victoria: ${h.details.winner_hit ? '✓' : '✗'}</div>
                <div class="stat-pill ${h.details.goals_hit ? 'success' : ''}">⚽ Goles ±1: ${h.details.goals_hit ? '✓' : '✗'}</div>
            </div>
        </div>
    `).join("");
}

function toggleHistoryDetails() {
    const details = document.getElementById("history-details");
    details.style.display = details.style.display === "none" ? "grid" : "none";
}

// ─────────────────────────────────────────────
// MATCHES — two-phase loading
// Phase 1: /upcoming-basic → show cards instantly (< 2s)
// Phase 2: /upcoming       → update cards with real predictions
// ─────────────────────────────────────────────
async function loadMatches() {
    if (currentController) currentController.abort();
    currentController = new AbortController();
    const signal = currentController.signal;

    const leagueCode = document.getElementById("league").value;
    const section    = document.getElementById("matches-section");
    const loader     = document.getElementById("grid-loader");
    const grid       = document.getElementById("match-grid");
    const leagueLogo = document.getElementById("league-flag");
    const leagueTitle    = document.getElementById("current-league-name");
    const matchdayBadge  = document.getElementById("matchday-badge");
    const subtitle   = document.getElementById("app-subtitle");

    if (!leagueCode) return;
    const leagueMeta = leaguesData.find(l => l.code === leagueCode);

    // Show short spinner while fetching basic data
    section.style.display = "none";
    loader.classList.remove("loader-hidden");
    loader.innerHTML = `<div class="spinner"></div><p>Cargando partidos...</p>`;
    if (subtitle) subtitle.style.display = "none";

    function showSection() {
        loader.classList.add("loader-hidden");
        section.style.display = "block";
    }
    function updateHeader(data) {
        leagueTitle.innerText = leagueMeta ? leagueMeta.name : "Liga";
        if (leagueMeta?.flag) { leagueLogo.src = leagueMeta.flag; leagueLogo.style.display = "block"; }
        else { leagueLogo.style.display = "none"; }
        matchdayBadge.innerText = `Jornada ${data.matchday || '?'}`;
    }

    // ── PHASE 1: fast basic load ──────────────────────────────────────────
    try {
        const r1 = await fetch(`/upcoming-basic/${leagueCode}`, { signal });
        if (signal.aborted) return;

        if (r1.ok) {
            const basic = await r1.json();
            if (signal.aborted) return;
            updateHeader(basic);
            renderMatchGrid(basic.matches);
            showSection();
            setTimeout(triggerAnimations, 50);

            // ── PHASE 2: enrich with real predictions in background ───────
            loadPredictions(leagueCode, signal);
        } else {
            // Fall back to full load if basic fails
            await loadMatchesFull(leagueCode, signal, leagueMeta, section, loader, grid, leagueLogo, leagueTitle, matchdayBadge, subtitle);
        }
    } catch (err) {
        if (err.name === "AbortError") return;
        console.error("Phase-1 failed, falling back:", err);
        await loadMatchesFull(leagueCode, signal, leagueMeta, section, loader, grid, leagueLogo, leagueTitle, matchdayBadge, subtitle);
    }
}

async function loadPredictions(leagueCode, signal) {
    // Show subtle "calculating" overlay on each card
    document.querySelectorAll(".match-card").forEach(c => c.classList.add("predicting"));

    try {
        const r2 = await fetch(`/upcoming/${leagueCode}`, { signal });
        if (signal.aborted || !r2.ok) return;

        const data = await r2.json();
        if (signal.aborted) return;

        if (data.training_in_progress) {
            scheduleTrainingRetry(leagueCode);
            document.querySelectorAll(".match-card").forEach(c => c.classList.remove("predicting"));
            return;
        }

        // Patch each card's probability bars in-place
        (data.matches || []).forEach(m => {
            const card = document.querySelector(
                `.match-card[data-home-id="${m.homeTeam.id}"][data-away-id="${m.awayTeam.id}"]`
            );
            if (!card || !m.prediction) return;
            const pred = m.prediction;
            const parts = card.querySelectorAll(".quick-bar-part");
            if (parts[0]) { parts[0].setAttribute("data-target", pred.local);     parts[0].style.flex = pred.local; }
            if (parts[1]) { parts[1].setAttribute("data-target", pred.empate);    parts[1].style.flex = pred.empate; }
            if (parts[2]) { parts[2].setAttribute("data-target", pred.visitante); parts[2].style.flex = pred.visitante; }
            const labels = card.querySelectorAll(".quick-labels span");
            if (labels[0]) labels[0].textContent = `L: ${Number(pred.local).toFixed(1)}%`;
            if (labels[1]) labels[1].textContent = `E: ${Number(pred.empate).toFixed(1)}%`;
            if (labels[2]) labels[2].textContent = `V: ${Number(pred.visitante).toFixed(1)}%`;
            card.classList.remove("predicting");
        });
    } catch (err) {
        if (err.name === "AbortError") return;
        console.warn("Predictions fetch failed (non-blocking):", err);
        document.querySelectorAll(".match-card").forEach(c => c.classList.remove("predicting"));
    }
}

// Full load (fallback when basic endpoint unavailable)
async function loadMatchesFull(leagueCode, signal, leagueMeta, section, loader, grid, leagueLogo, leagueTitle, matchdayBadge, subtitle) {
    loader.classList.remove("loader-hidden");
    loader.innerHTML = `<div class="spinner"></div><p>Calculando probabilidades de la jornada...</p>`;
    function showSection() {
        loader.classList.add("loader-hidden");
        section.style.display = "block";
    }
    function updateHeader(data) {
        leagueTitle.innerText = leagueMeta ? leagueMeta.name : "Liga";
        if (leagueMeta?.flag) { leagueLogo.src = leagueMeta.flag; leagueLogo.style.display = "block"; }
        else { leagueLogo.style.display = "none"; }
        matchdayBadge.innerText = `Jornada ${data.matchday || '?'}`;
    }
    try {
        const r = await fetch(`/upcoming/${leagueCode}`, { signal });
        if (signal.aborted) return;
        if (r.status === 202) {
            let msg = "El modelo está entrenándose (~30s).";
            try { const d = await r.json(); if (d?.detail?.message) msg = d.detail.message; } catch(_){}
            leagueTitle.innerText = leagueMeta?.name || "Liga";
            matchdayBadge.innerText = "Entrenando...";
            grid.innerHTML = buildTrainingBanner(msg);
            showSection();
            scheduleTrainingRetry(leagueCode);
            return;
        }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (signal.aborted) return;
        updateHeader(data);
        if (data.training_in_progress) {
            grid.innerHTML = buildTrainingBanner(data.training_message || "Entrenando...");
            showSection();
            scheduleTrainingRetry(leagueCode);
            return;
        }
        renderMatchGrid(data.matches);
        showSection();
        setTimeout(triggerAnimations, 50);
    } catch(err) {
        if (err.name === "AbortError") return;
        leagueTitle.innerText = leagueMeta?.name || "Liga";
        matchdayBadge.innerText = "Error";
        grid.innerHTML = `<div class="no-matches" style="grid-column:1/-1;text-align:center;color:#ff4757;padding:3rem;">
            <div style="font-size:2rem;margin-bottom:1rem">⚠️</div>
            <p>Error al obtener los partidos.</p>
            <p style="font-size:0.85rem;color:#94a3b8;margin-top:0.5rem">${err.message}</p>
            <button onclick="loadMatches()" style="margin-top:1.5rem;padding:0.6rem 1.5rem;background:#3b82f6;color:white;border:none;border-radius:8px;cursor:pointer;">🔄 Reintentar</button>
        </div>`;
        showSection();
    }
}

function buildTrainingBanner(message) {
    return `<div class="no-matches" style="grid-column:1/-1;text-align:center;padding:3rem;">
        <div class="spinner" style="margin:0 auto 1.5rem;width:48px;height:48px;border-color:#60a5fa transparent #60a5fa transparent;"></div>
        <h3 style="color:#60a5fa;margin-bottom:0.75rem">🤖 Entrenando Modelo de IA</h3>
        <p style="color:#94a3b8;max-width:320px;margin:0 auto;">${message}</p>
        <p style="color:#64748b;font-size:0.8rem;margin-top:1rem;">La página se actualizará automáticamente.</p>
    </div>`;
}

function scheduleTrainingRetry(leagueCode) {
    if (trainingPollingTimer) clearTimeout(trainingPollingTimer);
    trainingPollingTimer = setTimeout(() => {
        const current = document.getElementById("league")?.value;
        if (current === leagueCode) loadMatches();
    }, 30000);
}

// ─────────────────────────────────────────────
// MATCH GRID
// ─────────────────────────────────────────────
function renderMatchGrid(matches) {
    const grid = document.getElementById("match-grid");
    if (!matches || matches.length === 0) {
        grid.innerHTML = '<div class="no-matches" style="grid-column:1/-1;text-align:center;color:#94a3b8;padding:3rem;">No hay partidos próximos programados para esta liga.</div>';
        return;
    }

    grid.innerHTML = matches.map(m => {
        const date = new Date(m.utcDate).toLocaleString('es-ES', {
            weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
        });

        const pred = m.prediction || { local: 33.3, empate: 33.3, visitante: 33.3 };
        const isTraining = m.training === true;
        const verdictBadge = isTraining
            ? `<div style="font-size:0.7rem;color:#60a5fa;text-align:center;margin-top:0.3rem">🤖 Calculando...</div>`
            : '';

        return `
            <div class="match-card"
                 data-home-id="${m.homeTeam.id}"
                 data-away-id="${m.awayTeam.id}"
                 data-home-name="${m.homeTeam.name}"
                 data-away-name="${m.awayTeam.name}"
                 data-home-crest="${m.homeTeam.crest || ''}"
                 data-away-crest="${m.awayTeam.crest || ''}"
                 onclick="handleMatchClick(this)">
                <div class="match-teams">
                    <div class="team-mini">
                        <img src="${m.homeTeam.crest || ''}" alt="${m.homeTeam.name}" onerror="this.style.display='none'">
                        <span>${m.homeTeam.name}</span>
                    </div>
                    <div class="card-vs">VS</div>
                    <div class="team-mini">
                        <img src="${m.awayTeam.crest || ''}" alt="${m.awayTeam.name}" onerror="this.style.display='none'">
                        <span>${m.awayTeam.name}</span>
                    </div>
                </div>
                <div class="quick-prediction">
                    <div class="quick-labels">
                        <span>L: ${Number(pred.local).toFixed(1)}%</span>
                        <span>E: ${Number(pred.empate).toFixed(1)}%</span>
                        <span>V: ${Number(pred.visitante).toFixed(1)}%</span>
                    </div>
                    <div class="quick-bar-container">
                        <div class="quick-bar-part local animate-bar" data-target="${pred.local}" style="flex:0"></div>
                        <div class="quick-bar-part draw animate-bar" data-target="${pred.empate}" style="flex:0"></div>
                        <div class="quick-bar-part away animate-bar" data-target="${pred.visitante}" style="flex:0"></div>
                    </div>
                </div>
                ${verdictBadge}
                <div class="match-footer" style="text-align:center;font-size:0.75rem;color:#94a3b8;">
                    ${date}
                </div>
            </div>`;
    }).join("");
}

function handleMatchClick(card) {
    const homeId = card.getAttribute('data-home-id');
    const awayId = card.getAttribute('data-away-id');
    const homeName = card.getAttribute('data-home-name');
    const awayName = card.getAttribute('data-away-name');
    const homeCrest = card.getAttribute('data-home-crest');
    const awayCrest = card.getAttribute('data-away-crest');
    openMatchDetail(homeId, awayId, homeName, awayName, homeCrest, awayCrest);
}

// ─────────────────────────────────────────────
// MATCH DETAIL MODAL
// ─────────────────────────────────────────────
async function openMatchDetail(homeId, awayId, homeName, visitName, homeCrest, visitCrest) {
    const leagueCode = document.getElementById("league").value;
    const modal = document.getElementById("prediction-modal");
    const resultArea = document.getElementById("result");

    modal.style.display = "flex";
    document.body.style.overflow = "hidden";

    resultArea.innerHTML = `
        <div class="modal-match-header">
            <div class="header-team">
                <img src="${homeCrest}" alt="${homeName}" onerror="this.style.display='none'">
                <span>${homeName}</span>
            </div>
            <div class="header-vs">VS</div>
            <div class="header-team">
                <img src="${visitCrest}" alt="${visitName}" onerror="this.style.display='none'">
                <span>${visitName}</span>
            </div>
        </div>
        <div class="loader-container">
            <div class="spinner"></div>
            <p>Generando reporte exhaustivo...</p>
        </div>`;

    try {
        const response = await fetch(`/predict?league_code=${leagueCode}&team_local=${homeId}&team_visitante=${awayId}`);

        if (!response.ok) {
            let errorMsg = "Error en la predicción avanzada";
            try {
                const errData = await response.json();
                if (errData?.detail) {
                    errorMsg = typeof errData.detail === 'string' ? errData.detail : (errData.detail.message || errorMsg);
                }
            } catch (_) {}

            if (response.status === 202) {
                resultArea.innerHTML = `
                    <div style="text-align:center;padding:3rem;">
                        <div class="spinner" style="margin:0 auto 1rem;width:50px;height:50px;border-color:#60a5fa transparent #60a5fa transparent;"></div>
                        <h3 style="color:#60a5fa;margin-bottom:1rem">Entrenando Modelo Inteligente</h3>
                        <p style="color:#94a3b8;">${errorMsg}</p>
                        <p style="color:#64748b;font-size:0.85rem;margin-top:1rem">Cierra esta ventana e intenta de nuevo en ~30 segundos.</p>
                    </div>`;
                return;
            }

            if (response.status === 429 || response.status === 503) {
                resultArea.innerHTML = `
                    <div style="text-align:center;padding:3rem;">
                        <div style="font-size:3rem;margin-bottom:1rem">⏱️</div>
                        <h3 style="color:#fbbf24;margin-bottom:1rem">Límite de Consultas Alcanzado</h3>
                        <p style="color:#94a3b8;">Espera un minuto antes de intentar de nuevo.</p>
                    </div>`;
                return;
            }

            throw new Error(errorMsg);
        }

        const data = await response.json();
        renderDetailedPrediction(data, homeName, visitName, homeCrest, visitCrest);

    } catch (error) {
        resultArea.innerHTML = `<p style="color:#ff4757;padding:2rem;text-align:center;">${error.message}</p>`;
    }
}

function closeModal() {
    document.getElementById("prediction-modal").style.display = "none";
    document.body.style.overflow = "auto";
}

// ─────────────────────────────────────────────
// DETAILED PREDICTION RENDER
// ─────────────────────────────────────────────
function renderDetailedPrediction(data, localName, visitName, localCrest, visitCrest) {
    const resultArea = document.getElementById("result");
    const info = data.modelo_info;
    const dist = data.distribucion_goles;
    const markets = data.metricas_mercado;
    const xG = data.expected_goals;
    const explicacion = info.explicacion;

    if (data.rapidapi_rate_limit) showPushNotification(data.rapidapi_rate_limit);

    let advancedStatsHtml = "";
    if (!data.estadisticas_esperadas) {
        advancedStatsHtml = `
            <div class="section-title" style="margin-top:1.5rem">Métricas de Juego (API-Football)</div>
            <div style="text-align:center;color:#94a3b8;font-size:0.9rem;padding:1rem;font-style:italic;">
                📊 Estadísticas en tiempo real — Próximamente
            </div>`;
    } else {
        const stats = data.estadisticas_esperadas;
        const noteHtml = data.nota
            ? `<div style="text-align:center;color:#ffa502;font-size:0.8rem;margin-bottom:1rem;font-style:italic;">${data.nota}</div>`
            : "";
        advancedStatsHtml = `
            <div class="section-title" style="margin-top:1.5rem">Métricas de Juego (API-Football)</div>
            ${noteHtml}
            <div class="advanced-stats-grid">
                <div class="stat-box">
                    <div class="stat-box-title">Amarillas Esperadas</div>
                    <div class="cards-container">
                        <div class="team-stat">
                            <div class="card-yellow"></div>
                            <span class="team-stat-val">${stats.tarjetas_amarillas.local}</span>
                            <span class="team-stat-name">L</span>
                        </div>
                        <div class="team-stat">
                            <div class="card-yellow"></div>
                            <span class="team-stat-val">${stats.tarjetas_amarillas.visitante}</span>
                            <span class="team-stat-name">V</span>
                        </div>
                    </div>
                </div>
                <div class="stat-box">
                    <div class="stat-box-title">Tiros a Puerta</div>
                    <div class="shots-container">
                        <div class="team-stat">
                            <div class="shot-icon">🎯</div>
                            <span class="team-stat-val">${stats.tiros_arco.local}</span>
                            <span class="team-stat-name">L</span>
                        </div>
                        <div class="team-stat">
                            <div class="shot-icon">🎯</div>
                            <span class="team-stat-val">${stats.tiros_arco.visitante}</span>
                            <span class="team-stat-name">V</span>
                        </div>
                    </div>
                </div>
            </div>`;
    }

    let p_l = data.probabilidades.local;
    let p_e = data.probabilidades.empate;
    let p_v = data.probabilidades.visitante;

    let winnerPrefix = "";
    if (p_l > p_e && p_l > p_v) winnerPrefix = "local";
    else if (p_v > p_e && p_v > p_l) winnerPrefix = "visitante";
    else winnerPrefix = "empate";

    const scorelineRows = data.marcadores_probables.map((s, i) => `
        <div class="scoreline-row ${i === 0 ? 'top' : ''}" data-rank="${i+1}">
            <span class="score-val">${s.local} – ${s.visitante}</span>
            <span class="score-bar-wrap"><span class="score-bar animate-bar" data-target="${Math.min(s.probabilidad * 5, 100)}" style="width:0%"></span></span>
            <span class="score-pct">${s.probabilidad}%</span>
        </div>`).join("");

    let confClass = "confidence-baja";
    if (info.confianza === "Alta") confClass = "confidence-alta";
    else if (info.confianza === "Media") confClass = "confidence-media";

    const modelAgeTxt = info.model_age_days != null
        ? `· Modelo: ${Number(info.model_age_days).toFixed(1)} días`
        : "";

    resultArea.innerHTML = `
        <div class="modal-match-header">
            <div class="header-team">
                <img src="${localCrest}" alt="${localName}" onerror="this.style.display='none'">
                <span>${localName}</span>
            </div>
            <div class="header-vs-container">
                <div class="header-vs">VS</div>
                <div class="xg-badge">xG <span class="xg-val">${xG.local}</span> - <span class="xg-val">${xG.visitante}</span></div>
            </div>
            <div class="header-team">
                <img src="${visitCrest}" alt="${visitName}" onerror="this.style.display='none'">
                <span>${visitName}</span>
            </div>
        </div>

        <div class="model-badge">
            🤖 ${info.tipo} &nbsp;·&nbsp; Confianza: <strong class="${confClass}">${info.confianza}</strong>
            <span style="font-size:0.7rem;color:#64748b;margin-left:0.5rem">${modelAgeTxt}</span>
        </div>

        <div class="section-title">📊 ¿Por qué esta predicción?</div>
        <div class="explanation-verdict">${explicacion}</div>

        <div class="section-title">Probabilidades de Resultado</div>
        <div class="probability-list">
            ${renderProbRow("Local", p_l, "", winnerPrefix === "local")}
            ${renderProbRow("Empate", p_e, "draw", winnerPrefix === "empate")}
            ${renderProbRow("Visitante", p_v, "away", winnerPrefix === "visitante")}
        </div>

        ${advancedStatsHtml}

        <div class="section-title" style="margin-top:1.5rem">Mercados Monetarios</div>
        <div class="market-metrics-grid">
            ${renderMarketBox("Ambos Marcan", markets.btts)}
            ${renderMarketBox("Más de 2.5", markets.over_2_5)}
            ${renderMarketBox("Menos de 2.5", markets.under_2_5)}
            ${renderMarketBox("Arco 0 ("+localName.substring(0,3)+")", markets.clean_sheet_local)}
            ${renderMarketBox("Arco 0 ("+visitName.substring(0,3)+")", markets.clean_sheet_visitante)}
        </div>

        <div class="section-title" style="margin-top:2rem">Distribución de Goles</div>
        <div class="goals-section">
            <div class="team-goals">
                <div class="team-xg-header"><span class="team-xg-name">${localName}</span></div>
                ${renderGoalDist(dist.local, "bar-home")}
            </div>
            <div class="goals-divider"></div>
            <div class="team-goals">
                <div class="team-xg-header"><span class="team-xg-name">${visitName}</span></div>
                ${renderGoalDist(dist.visitante, "bar-away")}
            </div>
        </div>

        <div class="section-title" style="margin-top:2rem">Marcadores Probables</div>
        <div class="scorelines">${scorelineRows}</div>
    `;

    setTimeout(triggerAnimations, 50);
}

// ─────────────────────────────────────────────
// UI HELPERS
// ─────────────────────────────────────────────
function renderProbRow(label, val, colorClass, isGolden) {
    const goldClass = isGolden ? "gold-highlight" : "";
    const favBadge = isGolden ? `<span class="fav-badge">★ FAVORITO</span>` : "";
    return `
        <div class="prob-row ${goldClass}">
            <div class="prob-info">
                <div class="prob-label">${label}${favBadge}</div>
                <div class="prob-value">${val}%</div>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar animate-bar ${colorClass}" data-target="${val}" style="width:0%"></div>
            </div>
        </div>`;
}

function renderMarketBox(label, val) {
    let colorClass = "";
    if (val >= 55) colorClass = "market-high";
    else if (val <= 40) colorClass = "market-low";
    return `
        <div class="market-stat-box ${colorClass}">
            <div class="market-label">${label}</div>
            <div class="market-value">${val}%</div>
            <div class="market-bar-container">
                <div class="market-bar animate-bar" data-target="${val}" style="width:0%"></div>
            </div>
        </div>`;
}

function renderGoalDist(dist, colorClass) {
    const maxProb = Math.max(...dist.map(d => d.probabilidad));
    return dist.map(d => {
        const barWidth = maxProb > 0 ? (d.probabilidad / maxProb) * 100 : 0;
        return `
            <div class="goal-dist-row">
                <div class="goal-label">${d.goles}</div>
                <div class="goal-bar-wrap">
                    <div class="goal-bar animate-bar ${colorClass}" data-target="${barWidth}" style="width:0%"></div>
                </div>
                <div class="goal-pct">${d.probabilidad}%</div>
            </div>`;
    }).join("");
}

function triggerAnimations() {
    document.querySelectorAll('.animate-bar').forEach(bar => {
        const target = bar.getAttribute('data-target');
        if (target !== null) {
            if (bar.classList.contains('quick-bar-part')) {
                bar.style.flex = target;
            } else {
                bar.style.width = target + '%';
            }
        }
    });
}

function showPushNotification(remainingRequests) {
    const container = document.getElementById("toast-container");
    if (!container) return;
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.innerHTML = `<div><strong>API Status</strong><br><span style="font-size:0.8rem">Restantes: ${remainingRequests}</span></div>`;
    container.appendChild(toast);
    setTimeout(() => { if (container.contains(toast)) container.removeChild(toast); }, 5000);
}
