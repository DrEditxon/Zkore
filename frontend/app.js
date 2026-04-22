let leaguesData = [];

console.log("App script loaded");

document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM Content Loaded - Initializing...");
    loadLeagues();
});

// Fallback if DOMContentLoaded already fired
if (document.readyState === "complete" || document.readyState === "interactive") {
    console.log("Document already ready - Initializing...");
    loadLeagues();
}

async function loadLeagues() {
    const leagueSelect = document.getElementById("league");
    if (!leagueSelect) {
        console.error("League select element not found!");
        return;
    }
    
    console.log("Fetching leagues...");
    try {
        const response = await fetch("/leagues");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        console.log("Leagues received:", data);
        
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
        console.log("Leagues populated successfully");
    } catch (error) {
        console.error("Error loading leagues:", error);
        leagueSelect.innerHTML = '<option value="">Error al conectar con el servidor</option>';
    }
}

async function loadMatches() {
    const leagueSelect = document.getElementById("league");
    const leagueCode = leagueSelect.value;
    const section = document.getElementById("matches-section");
    const loader = document.getElementById("grid-loader");
    const leagueLogo = document.getElementById("league-flag");
    const leagueTitle = document.getElementById("current-league-name");
    const matchdayBadge = document.getElementById("matchday-badge");

    if (!leagueCode) return;

    console.log(`Loading matches for ${leagueCode}...`);
    const leagueMeta = leaguesData.find(l => l.code === leagueCode);
    
    // UI Transitions
    section.style.display = "none";
    loader.style.display = "block";
    const subtitle = document.getElementById("app-subtitle");
    if (subtitle) subtitle.style.display = "none";

    try {
        const response = await fetch(`/upcoming/${leagueCode}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        console.log("Upcoming matches data received");

        // Update Header info
        leagueTitle.innerText = leagueMeta ? leagueMeta.name : "Liga";
        if (leagueMeta && leagueMeta.flag) {
            leagueLogo.src = leagueMeta.flag;
            leagueLogo.style.display = "block";
        } else {
            leagueLogo.style.display = "none";
        }
        
        matchdayBadge.innerText = `Jornada ${data.matchday || '?'}`;

        renderMatchGrid(data.matches);
        
        loader.style.display = "none";
        section.style.display = "block";
        
        setTimeout(triggerAnimations, 50);
    } catch (error) {
        console.error("Error loading matches:", error);
        loader.innerHTML = `<p style="color: #ff4757; padding: 2rem;">Error al obtener los partidos: ${error.message}</p>`;
    }
}

function renderMatchGrid(matches) {
    const grid = document.getElementById("match-grid");
    if (!matches || matches.length === 0) {
        grid.innerHTML = '<div class="no-matches" style="grid-column: 1/-1; text-align: center; color: #94a3b8; padding: 3rem;">No hay partidos próximos programados para esta liga.</div>';
        return;
    }

    grid.innerHTML = matches.map(m => {
        const date = new Date(m.utcDate).toLocaleString('es-ES', { 
            weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' 
        });

        const pred = m.prediction || { local: 33.3, empate: 33.3, visitante: 33.3 };

        return `
            <div class="match-card" 
                 data-home-id="${m.homeTeam.id}" 
                 data-away-id="${m.awayTeam.id}" 
                 data-home-name="${m.homeTeam.name}" 
                 data-away-name="${m.awayTeam.name}"
                 data-home-crest="${m.homeTeam.crest || 'https://crests.football-data.org/770.svg'}"
                 data-away-crest="${m.awayTeam.crest || 'https://crests.football-data.org/770.svg'}"
                 onclick="handleMatchClick(this)">
                <div class="match-teams">
                    <div class="team-mini">
                        <img src="${m.homeTeam.crest || 'https://crests.football-data.org/770.svg'}" alt="${m.homeTeam.name}" onerror="this.src='https://crests.football-data.org/770.svg'">
                        <span>${m.homeTeam.name}</span>
                    </div>
                    <div class="card-vs">VS</div>
                    <div class="team-mini">
                        <img src="${m.awayTeam.crest || 'https://crests.football-data.org/770.svg'}" alt="${m.awayTeam.name}" onerror="this.src='https://crests.football-data.org/770.svg'">
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
                        <div class="quick-bar-part local animate-bar" data-target="${pred.local}" style="flex: 0"></div>
                        <div class="quick-bar-part draw animate-bar" data-target="${pred.empate}" style="flex: 0"></div>
                        <div class="quick-bar-part away animate-bar" data-target="${pred.visitante}" style="flex: 0"></div>
                    </div>
                </div>

                <div class="match-footer" style="text-align: center; font-size: 0.75rem; color: #94a3b8;">
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

async function openMatchDetail(homeId, awayId, homeName, visitName, homeCrest, visitCrest) {
    const leagueCode = document.getElementById("league").value;
    const modal = document.getElementById("prediction-modal");
    const resultArea = document.getElementById("result");
    
    modal.style.display = "flex";
    resultArea.innerHTML = `
        <div class="modal-match-header">
            <div class="header-team">
                <img src="${homeCrest}" alt="${homeName}">
                <span>${homeName}</span>
            </div>
            <div class="header-vs">VS</div>
            <div class="header-team">
                <img src="${visitCrest}" alt="${visitName}">
                <span>${visitName}</span>
            </div>
        </div>
        <div class="loader-container">
            <div class="spinner"></div>
            <p>Generando reporte exhaustivo...</p>
        </div>
    `;
    document.body.style.overflow = "hidden";

    try {
        const response = await fetch(`/predict?league_code=${leagueCode}&team_local=${homeId}&team_visitante=${awayId}`);
        if (!response.ok) {
            let errorMsg = "Error en la predicción avanzada";
            try {
                const errData = await response.json();
                if (errData && errData.detail) {
                    errorMsg = typeof errData.detail === 'string' ? errData.detail : (errData.detail.message || errorMsg);
                }
            } catch (e) {
                // Keep default message if not JSON
            }
            
            if (response.status === 202) {
                resultArea.innerHTML = `
                    <div style="text-align: center; padding: 3rem;">
                        <div class="spinner" style="margin: 0 auto 1rem auto; width: 50px; height: 50px; border-color: #60a5fa transparent #60a5fa transparent;"></div>
                        <h3 style="color: #60a5fa; margin-bottom: 1rem;">Entrenando Modelo Inteligente</h3>
                        <p style="color: #94a3b8;">${errorMsg}</p>
                        <p style="color: #64748b; font-size: 0.85rem; margin-top: 1rem;">Por favor cierra esta ventana y vuelve a intentarlo en unos 15 a 30 segundos.</p>
                    </div>`;
                return;
            }

            if (response.status === 429 || response.status === 503) {
                resultArea.innerHTML = `
                    <div style="text-align: center; padding: 3rem;">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">⏱️</div>
                        <h3 style="color: #fbbf24; margin-bottom: 1rem;">Límite de Consultas Alcanzado</h3>
                        <p style="color: #94a3b8;">Has realizado demasiadas peticiones en muy poco tiempo.</p>
                        <p style="color: #64748b; font-size: 0.85rem; margin-top: 1rem;">Espera un minuto antes de intentar de nuevo.</p>
                    </div>`;
                return;
            }
            
            throw new Error(errorMsg);
        }
        
        const data = await response.json();
        renderDetailedPrediction(data, homeName, visitName, homeCrest, visitCrest);
    } catch (error) {
        resultArea.innerHTML = `<p style="color: #ff4757; padding: 2rem; text-align: center;">${error.message}</p>`;
    }
}

function closeModal() {
    document.getElementById("prediction-modal").style.display = "none";
    document.body.style.overflow = "auto";
}

function renderDetailedPrediction(data, localName, visitName, localCrest, visitCrest) {
    const resultArea = document.getElementById("result");
    const info = data.modelo_info;
    const dist = data.distribucion_goles;
    const markets = data.metricas_mercado;
    const xG = data.expected_goals;
    const explicacion = info.explicacion;

    if (data.rapidapi_rate_limit) showPushNotification(data.rapidapi_rate_limit);

    let advancedStatsHtml = "";
    if (data.estadisticas_esperadas) {
        const stats = data.estadisticas_esperadas;
        advancedStatsHtml = `
            <div class="section-title" style="margin-top:1.5rem">Métricas de Juego (API-Football)</div>
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
    } else if (data.nota) {
        advancedStatsHtml = `
            <div class="section-title" style="margin-top:1.5rem">Métricas de Juego (API-Football)</div>
            <div style="text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 1rem; font-style: italic;">
                ${data.nota}
            </div>
        `;
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

    resultArea.innerHTML = `
        <div class="modal-match-header">
            <div class="header-team">
                <img src="${localCrest}" alt="${localName}">
                <span>${localName}</span>
            </div>
            <div class="header-vs-container">
                <div class="header-vs">VS</div>
                <div class="xg-badge">xG <span class="xg-val">${xG.local}</span> - <span class="xg-val">${xG.visitante}</span></div>
            </div>
            <div class="header-team">
                <img src="${visitCrest}" alt="${visitName}">
                <span>${visitName}</span>
            </div>
        </div>

        <div class="model-badge">
            🤖 ${info.tipo} &nbsp;·&nbsp; Confianza: <strong>${info.confianza}</strong>
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

function renderProbRow(label, val, colorClass, isGolden) {
    const goldClass = isGolden ? "gold-highlight" : "";
    const favBadge = isGolden ? `<span class="fav-badge">★ FAVORITO</span>` : "";
    return `
        <div class="prob-row ${goldClass}">
            <div class="prob-info"><div class="prob-label">${label}${favBadge}</div><div class="prob-value">${val}%</div></div>
            <div class="prob-bar-container"><div class="prob-bar animate-bar ${colorClass}" data-target="${val}" style="width:0%"></div></div>
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
            <div class="market-bar-container"><div class="market-bar animate-bar" data-target="${val}" style="width:0%"></div></div>
        </div>`;
}

function renderGoalDist(dist, colorClass) {
    const maxProb = Math.max(...dist.map(d => d.probabilidad));
    return dist.map(d => {
        const barWidth = maxProb > 0 ? (d.probabilidad / maxProb) * 100 : 0;
        return `
            <div class="goal-dist-row">
                <div class="goal-label">${d.goles}</div>
                <div class="goal-bar-wrap"><div class="goal-bar animate-bar ${colorClass}" data-target="${barWidth}" style="width:0%"></div></div>
                <div class="goal-pct">${d.probabilidad}%</div>
            </div>`;
    }).join("");
}

function triggerAnimations() {
    const bars = document.querySelectorAll('.animate-bar');
    bars.forEach(bar => {
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
    setTimeout(() => { if(container.contains(toast)) container.removeChild(toast); }, 5000);
}