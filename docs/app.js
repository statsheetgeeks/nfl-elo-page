const DATA = "data";

async function loadJSON(name) {
  const res = await fetch(`${DATA}/${name}?_=${Date.now()}`);
  if (!res.ok) throw new Error(`Failed to load ${name}`);
  return res.json();
}

function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { weekday: "short", month: "short", day: "numeric" });
}

function moveIndicator(change) {
  if (change > 0) return `<span class="move-up">&#9650; ${change}</span>`;
  if (change < 0) return `<span class="move-down">&#9660; ${Math.abs(change)}</span>`;
  return `<span class="move-flat">&#9644;</span>`;
}

/* ---------------- rankings ---------------- */

function renderRankingCard(container, { title, rows, primary }) {
  const card = document.createElement("div");
  card.className = "rank-card" + (primary ? " primary" : "");

  const head = document.createElement("div");
  head.className = "rank-card-head";
  head.innerHTML = `<h3>${title}</h3>` + (primary ? `<span class="badge">Used for predictions</span>` : "");
  card.appendChild(head);

  let sortKey = "rank";
  let sortDir = 1;

  const table = document.createElement("table");
  card.appendChild(table);
  container.appendChild(card);

  function draw() {
    const sorted = [...rows].sort((a, b) => (a[sortKey] - b[sortKey]) * sortDir);
    const arrow = (key) => (sortKey === key ? `<span class="sort-arrow">${sortDir === 1 ? "&#9650;" : "&#9660;"}</span>` : "");
    table.innerHTML = `
      <thead>
        <tr>
          <th data-key="rank">#${arrow("rank")}</th>
          <th data-key="team">Team${arrow("team")}</th>
          <th class="num" data-key="rating">Rating${arrow("rating")}</th>
        </tr>
      </thead>
      <tbody>
        ${sorted.map(r => `
          <tr>
            <td class="rank-num">${r.rank}</td>
            <td><span class="team-cell">${r.team} ${moveIndicator(r.rank_change)}</span></td>
            <td class="num">${r.rating.toFixed(1)}</td>
          </tr>
        `).join("")}
      </tbody>
    `;
    table.querySelectorAll("th[data-key]").forEach(th => {
      th.addEventListener("click", () => {
        const key = th.dataset.key;
        if (key === "team") {
          if (sortKey === "team") sortDir *= -1; else { sortKey = "team"; sortDir = 1; }
          rows.sort((a, b) => a.team.localeCompare(b.team) * sortDir);
        } else {
          if (sortKey === key) sortDir *= -1; else { sortKey = key; sortDir = key === "rank" ? 1 : -1; }
        }
        draw();
      });
    });
  }
  draw();
}

function renderRankings(ratings) {
  const grid = document.getElementById("rankings-grid");
  grid.innerHTML = "";
  renderRankingCard(grid, { title: "Combined", rows: ratings.combined, primary: true });
  renderRankingCard(grid, { title: "Classic Elo", rows: ratings.classic });
  renderRankingCard(grid, { title: "G-Elo (MOV)", rows: ratings.geloac });
  renderRankingCard(grid, { title: "ML Elo", rows: ratings.mlelo });
}

/* ---------------- games ---------------- */

function renderGames(predictions) {
  const list = document.getElementById("games-list");
  const sub = document.getElementById("games-subline");
  sub.textContent = `Week ${predictions.week}`;

  if (!predictions.games || predictions.games.length === 0) {
    list.innerHTML = `<div class="empty-note">No games left to predict this week &mdash; check back after Monday night.</div>`;
    return;
  }

  list.innerHTML = predictions.games.map(g => {
    const pct = Math.round(g.predicted_prob * 100);
    return `
      <div class="game-card">
        <div class="game-top">
          <div class="matchup">${g.away_team}<span class="at">at</span>${g.home_team}</div>
          <div class="pick">${g.predicted_winner} ${pct}%</div>
        </div>
        <div class="qb-line">${g.away_team} QB: ${g.away_qb} &nbsp;&middot;&nbsp; ${g.home_team} QB: ${g.home_qb}</div>
        <div class="prob-track"><div class="prob-fill" style="width:${pct}%"></div></div>
      </div>
    `;
  }).join("");
}

/* ---------------- performance ---------------- */

function statCard(label, record) {
  const pct = record.accuracy === null ? "&ndash;" : Math.round(record.accuracy * 100) + "%";
  const record_str = record.n === 0 ? "No games yet" : `${record.wins}-${record.losses}`;
  return `
    <div class="stat-card">
      <div class="stat-label">${label}</div>
      <div class="stat-value">${record_str} ${record.n ? `<span class="pct">(${pct})</span>` : ""}</div>
    </div>
  `;
}

function renderPerformance(perf) {
  document.getElementById("stat-row").innerHTML =
    statCard("Last week", perf.last_week) + statCard("Season", perf.season);

  document.getElementById("bucket-list").innerHTML = perf.confidence_buckets.map(b => {
    const pct = b.accuracy === null ? 0 : Math.round(b.accuracy * 100);
    const label = b.n === 0 ? `${b.label} &mdash; no games yet` : `${b.label} &mdash; ${pct}% correct`;
    return `
      <div>
        <div class="bucket-row-label"><span>${label}</span><span class="n">${b.n} game${b.n === 1 ? "" : "s"}</span></div>
        <div class="prob-track"><div class="prob-fill" style="width:${pct}%"></div></div>
      </div>
    `;
  }).join("");
}

/* ---------------- boot ---------------- */

async function main() {
  try {
    const [ratings, predictions, performance, meta] = await Promise.all([
      loadJSON("ratings.json"), loadJSON("predictions.json"),
      loadJSON("performance.json"), loadJSON("meta.json"),
    ]);
    document.getElementById("meta-line").textContent =
      `Season ${meta.season} &middot; Week ${meta.current_week} &middot; updated ${fmtDate(meta.updated_at)}`
        .replace("&middot;", "\u00b7");
    renderRankings(ratings);
    renderGames(predictions);
    renderPerformance(performance);
  } catch (err) {
    document.getElementById("meta-line").textContent = "No data yet";
    document.getElementById("rankings-grid").innerHTML =
      `<div class="empty-note">Ratings haven't been generated yet. Run the pipeline once to populate docs/data/.</div>`;
    console.error(err);
  }
}

main();
