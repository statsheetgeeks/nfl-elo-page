"""
Thin client around ESPN's public (unofficial, undocumented) NFL JSON
endpoints. These are the same endpoints espn.com's own site uses, and are
widely relied on by other open-source NFL tools since there's no official
public API. They can change without notice, so every parser here fails
soft (returns None / "TBD") rather than raising, and callers should log
when a field comes back empty so it's easy to spot when ESPN changes shape.

NOTE: this file can only be *exercised* from an environment with outbound
internet access (e.g. the GitHub Actions runner) - the sandbox this was
written in cannot reach espn.com to verify exact field names live. Treat
the field paths below as "best known as of research" and confirm against
a real response on the first Actions run (print/log the raw JSON for one
game if a field seems off).
"""

from __future__ import annotations
import requests
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
TIMEOUT = 15

# ESPN's edge (Akamai) rejects requests that look scripted - a generic or
# custom User-Agent, missing Accept headers, etc. A realistic desktop
# browser header set is enough; no cookies/session needed for these
# public endpoints.
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/",
    "Origin": "https://www.espn.com",
}


def _get(url: str, params: dict = None, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT, headers=_HEADERS)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            last_err = e
            if resp.status_code in (403, 429) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise last_err


def get_scoreboard(season: int, week: int, season_type: int = 2) -> dict:
    """
    season_type: 1=preseason, 2=regular season, 3=postseason.
    Returns the raw ESPN scoreboard payload for one week.
    """
    return _get(f"{BASE}/scoreboard", params={"year": season, "week": week,
                                               "seasontype": season_type})


def parse_games(scoreboard_json: dict) -> List[Dict[str, Any]]:
    """
    Extract a simple per-game record from the raw scoreboard payload:
        {event_id, date, home_team, away_team, home_score, away_score,
         completed, home_abbrev, away_abbrev}
    home_score/away_score are None until the game has started.
    """
    games = []
    for event in scoreboard_json.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        status = comp.get("status", {}).get("type", {})
        completed = bool(status.get("completed"))

        def _score(team_obj):
            s = team_obj.get("score")
            try:
                return int(s) if s is not None and completed else None
            except (TypeError, ValueError):
                return None

        games.append(dict(
            event_id=event.get("id"),
            date=event.get("date"),
            home_team=(home.get("team") or {}).get("displayName"),
            home_abbrev=(home.get("team") or {}).get("abbreviation"),
            away_team=(away.get("team") or {}).get("displayName"),
            away_abbrev=(away.get("team") or {}).get("abbreviation"),
            home_score=_score(home),
            away_score=_score(away),
            completed=completed,
        ))
    return games


def week_is_complete(games: List[Dict[str, Any]]) -> bool:
    """True once every game in the week (including a Monday-night finale)
    has a final score. An empty list is treated as NOT complete (safer
    default - don't advance the week if we couldn't fetch the schedule)."""
    if not games:
        return False
    return all(g["completed"] for g in games)


def get_starting_qb(team_abbrev: str) -> Optional[str]:
    """
    Best-effort lookup of a team's current starting QB, for display on the
    upcoming-week game cards. Tries the team's depth chart endpoint first
    (most direct signal); falls back to None ("TBD") if the shape doesn't
    match what's expected, so a pipeline run never crashes on this.

    CONFIRM ON FIRST LIVE RUN: ESPN's depth-chart JSON structure has
    changed before; verify the offense/QB path against a real response.
    """
    try:
        data = _get(f"{BASE}/teams/{team_abbrev}/depthchart")
        for chart in data.get("items", []) or data.get("depthchart", []):
            positions = chart.get("positions", {})
            qb_group = positions.get("qb") or positions.get("QB")
            if qb_group:
                athletes = qb_group.get("athletes", [])
                if athletes:
                    starter = athletes[0].get("athlete", {}) or athletes[0]
                    name = starter.get("displayName") or starter.get("fullName")
                    if name:
                        return name
    except Exception:
        pass
    return None


def get_team_injuries(team_abbrev: str) -> List[Dict[str, str]]:
    """Best-effort injury report for a team; used to sanity-check / flag
    a starting QB who's listed as OUT despite depth-chart data lagging."""
    try:
        data = _get(f"{BASE}/teams/{team_abbrev}", params={"enable": "injuries"})
        team = data.get("team", {})
        out = []
        for inj in team.get("injuries", []):
            athlete = inj.get("athlete", {})
            out.append(dict(name=athlete.get("displayName", "?"),
                             position=(athlete.get("position") or {}).get("abbreviation", "?"),
                             status=inj.get("status", "?")))
        return out
    except Exception:
        return []
