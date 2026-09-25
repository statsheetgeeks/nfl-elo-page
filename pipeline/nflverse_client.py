"""
Data client built on nflverse's community-maintained, GitHub-hosted data
instead of ESPN's live site API.

Why the switch: ESPN's site.api.espn.com returns 403 Forbidden for every
request made from a GitHub Actions runner (confirmed - not a header/UA
issue, since a realistic browser header set still gets blocked; almost
certainly an IP-range block on Azure's datacenter ranges). nflverse's data
is republished on GitHub itself (raw.githubusercontent.com and GitHub
Releases), which Actions runners can obviously reach.

Two sources:
  - Schedule/scores: nflverse/nfldata's games.csv - the full historical +
    current-season schedule, with scores populated once a game finishes.
    Confirmed fresh (same-day updates) as of the date this was written.
  - Starting QB: nflverse-data's per-season depth_charts_{year}.csv, an
    ESPN depth-chart scrape refreshed multiple times a day. Confirmed to
    correctly reflect a same-week starter change (verified against the
    Dart-injury -> Winston-starts situation from this project's Week 2).
    ~50MB per season file - fetched ONCE per pipeline run and reused for
    every team's lookup, not re-fetched per team.
"""

from __future__ import annotations
import pandas as pd
from typing import Optional, List, Dict, Any

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
DEPTH_CHART_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{year}.csv"


def fetch_multiple_seasons(seasons: List[int]) -> pd.DataFrame:
    """Full schedule for several seasons at once (used for the one-time
    historical backfill - see build_site_data.bootstrap_history)."""
    df = pd.read_csv(GAMES_URL, low_memory=False)
    return df[df.season.isin(seasons)].copy()


# ---------------------------------------------------------------- schedule
def fetch_season_games(season: int) -> pd.DataFrame:
    """Full schedule for one season: every week, with scores populated
    only for games that have finished."""
    df = pd.read_csv(GAMES_URL, low_memory=False)
    return df[df.season == season].copy()


def games_for_week(season_games: pd.DataFrame, week: int,
                    game_type: str = "REG") -> List[Dict[str, Any]]:
    """Same shape as the old espn_client.parse_games(), so the rest of the
    pipeline doesn't need to change: one dict per game with
    {event_id, date, home_team, away_team, home_score, away_score, completed,
     home_rest, away_rest}. Rest days are used as an ML-Elo feature."""
    wk = season_games[(season_games.week == week) & (season_games.game_type == game_type)]
    games = []
    for _, r in wk.iterrows():
        completed = pd.notna(r.home_score) and pd.notna(r.away_score)
        games.append(dict(
            event_id=r.game_id,
            date=r.gameday,
            home_team=r.home_team, home_abbrev=r.home_team,
            away_team=r.away_team, away_abbrev=r.away_team,
            home_score=int(r.home_score) if completed else None,
            away_score=int(r.away_score) if completed else None,
            completed=bool(completed),
            home_rest=int(r.home_rest) if pd.notna(r.get("home_rest")) else 7,
            away_rest=int(r.away_rest) if pd.notna(r.get("away_rest")) else 7,
        ))
    return games


def week_is_complete(games: List[Dict[str, Any]]) -> bool:
    if not games:
        return False
    return all(g["completed"] for g in games)


# ---------------------------------------------------------------- starters
def fetch_depth_charts(season: int) -> pd.DataFrame:
    """One network call per pipeline run; reuse the returned frame for
    every team's starting-QB lookup via get_starting_qb(df, team)."""
    return pd.read_csv(DEPTH_CHART_URL_TMPL.format(year=season), low_memory=False,
                        usecols=["dt", "team", "player_name", "pos_abb", "pos_rank"])


def get_starting_qb(depth_charts: pd.DataFrame, team_abbrev: str) -> Optional[str]:
    """Most recent depth-chart snapshot's pos_rank==1 QB for this team.
    Returns None ("TBD" at the call site) if the team/position isn't
    found, rather than raising - a missing lookup should never crash a
    pipeline run."""
    sub = depth_charts[(depth_charts.team == team_abbrev) & (depth_charts.pos_abb == "QB")]
    if sub.empty:
        return None
    latest_dt = sub.dt.max()
    starter = sub[(sub.dt == latest_dt) & (sub.pos_rank == 1)]
    if starter.empty:
        return None
    return str(starter.iloc[0].player_name)
