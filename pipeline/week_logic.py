"""
Week-rollover logic.

Rule: the site stays on week N (showing N's games as "upcoming" /
"in progress") until every game in week N has a final score - i.e. until
Monday Night Football has finished. Only then does a pipeline run advance
to week N+1.

State is persisted in data/state.json so we don't have to re-derive it
from scratch on every run.
"""

from __future__ import annotations
import json
import os

from nflverse_client import fetch_season_games, games_for_week, week_is_complete

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "state.json")


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def determine_current_week(season: int, game_type: str = "REG",
                            max_week: int = 18):
    """
    Returns (current_week, games_for_current_week, season_games_df).

    Starts from the last known week in state.json (or week 1 if none
    recorded for this season). If that week is complete, advances one
    week at a time until it finds a week that is NOT fully complete (or
    hits max_week, i.e. end of regular season).

    season_games_df is returned too so callers (build_site_data.py) don't
    have to re-fetch the whole-season schedule a second time.
    """
    state = load_state()
    week = state.get("current_week", 1) if state.get("season") == season else 1

    season_games = fetch_season_games(season)
    games = games_for_week(season_games, week, game_type)

    while week_is_complete(games) and week < max_week:
        week += 1
        games = games_for_week(season_games, week, game_type)

    save_state({"season": season, "current_week": week})
    return week, games, season_games
