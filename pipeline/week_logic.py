"""
Week-rollover logic.

Rule: the site stays on week N (showing N's games as "upcoming" /
"in progress") until every game in week N has a final score - i.e. until
Monday Night Football has finished. Only then does a pipeline run advance
to week N+1.

State is persisted in data/state.json so we don't have to re-derive it
from scratch (and re-fetch every prior week) on every run.
"""

from __future__ import annotations
import json
import os
from typing import Optional

from espn_client import get_scoreboard, parse_games, week_is_complete

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


def determine_current_week(season: int, season_type: int = 2,
                            max_week: int = 18) -> tuple[int, list, bool]:
    """
    Returns (current_week, games_for_current_week, just_advanced).

    Starts from the last known week in state.json (or week 1 if none
    recorded for this season). If that week is complete, advances one
    week at a time until it finds a week that is NOT fully complete (or
    hits max_week, i.e. end of regular season).
    """
    state = load_state()
    week = state.get("season") == season and state.get("current_week") or 1
    just_advanced = False

    while week <= max_week:
        scoreboard = get_scoreboard(season, week, season_type)
        games = parse_games(scoreboard)
        if week_is_complete(games) and week < max_week:
            week += 1
            just_advanced = True
            continue
        break

    save_state({"season": season, "current_week": week})
    return week, games, just_advanced
