"""
Main weekly pipeline. Run by GitHub Actions (schedule: Wed 5pm, plus
manual workflow_dispatch) or locally for testing.

Steps
-----
1. Figure out the "current" week (week_logic: stays put until Monday
   Night Football of week N has a final score).
2. Pull any newly-completed games into data/games_history.csv (the
   append-only record the rating engines replay each run).
3. Replay ClassicElo and GEloAC over the full history to get current
   ratings; refit GEloAC's margin-of-victory coefficients on the way.
4. Combined rating = average of the two (see note in README on why,
   and how to change the blend later).
5. For any game that JUST became final this run, grade the prediction
   that was live for it (from the previous predictions.json) into
   data/predictions_log.csv - this is what "last week" / "season" /
   "confidence bucket" performance is computed from.
6. Build predictions for the current week's not-yet-played games, using
   Combined ratings + live starting QBs (with manual overrides applied).
7. Write docs/data/{ratings,predictions,performance,meta}.json.

This script intentionally recomputes ratings from full history each run
rather than maintaining incremental state - simpler and safe to rerun
manually (idempotent for a given history + a given week's not-yet-final
games), at the cost of a bit of recomputation. Fine at NFL data volumes
(a few hundred games/season).
"""

from __future__ import annotations
import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

from model import ClassicElo, GEloAC
from nflverse_client import games_for_week, fetch_depth_charts, get_starting_qb, fetch_multiple_seasons
from week_logic import determine_current_week
import ml_elo

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
DOCS_DATA_DIR = os.path.join(ROOT, "docs", "data")
HISTORY_CSV = os.path.join(DATA_DIR, "games_history.csv")
PRED_LOG_CSV = os.path.join(DATA_DIR, "predictions_log.csv")
OVERRIDES_JSON = os.path.join(os.path.dirname(__file__), "starter_overrides.json")

HISTORY_FIELDS = ["season", "week", "event_id", "date", "home_team", "away_team",
                   "home_score", "away_score", "home_rest", "away_rest"]
PRED_LOG_FIELDS = ["season", "week", "event_id", "home_team", "away_team",
                    "predicted_winner", "predicted_prob", "actual_winner", "correct"]

RATING_KWARGS_BASE = dict(k=20.0, regression_fraction=1 / 3)  # home_field_advantage added after calibration

# How many prior seasons to backfill the FIRST time the pipeline ever runs
# (i.e. when data/games_history.csv doesn't exist yet). Without this, every
# team starts a fresh season at a flat 1500 with zero prior information, so
# in the first few weeks the home-field bonus can outweigh the tiny rating
# gap the model has actually had a chance to learn - which is exactly what
# produced the "every home team wins" bug. Backfilling lets real
# prior-season strength carry in through the normal season-to-season
# regression already built into ClassicElo/GEloAC, instead of starting
# blind. Only used once; after that, sync_history() just appends new games.
BACKFILL_SEASONS = 4


# ---------------------------------------------------------------- helpers
def _read_csv(path: str, fields: List[str]) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _append_csv(path: str, fields: List[str], rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load_overrides() -> Dict[str, str]:
    if os.path.exists(OVERRIDES_JSON):
        with open(OVERRIDES_JSON) as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    return {}


def current_nfl_season(today: datetime = None) -> int:
    """NFL 'season year' is the year the season starts (Sept); Jan-Jul
    games belong to the previous year's season."""
    today = today or datetime.now(timezone.utc)
    return today.year if today.month >= 8 else today.year - 1


# ---------------------------------------------------------- history sync
def bootstrap_history(season: int) -> None:
    """One-time only: if HISTORY_CSV doesn't exist yet, backfill
    BACKFILL_SEASONS worth of completed prior-season games so the model
    starts the tracked season with a real prior instead of a blind 1500
    for every team. No-ops on every subsequent run."""
    if os.path.exists(HISTORY_CSV):
        return
    prior_seasons = list(range(season - BACKFILL_SEASONS, season))
    games = fetch_multiple_seasons(prior_seasons)
    games = games[(games.game_type == "REG") & games.home_score.notna()]
    games = games.sort_values(["season", "week"])

    rows = [dict(season=int(r.season), week=int(r.week), event_id=r.game_id,
                  date=r.gameday, home_team=r.home_team, away_team=r.away_team,
                  home_score=int(r.home_score), away_score=int(r.away_score),
                  home_rest=int(r.home_rest) if pd.notna(r.get("home_rest")) else 7,
                  away_rest=int(r.away_rest) if pd.notna(r.get("away_rest")) else 7)
            for _, r in games.iterrows()]
    if rows:
        _append_csv(HISTORY_CSV, HISTORY_FIELDS, rows)
    print(f"Bootstrapped history with {len(rows)} games from seasons {prior_seasons}.")


def sync_history(season: int, up_to_week: int, season_games) -> None:
    """Pull weeks 1..up_to_week from the already-fetched nflverse schedule
    and append any newly-completed games not already recorded in
    HISTORY_CSV."""
    existing = _read_csv(HISTORY_CSV, HISTORY_FIELDS)
    known_ids = {row["event_id"] for row in existing}

    new_rows = []
    for week in range(1, up_to_week + 1):
        for g in games_for_week(season_games, week):
            if g["completed"] and g["event_id"] not in known_ids:
                new_rows.append(dict(season=season, week=week, event_id=g["event_id"],
                                      date=g["date"], home_team=g["home_abbrev"],
                                      away_team=g["away_abbrev"],
                                      home_score=g["home_score"], away_score=g["away_score"],
                                      home_rest=g["home_rest"], away_rest=g["away_rest"]))
    if new_rows:
        _append_csv(HISTORY_CSV, HISTORY_FIELDS, new_rows)


# ---------------------------------------------------------- HFA calibration
def fit_home_field_advantage(history: List[dict], k: float = 20.0) -> float:
    """
    Empirically calibrates the home-field-advantage constant from real
    history, instead of assuming a fixed value. Runs a neutral-field
    (hfa=0) ClassicElo pass to get each game's pre-game rating gap, then
    finds the single additive constant (in Elo points) that best predicts
    the actual home/away outcomes via maximum likelihood.

    Why this exists: a hardcoded guess (~55 points, from a rule of thumb
    in the literature) turned out to be too large for this model's actual
    rating scale - large enough that it was overriding real, meaningful
    rating gaps (e.g. picking a mediocre home team over a genuinely
    stronger visiting team). Recalibrating from data fixes that, and keeps
    fixing it automatically as more of the season accumulates.
    """
    from scipy.optimize import minimize_scalar

    neutral = ClassicElo(k=k, home_field_advantage=0.0, regression_fraction=1 / 3)
    z_values, outcomes = [], []
    for row in history:
        home, away = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        z_values.append(neutral.get_rating(home) - neutral.get_rating(away))
        outcomes.append(1.0 if hs > as_ else (0.0 if hs < as_ else 0.5))
        neutral.process_game(int(row["season"]), int(row["week"]), row["date"], home, away, hs, as_)

    if not z_values:
        return 55.0  # no history yet (shouldn't happen after bootstrap) - fall back to the old guess

    z_values = np.array(z_values)
    outcomes = np.array(outcomes)

    def neg_log_likelihood(hfa):
        p = 1.0 / (1.0 + 10 ** (-(z_values + hfa) / 400.0))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p))

    result = minimize_scalar(neg_log_likelihood, bounds=(-50, 150), method="bounded")
    return float(result.x)


# ---------------------------------------------------------- rating replay
def replay_ratings(home_field_advantage: float):
    """Runs ClassicElo and a freshly-refit GEloAC over the full recorded
    history, in chronological order. Returns (classic, geloac)."""
    history = _read_csv(HISTORY_CSV, HISTORY_FIELDS)
    history.sort(key=lambda r: (int(r["season"]), int(r["week"])))

    rating_kwargs = dict(**RATING_KWARGS_BASE, home_field_advantage=home_field_advantage)
    geloac_kwargs = dict(thresholds=(5.0, 10.0), **rating_kwargs)

    # Pass 1: ClassicElo, also collecting (z, category) pairs for GEloAC fitting.
    classic = ClassicElo(**rating_kwargs)
    geloac_template = GEloAC(**geloac_kwargs)
    z_values, categories = [], []
    for row in history:
        home, away = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        r_home, r_away = classic.get_rating(home), classic.get_rating(away)
        z = (r_home + classic.hfa) - r_away
        z_values.append(z)
        categories.append(geloac_template.categorize(hs - as_))
        classic.process_game(int(row["season"]), int(row["week"]), row["date"],
                              home, away, hs, as_)

    if z_values:
        geloac_template.fit(np.array(z_values), np.array(categories))

    # Pass 2: GEloAC online, using its freshly-fit coefficients throughout.
    geloac = GEloAC(**geloac_kwargs)
    geloac.alpha, geloac.delta = geloac_template.alpha, geloac_template.delta
    geloac.delta_tilde = geloac_template.delta_tilde
    for row in history:
        geloac.process_game(int(row["season"]), int(row["week"]), row["date"],
                             row["home_team"], row["away_team"],
                             int(row["home_score"]), int(row["away_score"]))

    return classic, geloac


def combined_ratings(classic: ClassicElo, geloac: GEloAC) -> Dict[str, float]:
    """Combined = simple average of the two engines' ratings. Easy to
    swap for a weighted blend later (e.g. once we have enough held-out
    performance data to justify weighting one more than the other)."""
    teams = set(classic.ratings) | set(geloac.ratings)
    return {t: (classic.ratings.get(t, classic.initial_rating) +
                geloac.ratings.get(t, geloac.initial_rating)) / 2
            for t in teams}


def win_probability_from_ratings(ratings: Dict[str, float], home: str, away: str,
                                  hfa: float) -> float:
    z = (ratings.get(home, 1500.0) + hfa) - ratings.get(away, 1500.0)
    return 1.0 / (1.0 + 10 ** (-z / 400.0))


# ---------------------------------------------------------- ranking table
def build_rankings(ratings: Dict[str, float], previous: Dict[str, float] = None) -> List[dict]:
    previous = previous or {}
    ranked = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)
    prev_ranked = sorted(previous.items(), key=lambda kv: kv[1], reverse=True) if previous else []
    prev_rank = {team: i + 1 for i, (team, _) in enumerate(prev_ranked)}

    out = []
    for i, (team, rating) in enumerate(ranked):
        rank = i + 1
        prior = prev_rank.get(team)
        if prior is None:
            change = 0
        else:
            change = prior - rank  # positive = moved up
        out.append(dict(team=team, rating=round(rating, 1), rank=rank, rank_change=change))
    return out


# ---------------------------------------------------------- predictions
def grade_completed_predictions(season: int, current_week: int) -> None:
    """For games that just became final, compare their actual result to
    whatever prediction was live for them in the previous predictions.json,
    and append the outcome to PRED_LOG_CSV (idempotent: skips event_ids
    already logged)."""
    pred_path = os.path.join(DOCS_DATA_DIR, "predictions.json")
    if not os.path.exists(pred_path):
        return
    with open(pred_path) as f:
        prior_predictions = json.load(f).get("games", [])

    history = _read_csv(HISTORY_CSV, HISTORY_FIELDS)
    completed_ids = {row["event_id"]: row for row in history}
    already_logged = {row["event_id"] for row in _read_csv(PRED_LOG_CSV, PRED_LOG_FIELDS)}

    new_log_rows = []
    for pred in prior_predictions:
        eid = pred["event_id"]
        if eid in already_logged or eid not in completed_ids:
            continue
        game = completed_ids[eid]
        hs, as_ = int(game["home_score"]), int(game["away_score"])
        actual_winner = game["home_team"] if hs > as_ else game["away_team"]
        correct = int(actual_winner == pred["predicted_winner"])
        new_log_rows.append(dict(season=game["season"], week=game["week"], event_id=eid,
                                  home_team=game["home_team"], away_team=game["away_team"],
                                  predicted_winner=pred["predicted_winner"],
                                  predicted_prob=pred["predicted_prob"],
                                  actual_winner=actual_winner, correct=correct))
    if new_log_rows:
        _append_csv(PRED_LOG_CSV, PRED_LOG_FIELDS, new_log_rows)


def build_predictions(season: int, week: int, week_games: List[dict],
                       ratings: Dict[str, float], depth_charts, hfa: float) -> dict:
    overrides = _load_overrides()
    games_out = []
    for g in week_games:
        if g["completed"]:
            continue  # only forecast games not yet played
        home, away = g["home_abbrev"], g["away_abbrev"]
        p_home = win_probability_from_ratings(ratings, home, away, hfa)
        predicted_winner = home if p_home >= 0.5 else away
        predicted_prob = p_home if p_home >= 0.5 else 1 - p_home

        home_qb = overrides.get(home) or get_starting_qb(depth_charts, home) or "TBD"
        away_qb = overrides.get(away) or get_starting_qb(depth_charts, away) or "TBD"

        games_out.append(dict(
            event_id=g["event_id"], date=g["date"],
            home_team=home, away_team=away,
            home_qb=home_qb, away_qb=away_qb,
            predicted_winner=predicted_winner,
            predicted_prob=round(predicted_prob, 3),
        ))
    return dict(season=season, week=week, generated_at=datetime.now(timezone.utc).isoformat(),
                games=games_out)


def build_performance(season: int, current_week: int) -> dict:
    log = _read_csv(PRED_LOG_CSV, PRED_LOG_FIELDS)
    season_log = [r for r in log if int(r["season"]) == season]
    last_week_log = [r for r in season_log if int(r["week"]) == current_week - 1]

    def _record(rows):
        if not rows:
            return dict(wins=0, losses=0, accuracy=None, n=0)
        wins = sum(int(r["correct"]) for r in rows)
        return dict(wins=wins, losses=len(rows) - wins,
                    accuracy=round(wins / len(rows), 3), n=len(rows))

    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    bucket_out = []
    for lo, hi in buckets:
        rows = [r for r in season_log if lo <= float(r["predicted_prob"]) < hi]
        rec = _record(rows)
        label = f"{int(lo*100)}-{min(int(hi*100), 100)}%"
        bucket_out.append(dict(label=label, **rec))

    return dict(last_week=_record(last_week_log), season=_record(season_log),
                confidence_buckets=bucket_out)


# ---------------------------------------------------------------- main
def main():
    season = current_nfl_season()
    week, week_games, season_games = determine_current_week(season)

    bootstrap_history(season)
    sync_history(season, up_to_week=week, season_games=season_games)
    grade_completed_predictions(season, week)

    history = _read_csv(HISTORY_CSV, HISTORY_FIELDS)
    history.sort(key=lambda r: (int(r["season"]), int(r["week"])))
    hfa = fit_home_field_advantage(history, k=RATING_KWARGS_BASE["k"])
    print(f"Calibrated home_field_advantage: {hfa:.1f} Elo points")

    classic, geloac = replay_ratings(home_field_advantage=hfa)
    combined = combined_ratings(classic, geloac)

    prior_ratings = None
    ratings_path = os.path.join(DOCS_DATA_DIR, "ratings.json")
    if os.path.exists(ratings_path):
        with open(ratings_path) as f:
            prior = json.load(f)
            prior_ratings = {r["team"]: r["rating"] for r in prior.get("combined", [])}

    ratings_out = dict(
        combined=build_rankings(combined, prior_ratings),
        classic=build_rankings(classic.ratings),
        geloac=build_rankings(geloac.ratings),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

    # ML Elo: fourth column, watched alongside the other three rather than
    # feeding into Combined (yet) - see README for the evaluation plan.
    ml_model = ml_elo.fit_ml_elo(history, k=RATING_KWARGS_BASE["k"])
    ml_ratings = ml_elo.replay_ml_elo(history, ml_model, k=RATING_KWARGS_BASE["k"])
    prior_ml_ratings = None
    if os.path.exists(ratings_path):
        with open(ratings_path) as f:
            prior = json.load(f)
            prior_ml_ratings = {r["team"]: r["rating"] for r in prior.get("mlelo", [])}
    ratings_out["mlelo"] = build_rankings(ml_ratings, prior_ml_ratings)

    depth_charts = fetch_depth_charts(season)
    predictions_out = build_predictions(season, week, week_games, combined, depth_charts, hfa)
    performance_out = build_performance(season, week)
    meta_out = dict(season=season, current_week=week, home_field_advantage=round(hfa, 1),
                     updated_at=datetime.now(timezone.utc).isoformat())

    _write_json(os.path.join(DOCS_DATA_DIR, "ratings.json"), ratings_out)
    _write_json(os.path.join(DOCS_DATA_DIR, "predictions.json"), predictions_out)
    _write_json(os.path.join(DOCS_DATA_DIR, "performance.json"), performance_out)
    _write_json(os.path.join(DOCS_DATA_DIR, "meta.json"), meta_out)

    print(f"Season {season}, week {week}: wrote ratings/predictions/performance/meta JSON.")


if __name__ == "__main__":
    main()
