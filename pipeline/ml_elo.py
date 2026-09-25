"""
ML Elo: an Elo-style rating whose "expected score" function is fit by
logistic regression over several features, instead of assumed to be the
standard two-outcome logistic curve on rating difference alone.

Design (mirrors GEloAC's bootstrap-then-fit-then-replay pattern):
  1. Bootstrap pass: run a neutral (no home bonus, no extra features)
     Elo-style update to get a rating-difference time series. Also collect
     the non-rating features (rest-day differential, recent-form
     differential, season win-pct differential) for every game - these
     don't depend on which rating trajectory is used, only on the schedule
     and past results, so they're computed once and reused in both passes.
  2. Fit a logistic regression: P(home win) ~ rating_diff + rest_diff +
     form_diff + win_pct_diff. The intercept absorbs the average home-field
     effect automatically (no separate HFA constant needed - it's just
     whatever the model learns).
  3. Replay pass: reset ratings, replay history again, but now the
     "expected score" at each game comes from the fitted model's
     predict_proba on that game's live rating difference + features. The
     update rule itself (theta += k*(observed-expected)) is unchanged.

Kept intentionally modest: logistic regression, not a gradient-boosted
model. With ~1,000-1,500 games and a handful of features, a regularized
linear model is much less likely to overfit than a tree ensemble - this is
meant to be a defensible small ML layer, not a stretch for its own sake.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["rating_diff", "rest_diff", "form_diff", "win_pct_diff"]
FORM_WINDOW = 5          # trailing games used for "recent form"
DEFAULT_REST = 7         # standard week-to-week rest, used when unknown


class _TeamContext:
    """Tracks the causal (pre-game) state needed for non-rating features:
    recent point differential and this-season win/loss record. Call
    observe() AFTER computing a game's features, never before, so a game
    never sees its own outcome."""

    def __init__(self):
        self.recent_margins: Dict[str, List[float]] = {}
        self.season_record: Dict[Tuple[int, str], List[int]] = {}  # (season, team) -> [wins, games]

    def form(self, team: str) -> float:
        margins = self.recent_margins.get(team, [])
        return float(np.mean(margins)) if margins else 0.0

    def win_pct(self, season: int, team: str) -> float:
        rec = self.season_record.get((season, team))
        if not rec or rec[1] == 0:
            return 0.5  # no games yet this season - neutral prior
        return rec[0] / rec[1]

    def observe(self, season: int, home: str, away: str, home_score: int, away_score: int):
        margin = home_score - away_score
        for team, m in [(home, margin), (away, -margin)]:
            lst = self.recent_margins.setdefault(team, [])
            lst.append(m)
            if len(lst) > FORM_WINDOW:
                lst.pop(0)

        home_win = 1 if home_score > away_score else 0
        away_win = 1 if away_score > home_score else 0
        for team, win in [(home, home_win), (away, away_win)]:
            rec = self.season_record.setdefault((season, team), [0, 0])
            rec[0] += win
            rec[1] += 1


def _precompute_context_features(history: List[dict]) -> List[Dict[str, float]]:
    """One pass over history producing, for each game IN ORDER, the
    rest/form/win_pct features as they looked BEFORE that game was played."""
    ctx = _TeamContext()
    out = []
    for row in history:
        season = int(row["season"])
        home, away = row["home_team"], row["away_team"]
        home_rest = float(row.get("home_rest", DEFAULT_REST))
        away_rest = float(row.get("away_rest", DEFAULT_REST))

        out.append(dict(
            rest_diff=home_rest - away_rest,
            form_diff=ctx.form(home) - ctx.form(away),
            win_pct_diff=ctx.win_pct(season, home) - ctx.win_pct(season, away),
        ))
        ctx.observe(season, home, away, int(row["home_score"]), int(row["away_score"]))
    return out


def _bootstrap_rating_diffs(history: List[dict], k: float = 20.0) -> List[float]:
    """Neutral-field Elo pass (no home bonus, no extra features) purely to
    get a rating-difference time series for fitting."""
    ratings: Dict[str, float] = {}

    def get(t):
        return ratings.setdefault(t, 1500.0)

    diffs = []
    for row in history:
        home, away = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        z = get(home) - get(away)
        diffs.append(z)
        expected = 1.0 / (1.0 + 10 ** (-z / 400.0))
        observed = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        delta = k * (observed - expected)
        ratings[home] = get(home) + delta
        ratings[away] = get(away) - delta
    return diffs


def fit_ml_elo(history: List[dict], k: float = 20.0):
    """Returns a fitted sklearn pipeline (StandardScaler + LogisticRegression)
    predicting P(home win) from FEATURE_NAMES. Ties are dropped from the
    training labels (too rare in the NFL to matter, and awkward for a
    binary classifier) but still present in `history` for rating replay."""
    context_features = _precompute_context_features(history)
    rating_diffs = _bootstrap_rating_diffs(history, k=k)

    X, y = [], []
    for row, ctx_feats, z in zip(history, context_features, rating_diffs):
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        if hs == as_:
            continue  # drop the rare tie from training labels
        X.append([z, ctx_feats["rest_diff"], ctx_feats["form_diff"], ctx_feats["win_pct_diff"]])
        y.append(1 if hs > as_ else 0)

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(np.array(X), np.array(y))
    return model


def replay_ml_elo(history: List[dict], model, k: float = 20.0) -> Dict[str, float]:
    """Second pass: replay history with the FITTED model supplying the
    expected score at each game, updating ratings the same Elo-style way.
    Returns the final rating dict."""
    context_features = _precompute_context_features(history)
    ratings: Dict[str, float] = {}

    def get(t):
        return ratings.setdefault(t, 1500.0)

    for row, ctx_feats in zip(history, context_features):
        home, away = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        z = get(home) - get(away)
        features = np.array([[z, ctx_feats["rest_diff"], ctx_feats["form_diff"], ctx_feats["win_pct_diff"]]])
        expected = float(model.predict_proba(features)[0, 1])
        observed = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        delta = k * (observed - expected)
        ratings[home] = get(home) + delta
        ratings[away] = get(away) - delta

    return ratings


def predict_ml_elo(model, ratings: Dict[str, float], context_feats: Dict[str, float],
                    home: str, away: str) -> float:
    """P(home win) for an upcoming game, given final replayed ratings and
    that game's live rest/form/win-pct context features."""
    z = ratings.get(home, 1500.0) - ratings.get(away, 1500.0)
    features = np.array([[z, context_feats["rest_diff"], context_feats["form_diff"],
                           context_feats["win_pct_diff"]]])
    return float(model.predict_proba(features)[0, 1])


def live_context_features(history: List[dict], season: int, home: str, away: str,
                           home_rest: float = DEFAULT_REST, away_rest: float = DEFAULT_REST) -> Dict[str, float]:
    """Rest/form/win-pct features for a game that HASN'T been played yet,
    built from the full recorded history so far (no leakage: this game
    isn't in `history`)."""
    ctx = _TeamContext()
    for row in history:
        ctx.observe(int(row["season"]), row["home_team"], row["away_team"],
                    int(row["home_score"]), int(row["away_score"]))
    return dict(
        rest_diff=float(home_rest) - float(away_rest),
        form_diff=ctx.form(home) - ctx.form(away),
        win_pct_diff=ctx.win_pct(season, home) - ctx.win_pct(season, away),
    )
