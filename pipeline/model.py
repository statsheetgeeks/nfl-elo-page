"""
NFL Elo Rating Models
======================
Two rating engines sharing the same online-update structure:

  ClassicElo  - standard win/loss Elo (FiveThirtyEight-NFL style)
  GEloAC      - Generalized Elo using the discretized-margin-of-victory
                Adjacent Categories (AC) model (Szczecinski, "G-Elo", 2022)

Both implement the update rule:

    theta_i <- theta_i + step * (score - expected_score)

ClassicElo's score/expected-score are the classic 0/0.5/1 win indicator and
the two-outcome logistic curve. GEloAC's score/expected-score are derived
from a fitted multinomial model over discretized margin-of-victory bins, so
a blowout and a squeaker move ratings by different amounts even when both
are technically "wins."

Input data schema (see backtest.py for a loader / simulate.py for synthetic
data):  one row per game, columns:
    season, week, date, home_team, away_team, home_score, away_score
ordered chronologically within each season.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from scipy.optimize import minimize

DEFAULT_INITIAL_RATING = 1500.0


# ----------------------------------------------------------------------
# 1. Classic Elo (baseline)
# ----------------------------------------------------------------------
class ClassicElo:
    """
    Standard Elo rating engine (FiveThirtyEight-NFL style).

    k : update step size. NFL literature suggests K ~ 20 for a 17-game
        season (vs K~5 for MLB), since ratings must move faster with
        fewer games per team.
    home_field_advantage : Elo-point bonus applied to the home team when
        computing win probability (not permanently added to the rating).
        ~55-65 Elo points ~= a 3-point home edge; recalibrate empirically.
    regression_fraction : fraction each team's rating is pulled toward the
        league mean at the start of a new season (FiveThirtyEight uses 1/3,
        to reflect roster turnover / draft / free agency).
    """

    def __init__(self, k: float = 20.0, home_field_advantage: float = 55.0,
                 regression_fraction: float = 1 / 3,
                 initial_rating: float = DEFAULT_INITIAL_RATING):
        self.k = k
        self.hfa = home_field_advantage
        self.regression_fraction = regression_fraction
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        self.history: List[dict] = []
        self._current_season: Optional[int] = None

    def get_rating(self, team: str) -> float:
        return self.ratings.setdefault(team, self.initial_rating)

    def win_probability(self, home: str, away: str) -> float:
        z = (self.get_rating(home) + self.hfa) - self.get_rating(away)
        return 1.0 / (1.0 + 10 ** (-z / 400.0))

    def _maybe_regress_new_season(self, season: int):
        if self._current_season is None:
            self._current_season = season
            return
        if season != self._current_season:
            if self.ratings:
                league_mean = float(np.mean(list(self.ratings.values())))
                for team in self.ratings:
                    r = self.ratings[team]
                    self.ratings[team] = r + self.regression_fraction * (league_mean - r)
            self._current_season = season

    def process_game(self, season: int, week, date, home: str, away: str,
                      home_score: int, away_score: int) -> dict:
        """Predict, then update. Returns a record for backtesting/logging."""
        self._maybe_regress_new_season(season)
        p_home = self.win_probability(home, away)

        if home_score > away_score:
            s_home = 1.0
        elif home_score < away_score:
            s_home = 0.0
        else:
            s_home = 0.5

        r_home_pre, r_away_pre = self.get_rating(home), self.get_rating(away)
        delta = self.k * (s_home - p_home)
        self.ratings[home] = r_home_pre + delta
        self.ratings[away] = r_away_pre - delta

        rec = dict(season=season, week=week, date=date, home=home, away=away,
                   home_score=home_score, away_score=away_score,
                   p_home_pred=p_home, s_home=s_home,
                   home_rating_pre=r_home_pre, away_rating_pre=r_away_pre)
        self.history.append(rec)
        return rec


# ----------------------------------------------------------------------
# 2. G-Elo: Adjacent-Categories margin-of-victory model
# ----------------------------------------------------------------------
class GEloAC:
    """
    Generalized Elo using a discretized-margin-of-victory Adjacent
    Categories model.

    thresholds : ascending positive cut points on the margin |home-away|,
        e.g. [5, 10] yields 7 ordered categories (see `categorize`):
            h=0: d <= -10         (blowout away win)
            h=1: -10 < d <= -5    (moderate away win)
            h=2: -5  < d <  0     (narrow away win)
            h=3: d == 0           (tie - essentially never in NFL)
            h=4: 0   < d <= 5     (narrow home win)
            h=5: 5   < d <= 10    (moderate home win)
            h=6: d > 10           (blowout home win)

    Coefficients alpha_h, delta_h are fit by maximum likelihood on
    historical (rating_diff, category) pairs, subject to the AC model's
    symmetry constraints (alpha_h = alpha_{J-h}, delta_h = -delta_{J-h}).
    Once fit, ratings update online with the same Elo-style form as
    ClassicElo, using a margin-aware score/expected-score pulled from the
    fitted category distribution.
    """

    def __init__(self, thresholds: Sequence[float] = (5.0, 10.0), k: float = 20.0,
                 sigma: float = 400.0, home_field_advantage: float = 55.0,
                 regression_fraction: float = 1 / 3,
                 initial_rating: float = DEFAULT_INITIAL_RATING):
        self.thresholds = sorted(thresholds)
        self.n_side = len(self.thresholds) + 1      # bins per side (away / home)
        self.J = 2 * self.n_side                     # categories are h = 0..J
        self.k = k
        self.sigma = sigma
        self.hfa = home_field_advantage
        self.regression_fraction = regression_fraction
        self.initial_rating = initial_rating

        # free parameters (see symmetry derivation in module docstring / notes):
        #   alpha: n_side free values -> alpha[1..n_side], mirrored to alpha[J-1..n_side]
        #   delta: n_side-1 free values -> delta[1..n_side-1], antisymmetric mirror
        self.free_alpha = np.zeros(self.n_side)
        self.free_delta = np.zeros(max(self.n_side - 1, 0))
        self.alpha = self._build_alpha(self.free_alpha)
        self.delta = self._build_delta(self.free_delta)
        self.delta_tilde = self._to_score_scale(self.delta)  # score assigned to each bin, in [0,1]

        self.ratings: Dict[str, float] = {}
        self.history: List[dict] = []
        self._current_season: Optional[int] = None
        self.is_fit = False

    # ---------------- category assignment ----------------
    def categorize(self, margin: float) -> int:
        """Map a signed home-minus-away margin to a category index h in [0, J]."""
        thr = self.thresholds
        n_side, J = self.n_side, self.J
        if margin == 0:
            return n_side
        if margin < 0:
            m = -margin
            for i, t in enumerate(thr[::-1]):
                if m > t:
                    return i
            return n_side - 1
        else:
            for i, t in enumerate(thr):
                if margin <= t:
                    return n_side + 1 + i
            return J

    # ---------------- symmetric parameter construction ----------------
    def _build_alpha(self, free_alpha: np.ndarray) -> np.ndarray:
        n_side, J = self.n_side, self.J
        alpha = np.zeros(J + 1)
        for i, h in enumerate(range(1, n_side + 1)):
            alpha[h] = free_alpha[i]
            if h != n_side:
                alpha[J - h] = free_alpha[i]
        alpha[0] = 0.0
        alpha[J] = 0.0
        return alpha

    def _build_delta(self, free_delta: np.ndarray) -> np.ndarray:
        n_side, J = self.n_side, self.J
        delta = np.zeros(J + 1)
        delta[0] = -1.0
        delta[J] = 1.0
        delta[n_side] = 0.0
        for i, h in enumerate(range(1, n_side)):
            delta[h] = free_delta[i]
            delta[J - h] = -free_delta[i]
        return delta

    @staticmethod
    def _to_score_scale(delta: np.ndarray) -> np.ndarray:
        """delta_h in [-1,1] -> delta_tilde_h in [0,1] (the 'score' for bin h)."""
        return (delta + 1.0) / 2.0

    # ---------------- probability model ----------------
    def category_probs(self, z: np.ndarray, alpha: Optional[np.ndarray] = None,
                        delta: Optional[np.ndarray] = None) -> np.ndarray:
        """
        P(Y=h|z) for h=0..J, for one or many z values.
        z : rating difference (home - away), same scale as self.sigma.
        Returns array of shape (..., J+1).
        """
        alpha = self.alpha if alpha is None else alpha
        delta = self.delta if delta is None else delta
        z = np.asarray(z, dtype=float)
        exponent = alpha[None, :] + delta[None, :] * (z[..., None] / self.sigma)
        exponent = exponent - exponent.max(axis=-1, keepdims=True)  # numerical stability
        unnorm = 10.0 ** exponent
        return unnorm / unnorm.sum(axis=-1, keepdims=True)

    def expected_score(self, z: np.ndarray, delta: Optional[np.ndarray] = None) -> np.ndarray:
        delta = self.delta if delta is None else delta
        delta_tilde = self._to_score_scale(delta)
        probs = self.category_probs(z, delta=delta)
        return probs @ delta_tilde

    def home_win_probability(self, home: str, away: str) -> float:
        """P(home wins outright), i.e. P(margin > 0), for reporting/backtesting."""
        z = (self.get_rating(home) + self.hfa) - self.get_rating(away)
        probs = self.category_probs(np.array([z]))[0]
        return float(probs[self.n_side + 1:].sum())

    def get_rating(self, team: str) -> float:
        return self.ratings.setdefault(team, self.initial_rating)

    # ---------------- fitting (maximum likelihood) ----------------
    def fit(self, z_values: np.ndarray, categories: np.ndarray,
             l2: float = 1e-3) -> "GEloAC":
        """
        Fit alpha_h, delta_h by maximum likelihood given paired
        (rating_difference, observed_category) data, e.g. produced by a
        first pass of ClassicElo over the training seasons (see backtest.py).

        l2 : small ridge penalty on the free parameters for stability with
             limited data (categories in the tails can be sparse).
        """
        z_values = np.asarray(z_values, dtype=float)
        categories = np.asarray(categories, dtype=int)
        n_alpha, n_delta = self.n_side, max(self.n_side - 1, 0)

        def unpack(p):
            return p[:n_alpha], p[n_alpha:n_alpha + n_delta]

        def neg_log_likelihood(p):
            free_alpha, free_delta = unpack(p)
            alpha = self._build_alpha(free_alpha)
            delta = self._build_delta(free_delta)
            probs = self.category_probs(z_values, alpha=alpha, delta=delta)
            chosen = probs[np.arange(len(categories)), categories]
            chosen = np.clip(chosen, 1e-12, 1.0)
            nll = -np.log(chosen).sum()
            reg = l2 * (np.sum(free_alpha ** 2) + np.sum(free_delta ** 2))
            return nll + reg

        p0 = np.zeros(n_alpha + n_delta)
        result = minimize(neg_log_likelihood, p0, method="L-BFGS-B")
        free_alpha, free_delta = unpack(result.x)
        self.free_alpha, self.free_delta = free_alpha, free_delta
        self.alpha = self._build_alpha(free_alpha)
        self.delta = self._build_delta(free_delta)
        self.delta_tilde = self._to_score_scale(self.delta)
        self.is_fit = True
        return self

    # ---------------- online update ----------------
    def _maybe_regress_new_season(self, season: int):
        if self._current_season is None:
            self._current_season = season
            return
        if season != self._current_season:
            if self.ratings:
                league_mean = float(np.mean(list(self.ratings.values())))
                for team in self.ratings:
                    r = self.ratings[team]
                    self.ratings[team] = r + self.regression_fraction * (league_mean - r)
            self._current_season = season

    def process_game(self, season: int, week, date, home: str, away: str,
                      home_score: int, away_score: int) -> dict:
        """Predict, then update. Returns a record for backtesting/logging."""
        self._maybe_regress_new_season(season)
        r_home_pre, r_away_pre = self.get_rating(home), self.get_rating(away)
        z = (r_home_pre + self.hfa) - r_away_pre

        margin = home_score - away_score
        h = self.categorize(margin)

        p_home_win = self.home_win_probability(home, away)
        expected = float(self.expected_score(np.array([z]))[0])
        observed = float(self.delta_tilde[h])

        step = self.k * self.sigma * (observed - expected) / self.sigma  # = k*(observed-expected)
        # NOTE: kept the *self.sigma/self.sigma explicit above for readability/traceability
        # of the theoretical form theta += K*sigma*(score - G(z)); with K absorbing 1/sigma
        # this reduces to the same practical step size as ClassicElo's k*(s-p).
        self.ratings[home] = r_home_pre + step
        self.ratings[away] = r_away_pre - step

        rec = dict(season=season, week=week, date=date, home=home, away=away,
                   home_score=home_score, away_score=away_score,
                   margin=margin, category=h,
                   p_home_pred=p_home_win, observed_score=observed, expected_score=expected,
                   home_rating_pre=r_home_pre, away_rating_pre=r_away_pre)
        self.history.append(rec)
        return rec
