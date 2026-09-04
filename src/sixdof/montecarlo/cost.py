"""Expected cost of an engagement, and confidence intervals on hit rates.

Engagement cost
---------------

Rounds are fired one at a time until the target is destroyed or the magazine
allocation runs out.  With per-round kill probabilities ``p_1 ... p_n``, round
``i`` is only fired if every previous round missed, so the expected number of
rounds is

.. math:: E[N] = \\sum_{i=1}^{n} \\prod_{j<i} (1 - p_j)

and, charging ``c`` per round plus a penalty ``C`` if the target survives all
``n``,

.. math:: E[\\text{cost}] = c\\,E[N] + C \\prod_{j=1}^{n} (1 - p_j).

Confidence intervals
--------------------

Each hit rate comes from ``K`` independent Bernoulli trials, so the normal
approximation gives a half-width ``z sqrt(var / K)``.  The thesis uses the
conservative worst-case variance ``1/4``, which at ``K = 1000`` and
``z = 1.96`` gives a fixed margin of +/-3.10 percentage points regardless of the
observed rate.  The Wilson interval is provided alongside because it behaves
better for the near-zero rates at long range.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Sequence, Tuple

import numpy as np

#: Standard normal quantile for a two-sided 95% interval.
Z_95 = 1.96


@dataclass(frozen=True)
class EngagementCost:
    """Result of :func:`expected_engagement_cost`.

    Attributes
    ----------
    n_rounds:
        Rounds allocated to the engagement.
    expected_rounds:
        Expected number actually fired.
    ammunition_cost:
        ``c * expected_rounds``.
    total_failure_probability:
        Probability that every round misses.
    failure_cost:
        ``C * total_failure_probability``.
    total_expected_cost:
        Sum of the two cost terms.
    success_probability:
        ``1 - total_failure_probability``.
    """

    n_rounds: int
    expected_rounds: float
    ammunition_cost: float
    total_failure_probability: float
    failure_cost: float
    total_expected_cost: float
    success_probability: float

    def to_dict(self) -> Dict[str, float]:
        """Flat dictionary, for tabulating several engagements."""
        return {
            "n_disparos": self.n_rounds,
            "num_esperado_disparos": self.expected_rounds,
            "termo1_custo_municao": self.ammunition_cost,
            "prob_falha_total": self.total_failure_probability,
            "termo2_custo_falha": self.failure_cost,
            "custo_total_esperado": self.total_expected_cost,
            "prob_sucesso": self.success_probability,
        }


def expected_engagement_cost(
    hit_probabilities: Sequence[float],
    round_cost: float,
    target_value: float,
) -> EngagementCost:
    """Expected cost of firing a salvo with the given per-round kill chances.

    Parameters
    ----------
    hit_probabilities:
        Kill probability of each round, in firing order.
    round_cost:
        Cost of one round (``c``), in whatever currency the caller uses.
    target_value:
        Penalty charged if the target survives the whole salvo (``C``).

    Examples
    --------
    >>> result = expected_engagement_cost([0.5, 0.5], 2000, 1_000_000)
    >>> result.expected_rounds
    1.5
    >>> result.total_failure_probability
    0.25
    """
    probabilities = np.asarray(hit_probabilities, dtype=float)
    n = len(probabilities)
    if n == 0:
        raise ValueError("hit_probabilities must not be empty")

    survival_factors = []
    cumulative_failure = 1.0
    for index in range(n):
        survival_factors.append(cumulative_failure)
        if index < n - 1:
            cumulative_failure *= 1 - probabilities[index]

    expected_rounds = float(sum(survival_factors))
    ammunition_cost = round_cost * expected_rounds

    total_failure = float(np.prod(1.0 - probabilities))
    failure_cost = target_value * total_failure

    return EngagementCost(
        n_rounds=n,
        expected_rounds=expected_rounds,
        ammunition_cost=ammunition_cost,
        total_failure_probability=total_failure,
        failure_cost=failure_cost,
        total_expected_cost=ammunition_cost + failure_cost,
        success_probability=1.0 - total_failure,
    )


def wald_interval(
    p_hat: float,
    n_trials: int,
    z: float = Z_95,
    variance: float = 0.25,
) -> Tuple[float, float]:
    """Normal-approximation interval with a fixed variance.

    Parameters
    ----------
    p_hat:
        Observed hit rate, as a proportion.
    n_trials:
        Number of Bernoulli trials.
    z:
        Normal quantile; 1.96 for 95%.
    variance:
        Variance used in the half-width.  The default 0.25 is the worst case
        over all ``p``, which is what the thesis quotes.

    Returns
    -------
    tuple
        ``(lower, upper)``, clipped to ``[0, 1]``.

    Examples
    --------
    >>> lo, hi = wald_interval(0.5, 1000)
    >>> round(hi - 0.5, 4)
    0.031
    """
    margin = z * sqrt(variance / n_trials)
    return (max(0.0, p_hat - margin), min(1.0, p_hat + margin))


def wilson_interval(p_hat: float, n_trials: int, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score interval, which stays inside ``[0, 1]`` near the boundaries.

    Preferred over :func:`wald_interval` when the observed rate is close to 0
    or 1, where the normal approximation produces impossible bounds.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    denominator = 1.0 + z**2 / n_trials
    center = (p_hat + z**2 / (2 * n_trials)) / denominator
    half_width = (
        z * sqrt(p_hat * (1 - p_hat) / n_trials + z**2 / (4 * n_trials**2)) / denominator
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def margin_of_error(n_trials: int, z: float = Z_95, variance: float = 0.25) -> float:
    """Half-width of the fixed-variance interval, as a proportion.

    Examples
    --------
    >>> round(margin_of_error(1000), 4)
    0.031
    """
    return z * sqrt(variance / n_trials)


__all__ = [
    "expected_engagement_cost",
    "EngagementCost",
    "wald_interval",
    "wilson_interval",
    "margin_of_error",
    "Z_95",
]
