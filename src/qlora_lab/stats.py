"""Intervals and paired tests, so no rate in this repository is printed bare.

Thirty lines, repeated rather than imported. The fuller treatment — Cohen's kappa, Fleiss,
bootstrap for means, grader calibration — is a separate published package of mine,
`legal-llm-evals`. Depending on it here would mean a git dependency in CI for two functions, and
the dependency would cost more than it saves. What is repeated is the arithmetic, not the
argument: the argument is that a difference of four points on 300 records is not a difference,
and the only way to keep that honest is to make the interval impossible to omit.

`Wilson` and not the textbook normal approximation because the rates that matter here are near
the ends — JSON validity at 0.99, asserted absence at 0.02 — and the normal approximation puts
the bound above 1 or below 0 exactly there, which is where a reader is most likely to believe it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: 1.959964 is the two-sided 95% normal quantile. Spelled out rather than imported so the
#: module has no dependencies at all.
Z95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    """A proportion and the range the data actually supports."""

    successes: int
    total: int
    low: float
    high: float

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def __str__(self) -> str:
        if not self.total:
            return "n=0"
        return f"{self.rate:6.1%} [{self.low:.1%}, {self.high:.1%}] n={self.total}"

    def separated_from(self, other: Interval) -> bool:
        """True when the intervals do not overlap.

        A weaker claim than a paired test and a stronger one than comparing two point estimates.
        Where the two systems ran on the same records, prefer `sign_test`: overlapping intervals
        can still hide a consistent per-record difference, and this repository has a case of it.
        """
        return self.high < other.low or other.high < self.low


def wilson(successes: int, total: int, z: float = Z95) -> Interval:
    if total <= 0:
        return Interval(0, 0, 0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return Interval(successes, total, max(0.0, centre - spread), min(1.0, centre + spread))


def _binomial_tail(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2), computed exactly in integers.

    Exact rather than approximated because the discordant counts here are small — a comparison
    may turn on 14 pairs against 2 — and that is precisely where a normal approximation to a
    binomial is worst.
    """
    total = sum(math.comb(n, i) for i in range(k + 1))
    return total / (2**n)


@dataclass(frozen=True)
class Paired:
    """The result of comparing two systems record by record."""

    better: int
    worse: int
    same: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.better + self.worse

    def __str__(self) -> str:
        return (
            f"{self.better} improved, {self.worse} regressed, {self.same} unchanged, "
            f"p={self.p_value:.4f}"
        )


def sign_test(a: list[bool], b: list[bool]) -> Paired:
    """Exact two-sided sign test over the records where the two systems disagree.

    Two systems scored on the same 300 records are not two independent samples, and treating
    them as such throws away the pairing — which is the whole reason the same records were used.
    Only the discordant pairs carry information: a record both got right and a record both got
    wrong say nothing about which is better.
    """
    if len(a) != len(b):
        raise ValueError("the two systems must have been scored on the same records")
    better = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    worse = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    same = len(a) - better - worse
    n = better + worse
    if n == 0:
        return Paired(0, 0, same, 1.0)
    k = min(better, worse)
    # Two-sided: double the smaller tail, and never report more than certainty.
    p = min(1.0, 2 * _binomial_tail(k, n))
    return Paired(better, worse, same, p)
