"""Cost guards.

An agent team that fans out five namers over sixty leaves, two annotators over
six hundred rows, and a referee over every disagreement can spend a lot of money
between two glances at the terminal.  The budget is therefore a hard ceiling
enforced at call time, not a number reviewed afterwards.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call that would cross a ceiling."""


@dataclass
class UsageLedger:
    """Thread-safe accounting of everything the team spent.

    Broken down by role as well as in total, because "which agent burned the
    budget" is the first question you ask when a run costs more than expected.
    """

    max_calls: int = 4000
    max_output_tokens: int = 6_000_000
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    #: role -> ($/M input, $/M output), from the routing plan. Empty means every
    #: role falls back to the hardcoded rates and `unpriced_roles` will say so.
    rates: dict[str, tuple[float, float]] = field(default_factory=dict)
    unpriced_roles: list[str] = field(default_factory=list)
    errors: int = 0
    by_role: dict[str, dict[str, int]] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self, role: str = "") -> None:
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded(
                    f"call ceiling reached ({self.calls}/{self.max_calls}); "
                    f"raise llm.max_total_calls or narrow the run"
                )
            if self.output_tokens >= self.max_output_tokens:
                raise BudgetExceeded(
                    f"output-token ceiling reached ({self.output_tokens}/{self.max_output_tokens})"
                )

    def record(
        self,
        role: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached: bool = False,
        error: bool = False,
    ) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cache_hits += int(cached)
            self.errors += int(error)
            slot = self.by_role.setdefault(
                role, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_hits": 0, "errors": 0}
            )
            slot["calls"] += 1
            slot["input_tokens"] += input_tokens
            slot["output_tokens"] += output_tokens
            slot["cache_hits"] += int(cached)
            slot["errors"] += int(error)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_hits": self.cache_hits,
                "errors": self.errors,
                "elapsed_s": round(time.time() - self.started_at, 1),
                "by_role": {k: dict(v) for k, v in self.by_role.items()},
            }

    def _rate_for(self, role: str) -> tuple[float, float] | None:
        """The role's price, matching the way roles are actually named.

        Roles arrive suffixed — `researcher_log_reading`, `namer_3` — while the
        routing plan is keyed on the base role. An exact-match lookup silently
        misses every suffixed agent, which is the same trap `route_for` documents,
        and here it would quietly price them at the fallback rate instead.
        """
        if role in self.rates:
            return self.rates[role]
        for base in sorted(self.rates, key=len, reverse=True):
            if role.startswith(base):
                return self.rates[base]
        return None

    def estimated_cost_usd(self, in_rate: float = 3.0, out_rate: float = 15.0) -> float:
        """Cost at the routed models' own prices, per role.

        This used to charge every token at a hardcoded $3/$15 per million —
        frontier rates — while the run was actually on `deepseek-v4-flash` at
        $0.44/$1.32 and `qwen3-next-80b` at $0.15/$1.20. Measured on a live run:
        the ledger reported **$33.61** where the true cost was **$3.06**, and every
        figure this project has ever quoted was inflated by about 11x. Worse, it
        was compared against the planner's estimate — which uses real catalogue
        prices — and the *planner* was blamed for the gap.

        `rates` is filled from the routing plan, so the numbers agree with the
        thing that chose the models. The arguments remain only as the fallback for
        a role whose model carries no price, and `unpriced_roles` names them
        rather than letting a guess pass as a measurement.
        """
        with self._lock:
            total = 0.0
            unpriced = []
            for role, d in self.by_role.items():
                rate = self._rate_for(role)
                if rate is None:
                    unpriced.append(role)
                    rate = (in_rate, out_rate)
                total += (d["input_tokens"] * rate[0] + d["output_tokens"] * rate[1]) / 1_000_000
            self.unpriced_roles = sorted(unpriced)
            return total
