"""Turn several raw exports into one pooled corpus, reviewably.

The preparation the delivered studies did by hand — detect the columns, tag each
snapshot, strip the product's wrapper, tier out what the interface printed
rather than what a person typed, normalise weights within each snapshot — as a
closed vocabulary of named operations, a plan made of them, and an executor that
enforces the rules a plan cannot enforce about itself.

Nothing is ever deleted: every removal assigns a tier, and the removed rows ship
beside the corpus with the tier that removed them.
"""

from .execute import PrepError, execute
from .inspect import InputProfile, profile_input, profile_inputs
from .plan import InputSpec, PrepOp, PrepPlan
from .planner import heuristic_plan

__all__ = ["PrepError", "execute", "InputProfile", "profile_input", "profile_inputs",
           "InputSpec", "PrepOp", "PrepPlan", "heuristic_plan"]
