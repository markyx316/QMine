"""A conversation in front of the pipeline, so nobody has to remember the flags.

It resolves what someone says to a NAMED ACTION with typed parameters, prints
the equivalent command, and stops for agreement before anything that spends
money or writes a file. The model chooses from a fixed catalogue and fills in
parameters; a program decides what actually runs.
"""

from .actions import ACTION_NAMES, CATALOGUE, render_help
from .intent import ChatPlan, ChatStep, keyword_route, route
from .session import ChatState, Session

__all__ = ["ACTION_NAMES", "CATALOGUE", "render_help", "ChatPlan", "ChatStep",
           "keyword_route", "route", "ChatState", "Session"]
