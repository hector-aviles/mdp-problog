# This file is part of MDP-ProbLog.

# MDP-ProbLog is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# MDP-ProbLog is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with MDP-ProbLog.  If not, see <http://www.gnu.org/licenses/>.

"""
mdpproblog.errors - Project-wide exception hierarchy
------------------------------------------------------

Single source of truth for all custom exceptions raised by MDP-ProbLog.
Every module imports from here; no exceptions are defined elsewhere.

..
    Part of the MDP-ProbLog distribution.
    Licensed under the GNU General Public License v3.
"""


class MDPProbLogError(Exception):
    """Base exception for all MDP-ProbLog errors."""


# --- Fluent errors ---

class FluentDeclarationError(MDPProbLogError):
    """Syntactic error in a fluent declaration (``state_fluent/2``)."""


class FluentInferenceError(MDPProbLogError):
    """Failed to infer the type of an implicit fluent (``state_fluent/1``)."""


class FluentCardinalityError(MDPProbLogError):
    """Multivalued group with fewer than 2 options after grounding."""


# --- Engine errors ---

class EngineNodeError(MDPProbLogError):
    """ClauseDB node accessed with an unexpected type.

    Raised when a node retrieved from the ClauseDB is not of the type
    required by the calling method (e.g. a rule node where a fact is
    expected).
    """
