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
# along with MDP-ProbLog.  If not, see <http://www.gnu.org/licenses/>.\

class MDPProbLogError(Exception):
    """Base exception for all MDP-ProbLog errors."""


class FluentDeclarationError(MDPProbLogError):
    """Syntactic error in a fluent declaration (state_fluent/2)."""


class FluentInferenceError(MDPProbLogError):
    """Failed to infer the type of an implicit fluent (state_fluent/1)."""


class FluentCardinalityError(MDPProbLogError):
    """Multivalued group with fewer than 2 options after grounding."""


class FluentMassConservationError(MDPProbLogError):
    """Kolmogorov axiom violation: marginal probabilities do not sum to 1."""
