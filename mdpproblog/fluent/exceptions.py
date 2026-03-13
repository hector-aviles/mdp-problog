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
    """Excepción base para todos los errores de MDP-ProbLog."""


class FluentDeclarationError(MDPProbLogError):
    """Error en la declaración sintáctica de un fluente (state_fluent/2).
    Causas: etiqueta desconocida, índice mal formado."""


class FluentInferenceError(MDPProbLogError):
    """Fallo en la inferencia automática (state_fluent/1).
    Causas: origen del argumento indeterminado o inconsistente."""


class FluentCardinalityError(MDPProbLogError):
    """Grupo multivalued con cardinalidad insuficiente.
    Causa: grupo multivalued con menos de 2 opciones tras el grounding."""


class FluentMassConservationError(MDPProbLogError):
    """Violación del Segundo Axioma de Kolmogórov (Conservación de Masa).
    Causa: las probabilidades marginales de un grupo factorizado no suman 1.0
    (multivaluado) o caen fuera de [0, 1] (booleano), indicando reglas
    ProbLog inconsistentes o sobredeterminadas."""
