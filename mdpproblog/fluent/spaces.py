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

from collections import OrderedDict
from mdpproblog.fluent.schema import FluentSchema

class FactorSpace(object):
    """
    Abstract base class for iterating over factored valuations using a
    mixed-radix numeral system.

    Provides unified encoding (valuation -> index) and decoding
    (index -> valuation) logic for both :class:`StateSpace` and
    :class:`ActionSpace`. Each valuation is an :class:`~collections.OrderedDict`
    mapping :class:`problog.logic.Term` objects to integer values.

    When `timestep` is provided, the iteration is performed over
    temporally-stamped terms obtained from :meth:`FluentSchema.get_factors_at`.
    When `timestep` is `None`, the atemporal terms of the schema are
    used directly (intended for :class:`ActionSpace`, whose terms carry no
    timestep).

    :param schema: fluent schema defining the factor structure and bases
    :type schema: FluentSchema
    :param timestep: optional timestep for temporal term instantiation
    :type timestep: int or None
    """

    def __init__(self, schema, timestep=None):
        self._schema = schema
        self.__space_size = schema.total_states
        self.__strides = schema.strides

        if timestep is not None:
            self.__local_factors = schema.get_factors_at(timestep)
        else:
            # Atemporal access: read the raw factor lists directly.
            # ActionSpace terms have no timestep argument.
            self.__local_factors = schema.factors

    def __len__(self):
        """
        Return the total number of distinct valuations in this space.

        :rtype: int
        """
        return self.__space_size

    def __iter__(self):
        """
        Reset the iteration counter and return ``self`` as an iterator.

        :rtype: FactorSpace
        """
        self.__current_index = 0
        return self

    def __next__(self):
        """
        Return the next valuation and advance the internal counter.

        :raises StopIteration: when all valuations have been yielded
        :rtype: collections.OrderedDict of (problog.logic.Term, int)
        """
        if self.__current_index >= self.__space_size:
            raise StopIteration
        valuation = self[self.__current_index]
        self.__current_index += 1
        return valuation

    def __getitem__(self, index):
        """
        Decode `index` into a valuation using mixed-radix decomposition.

        For each factor, the active option index is extracted as
        `index % base`, then `index` is floor-divided by `base` to
        process the next factor. Bool factors assign the active index
        (0 or 1) directly to their single term; multivalued factors assign 1 to
        the active option and 0 to all others.

        :param index: integer index in the range `[0, len(self))`
        :type index: int
        :rtype: collections.OrderedDict of (problog.logic.Term, int)
        """
        valuation = OrderedDict()
        temp_index = index

        for base, options in zip(self._schema._FluentSchema__bases, self.__local_factors):
            active = temp_index % base
            temp_index //= base

            if base == 2 and len(options) == 1:
                # Bool: assign the active index (0 or 1) to the single term.
                valuation[options[0]] = active
            else:
                # Multivalued: one-hot encoding over the option list.
                for i, term in enumerate(options):
                    valuation[term] = 1 if i == active else 0

        return valuation


    def index(self, valuation):
        """
        Encode `valuation` into a single integer using mixed-radix encoding.

        For each factor, the active option index is determined and multiplied
        by the corresponding stride. Bool factors read their value directly;
        multivalued factors find the first term with value 1.

        :param valuation: mapping of fluent terms to their current values
        :type valuation: dict of (problog.logic.Term, int)
        :rtype: int
        """
        idx = 0
        for k, options in enumerate(self.__local_factors):
            if len(options) == 1:
                active_value = valuation.get(options[0], 0)
            else:
                active_value = 0
                for i, term in enumerate(options):
                    if valuation.get(term, 0) == 1:
                        active_value = i
                        break
            idx += active_value * self.__strides[k]
        return idx

    @property
    def weights(self):
        """
        Return the stride weights used for mixed-radix indexing.

        These are the same strides exposed by :attr:`FluentSchema.strides`,
        cached at construction time.

        :rtype: list of int
        """
        return self.__strides


class StateSpace(FactorSpace):
    """
    Iterable over all state valuations in a factored MDP.

    Each element is an :class:`~collections.OrderedDict` mapping
    temporally-stamped fluent terms (at the given ``timestep``) to their
    integer values (0 or 1 for Bool; 0/1 one-hot for Multivalued).

    :param schema: fluent schema defining the state space structure
    :type schema: FluentSchema
    :param timestep: timestep at which state fluent terms are stamped
                     (default: 0, representing the current state)
    :type timestep: int
    """

    def __init__(self, schema, timestep=0):
        super(StateSpace, self).__init__(schema, timestep=timestep)


class ActionSpace(FactorSpace):
    """
    Iterable over all action valuations in an MDP.

    Actions are represented as a single mutually exclusive group (annotated
    disjunction), so each element is a one-hot :class:`~collections.OrderedDict`
    over the provided action terms. Action terms carry no timestep argument.

    :param actions: ordered list of action terms
    :type actions: list of problog.logic.Term
    """

    def __init__(self, actions):
        schema = FluentSchema()
        schema.add_group(actions)
        super(ActionSpace, self).__init__(schema, timestep=None)