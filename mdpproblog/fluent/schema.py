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

from problog.logic import Constant

class Fluent(object):
    """
    Factory class for building temporally-stamped fluent terms.

    A fluent term is a :class:`problog.logic.Term` whose last argument is a
    :class:`problog.logic.Constant` representing a discrete timestep. All
    other arguments are inherited from the original atemporal term.
    """

    @classmethod
    def create_fluent(cls, term, timestep):
        """
        Return a new term derived from ``term`` with ``timestep`` appended
        as its last argument.

        :param term: an atemporal ProbLog term
        :type term: problog.logic.Term
        :param timestep: discrete timestep value to append
        :type timestep: int
        :rtype: problog.logic.Term
        """
        args = term.args + (Constant(timestep),)
        return term.with_args(*args)


class FluentSchema(object):
    """
    Descriptor of the factored structure of an MDP state space.

    Maintains an ordered list of *factors*, where each factor is either:

    - A Boolean State Fluent (bool): a single binary variable with
      base 2, taking values in {0, 1}.
    - A Multivalued group (multivalued): a mutually exclusive set of
      N options with base N, where exactly one option is active at a time
      (one-hot encoding).

    The schema stores atemporal terms only. Temporal copies for a specific
    timestep are produced on demand via :meth:`get_factors_at`.

    The total state space size is the product of all factor bases:
    `b_1 * b_2 * ... * b_n`, where `b_i` is the base of factor `i`.

    Mixed-radix indexing assigns a positional stride to each factor, computed
    as the cumulative product of all preceding bases. For example, bases
    `[2, 3, 2]` yield strides `[1, 2, 6]` and a state space of size 12.
    """

    def __init__(self):
        self.__factors = []    # list of list of Term
        self.__bases = []      # list of int (base per factor)
        self.__flattened = []  # flat list of all terms in registration order
        self.__strides_cache = None

    def add_bool(self, term):
        """
        Register a Boolean State Fluent (binary variable) as a new factor.

        The factor is stored as a single-element list ``[term]`` with base 2.

        :param term: atemporal state fluent term
        :type term: problog.logic.Term
        """
        self.__factors.append([term])
        self.__bases.append(2)
        self.__flattened.append(term)
        self.__strides_cache = None

    def add_group(self, terms):
        """
        Register a mutually exclusive multivalued group as a new factor.

        The factor is stored as a list of N terms with base N. Within the
        group, exactly one term is active at any given state (one-hot).

        :param terms: ordered list of mutually exclusive state fluent terms
        :type terms: list of problog.logic.Term
        """
        self.__factors.append(list(terms))
        self.__bases.append(len(terms))
        self.__flattened.extend(terms)
        self.__strides_cache = None

    # Schema properties

    @property
    def factors(self):
        """
        Return the list of registered factors.

        Each element is a list of one or more atemporal
        :class:`problog.logic.Term` objects. Single-element lists represent
        bool factors; multi-element lists represent multivalued groups.

        :rtype: list of list of problog.logic.Term
        """
        return self.__factors

    @property
    def total_states(self):
        """
        Return the total number of states in the state spaces.

        :rtype: int
        """
        product = 1
        for base in self.__bases:
            product *= base
        return product

    @property
    def strides(self):
        """
        Return the positional stride for each factor in the mixed-radix encoding.

        Stride `k` is the product of all bases with index less than `k`,
        i.e. For bases `[2, 3, 2]` the result is `[1, 2, 6]`.

        The result is cached after the first call; the cache is invalidated
        whenever a new factor is added via :meth:`add_bool` or
        :meth:`add_group`.

        :rtype: list of int
        """
        if self.__strides_cache is None:
            strides = []
            current = 1
            for base in self.__bases:
                strides.append(current)
                current *= base
            self.__strides_cache = strides
        return self.__strides_cache

    def get_factors_at(self, timestep):
        """
        Return a temporally-stamped copy of all factors for ``timestep``.

        Each atemporal term in every factor is converted to a temporal term
        via :meth:`Fluent.create_fluent`. The structure (bool vs. multivalued) and
        registration order of the original schema are preserved.

        :param timestep: discrete timestep value to stamp onto every term
        :type timestep: int
        :rtype: list of list of problog.logic.Term
        """
        return [
            [Fluent.create_fluent(term, timestep) for term in group]
            for group in self.__factors
        ]

    def get_flat_list(self):
        """
        Return a flat list of all registered atemporal terms in schema order.

        The order matches the registration sequence: bool terms appear in the
        order they were added via :meth:`add_bool`; multivalued terms appear in the
        order their group was added via :meth:`add_group`, with terms within
        each group in their original list order.

        :rtype: list of problog.logic.Term
        """
        return self.__flattened

    def get_local_index(self, factor_index, temporal_term):
        """
        Return the local index of `temporal_term` within factor `factor_index`.

        The local index determines the contribution of this factor to the
        global mixed-radix state index: `global_index += local_index * stride`.

        :param factor_index: index of the factor within the schema
        :type factor_index: int
        :param temporal_term: a temporally-stamped fluent term, or `None`
                              for the False branch of a bool factor
        :type temporal_term: problog.logic.Term or None
        :raises ValueError: if the term does not match any entry in the factor
        :rtype: int
        """
        factor = self.__factors[factor_index]

        # Bool False branch: None signals the inactive (0) side of a binary variable.
        if temporal_term is None:
            return 0

        # Strip the timestep argument to recover the atemporal base term.
        base_term = temporal_term.with_args(*temporal_term.args[:-1])

        if len(factor) == 1:
            # Bool True branch: verify the term matches and return index 1.
            if factor[0] == base_term:
                return 1
            raise ValueError(
                f"Term '{base_term}' does not match bool factor '{factor[0]}' "
                f"at factor index {factor_index}."
            )

        # Multivalued branch: linear search within the mutually exclusive group.
        for i, term in enumerate(factor):
            if term == base_term:
                return i

        raise ValueError(
            f"Term '{base_term}' not found in multivalued factor {factor_index}: {factor}."
        )

    def __str__(self):
        """
        Return a human-readable summary of the schema structure.

        Lists all Bool variables and Multivalued groups along with their bases and
        the total size of the resulting state space.

        :rtype: str
        """
        lines = []
        lines.append("=" * 60)
        lines.append(" MDP State Schema")
        lines.append("=" * 60)
        lines.append(f"Total state space size: {self.total_states}")
        lines.append("-" * 60)

        isf_list = []
        ads_list = []

        for i, factor in enumerate(self.__factors):
            if len(factor) == 1:
                isf_list.append(factor[0])
            else:
                ads_list.append((i, factor))

        # bool section
        lines.append(f"[BOOL] Boolean State Fluents ({len(isf_list)})")
        lines.append("      Each iterates independently over {{0, 1}}.")
        if not isf_list:
            lines.append("      (none)")
        else:
            for term in isf_list:
                lines.append(f"      [ ] {term}")
        lines.append("")

        # multivalued section
        lines.append(f"[MULTIVALUED] Multivalued State Fluents ({len(ads_list)})")
        lines.append("      Exactly one option is true per group (one-hot).")
        if not ads_list:
            lines.append("      (none)")
        else:
            for idx, options in ads_list:
                lines.append(f"      > Group #{idx}  (base: {len(options)})")
                for term in options:
                    lines.append(f"          (o) {term}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
