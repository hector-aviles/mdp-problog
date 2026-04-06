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

import logging

from mdpproblog.fluent.schema import FluentSchema
from mdpproblog.fluent.exceptions import (
    FluentDeclarationError, FluentCardinalityError
)
from collections import defaultdict
import warnings

logger = logging.getLogger("mdpproblog")


class FluentClassifier(object):
    """
    Extracts, infers, validates, and packages state fluents from the
    ClauseDB into a :class:`FluentSchema`.
    """
    def __init__(self, engine):
        self._engine = engine
        self._explicit_fluents = self._engine.assignments('state_fluent')
        self._implicit_fluents = self._engine.declarations('state_fluent')
        self._ads_inverted_index = self._engine.get_ads_inverted_index()

    def classify(self):
        """Build and return a validated :class:`FluentSchema`."""
        schema = FluentSchema()

        self._validate_fluent_declarations(self._explicit_fluents, self._implicit_fluents)

        explicit_registry = self._register_explicit(self._explicit_fluents)
        implicit_registry = self._register_implicit(self._implicit_fluents, explicit_registry, self._ads_inverted_index)

        full_registry = {**implicit_registry, **explicit_registry}

        mv_acc = self._separate_mv_fluents(full_registry, schema)
        self._validate_multivalued(schema, mv_acc)

        return schema

    def _register_explicit(self, explicit_fluents):
        """Parse and register explicitly declared fluents (state_fluent/2)."""
        registry = {}
        for term, tag_value in explicit_fluents.items():
            fluent_type = self._parse_fluent_tag(term, tag_value)
            registry[str(term)] = (term, fluent_type)
        return registry

    def _register_implicit(self, implicit_fluents, explicit_registry, ads_inverted_index):
        """Infer and register implicitly declared fluents (state_fluent/1)."""
        registry = {}
        implicit_by_predicate = defaultdict(list)

        for term in implicit_fluents:
            term_str = str(term)
            if term_str not in explicit_registry:
                key = (term.functor, len(term.args))
                implicit_by_predicate[key].append(term)

        for (functor, arity), grounded_terms in implicit_by_predicate.items():
            fluent_type = self._infer_fluent_type(grounded_terms, ads_inverted_index)
            for term in grounded_terms:
                registry[str(term)] = (term, fluent_type)

        return registry

    def _separate_mv_fluents(self, registry, schema):
        """Route bool fluents to the schema and accumulate multivalued groups for validation."""
        mv_accumulator = defaultdict(list)

        for term_str in sorted(registry.keys()):
            term, fluent_type = registry[term_str]
            if fluent_type == 'bool':
                schema.add_bool(term)
            else:
                mv_accumulator[term.functor].append(term)

        return mv_accumulator

    def _validate_multivalued(self, schema, mv_accumulator):
        """Validate cardinality of multivalued groups and register them in the schema."""
        cardinality_errors = []

        for key in sorted(mv_accumulator.keys()):
            terms_group = sorted(mv_accumulator[key], key=str)
            current_domain = {str(t) for t in terms_group}

            if len(current_domain) < 2:
                cardinality_errors.append(FluentCardinalityError(
                    f"Multivalued group '{key}' has only {len(current_domain)} option(s): "
                    f"{current_domain}. A mutually exclusive group requires at least 2 options."
                ))
            else:
                schema.add_group(terms_group)

        if cardinality_errors:
            error_messages = [str(e) for e in cardinality_errors]
            combined = "\n\n".join(f"[Cardinality Error {i+1}] {msg}" for i, msg in enumerate(error_messages))
            raise FluentCardinalityError(f"Found {len(cardinality_errors)} cardinality error(s):\n\n{combined}")


    def _infer_fluent_type(self, grounded_terms, ads_inverted_index):
        """
        Infer the type of an implicit fluent (state_fluent/1).
        Covers arity 0, 1 (including compound terms), and >= 2.

        :param grounded_terms: list of all grounded terms of the same fluent predicate
        :param ads_inverted_index: dict mapping value strings to the set of source predicates they appear in
        :returns: 'bool' or 'multivalued'
        """
        if not grounded_terms:
            return 'bool'

        sample = grounded_terms[0]
        arity = len(sample.args)

        if arity == 0:
            return 'bool'

        # Check whether any argument position belongs entirely to a single AD group.
        ad_positions = []
        for pos in range(arity):
            values_at_pos = {str(t.args[pos]) for t in grounded_terms}
            if not values_at_pos:
                continue

            iterator = iter(values_at_pos)

            first_val = next(iterator)
            common_groups = set(ads_inverted_index.get(first_val, set()))

            if not common_groups:
                continue

            for val in iterator:
                common_groups.intersection_update(ads_inverted_index.get(val, set()))
                if not common_groups:
                    break

            if common_groups:
                ad_positions.append(pos)

        if not ad_positions:
            return 'bool'

        return 'multivalued'

    def _parse_fluent_tag(self, term, tag_value):
        """
        Parse the second argument of state_fluent/2 into a fluent type string.

        :param term: the fluent term (first argument of state_fluent/2)
        :param tag_value: the tag term (second argument of state_fluent/2)
        :returns: 'bool' or 'multivalued'
        :raises FluentDeclarationError: if the tag is unknown
        """
        tag_str = str(tag_value)

        if tag_str == 'bool' or tag_str == 'multivalued':
            return tag_str

        raise FluentDeclarationError(
            f"Unknown type tag '{tag_value}' for fluent '{term}'. "
            f"Valid tags: bool, multivalued."
        )

    def _validate_fluent_declarations(self, explicit_fluents, implicit_fluents):
        """
        Validate explicit fluent tags and warn on implicit/explicit duplicates.

        :raises FluentDeclarationError: if any explicit tag is unknown.
        """
        errors = []

        # V1: Validate explicit fluent tags
        for term, tag_value in explicit_fluents.items():
            try:
                self._parse_fluent_tag(term, tag_value)
            except FluentDeclarationError as e:
                errors.append(e)

        # V2: Check for duplicates
        explicit_functors = {str(t) for t in explicit_fluents.keys()}
        for term in implicit_fluents:
            term_str = str(term)
            if term_str in explicit_functors:
                warnings.warn(
                    f"[V2] Fluent '{term_str}' is declared both implicitly (state_fluent/1) "
                    f"and explicitly (state_fluent/2). The explicit declaration takes precedence.",
                    stacklevel=2
                )

        if errors:
            error_messages = [str(e) for e in errors]
            combined = "\n\n".join(f"[Error {i+1}] {msg}" for i, msg in enumerate(error_messages))
            raise FluentDeclarationError(f"Found {len(errors)} fluent declaration error(s):\n\n{combined}")
