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

import mdpproblog.engine as eng
from mdpproblog.fluent import Fluent, FluentSchema, StateSpace, ActionSpace
from mdpproblog.fluent import FluentClassifier
from mdpproblog.util import Timer

logger = logging.getLogger("mdpproblog")

class MDP(object):
    """
    Representation of an MDP and its components. Implemented as a bridge
    class to the ProbLog programs specifying the MDP domain and problems.

    :param model: A valid MDP-ProbLog program string.
    :type model: str
    :param epsilon_thr: Threshold used to filter negligible probability mass when
        building structured transitions.
    :type epsilon_thr: float
    :param backend: ProbLog compilation backend. Use None for the let ProbLog automatically choose.
    :type backend: str or None
    """

    def __init__(self, model, epsilon_thr=1e-6, backend=None, darwiche=False):
        self._model = model
        self.epsilon_thr = epsilon_thr

        self._engine = eng.Engine(model, backend=backend, darwiche=darwiche)

        self._eval_cache = {}

        self._prepare()

        self._darwiche = darwiche

    def _prepare(self):
        """Prepare the knowledge database across Classification, Grounding, and Compile phases."""
        self._phase_classification()
        self._phase_grounding()
        self._phase_compile()


    def _phase_classification(self):
        with Timer("Classification"):
            classifier = FluentClassifier(self._engine)
            self.state_schema = classifier.classify()

        self._log_classification(classifier)
        logger.info("FluentSchema:\n%s", self.state_schema)

        self._next_state_factors = self.state_schema.get_factors_at(1)


    def _phase_grounding(self):
        nodes_before = len(self._engine._db._ClauseDB__nodes)

        with Timer("Grounding"):
            self._inject_current_state_fluents()
            actions = self.actions()
            self._engine.add_annotated_disjunction(actions, [1.0 / len(actions)] * len(actions))

            self._utilities = self._engine.assignments("utility")

            current_state_fluents = self.current_state_fluents()
            next_state_fluents = self.next_state_fluents()
            queries = list(set(self._utilities) | set(next_state_fluents) | set(actions) | set(current_state_fluents))
            self._engine.relevant_ground(queries)

        nodes_after = len(self._engine._db._ClauseDB__nodes)
        self._log_grounding(nodes_after - nodes_before, actions, queries)
        self._next_state_fluents = next_state_fluents  # cache for compile


    def _phase_compile(self):
        with Timer("Compile"):
            all_terms = self._next_state_fluents + list(self._utilities)
            self._compiled_nodes = self._engine.compile(all_terms)
            self._next_state_queries = {t: self._compiled_nodes[t] for t in self._next_state_fluents}
            self._reward_queries = {t: self._compiled_nodes[t] for t in self._utilities}

        # str(term) keys of next-state fluents, per factor; computed once.
        self._factor_keys = [[str(t) for t in factor] for factor in self._next_state_factors]

        self._log_compile()


    def _inject_current_state_fluents(self):
        for factor in self.state_schema.factors:
            if len(factor) == 1:
                for term in factor:
                    self._engine.add_fact(Fluent.create_fluent(term, 0), 0.5)
            else:
                current = [Fluent.create_fluent(term, 0) for term in factor]
                self._engine.add_annotated_disjunction( current, [1.0 / len(current)] * len(current))

    # LOG helpers for printing each preparation phase

    def _log_classification(self, classifier):
        explicit_terms = {str(term) for term in classifier._explicit_fluents}

        implicit_terms = []
        for term in classifier._implicit_fluents:
            term_str = str(term)
            if term_str not in explicit_terms:
                implicit_terms.append(term_str)

        implicit_terms = sorted(implicit_terms)

        explicit_block = "\n".join(f"    {t}" for t in explicit_terms) or "    (none)"
        implicit_block = "\n".join(f"    {t}" for t in implicit_terms) or "    (none)"

        logger.debug(
            "Classification: explicit=%d\n%s\n  implicit=%d\n%s",
            len(explicit_terms), explicit_block,
            len(implicit_terms), implicit_block,
        )


    def _log_grounding(self, nodes_added, actions, queries):
        factor_lines = []
        for k, factor in enumerate(self.state_schema.factors):
            if len(factor) == 1:
                factor_lines.append(f"  factor[{k}] bool")
            else:
                factor_lines.append(f"  factor[{k}] multivalued  base={len(factor)}")

        action_names = "  ".join(str(a) for a in actions)
        logger.debug(
            "Grounding: +%d nodes added\n%s\n  actions(%d): [%s]\n  queries grounded: %d",
            nodes_added,
            "\n".join(factor_lines),
            len(actions),
            action_names,
            len(queries),
        )


    def _log_compile(self):
        backend_name = type(self._engine._knowledge).__name__
        ns_lines = "\n".join(
            f"  {str(t):<35} -> node {n}"
            for t, n in sorted(self._next_state_queries.items(), key=lambda x: str(x[0]))
        )
        rw_lines = "\n".join(
            f"  {str(t):<35} -> node {n}"
            for t, n in sorted(self._reward_queries.items(), key=lambda x: str(x[0]))
        )
        logger.debug(
            "Compile: backend=%s\n  [next-state queries]\n%s\n  [reward queries]\n%s",
            backend_name, ns_lines, rw_lines,
        )

    def state_fluents(self):
        """
        Return tSI, he ordered list of atemporal state fluent terms.

        :return: Flat ordered list of atemporal state fluents.
        :rtype: list of problog.logic.Term
        """
        return self.state_schema.get_flat_list()

    def current_state_fluents(self):
        """
        Return the ordered list of current-state fluent terms (t = 0).

        :return: List of state fluents with timestep 0.
        :rtype: list of problog.logic.Term
        """
        return [Fluent.create_fluent(f, 0) for f in self.state_fluents()]

    def next_state_fluents(self):
        """
        Return the ordered list of next-state fluent terms (t = 1).

        :return: List of state fluents with timestep 1.
        :rtype: list of problog.logic.Term
        """
        return [Fluent.create_fluent(f, 1) for f in self.state_fluents()]

    def actions(self):
        """
        Return an ordered list of action objects.

        :rtype: list of action objects sorted by string representation
        """
        return sorted(self._engine.declarations('action'), key=str)
    
    def structured_transition(self, state, action, cache=None):
        """
        Return next-state probabilities grouped by schema factors.

        Each factor is a tuple of ``(local_index, probability)`` branches, where
        ``local_index`` is the factor's contribution to the mixed-radix state
        index (``global_index += local_index * stride[k]``):

        - Boolean factor: index 0 (false) and 1 (true).
        - Multivalued factor: position of the option within the group.

        Only branches whose probability exceeds :attr:`epsilon_thr` are kept.
        The structure is built once per cache key and reused afterwards.

        :param state: Evidence assignment for current-state fluents (t = 0).
        :type state: dict[problog.logic.Term, int]
        :param action: Evidence assignment for actions (one-hot).
        :type action: dict[problog.logic.Term, int]
        :param cache: Optional cache key shared with :meth:`reward`.
        :type cache: object or None
        :return: One tuple of ``(local_index, probability)`` pairs per factor.
        :rtype: tuple[tuple[tuple[int, float], ...], ...]
        """
        return self._cached_eval(state, action, cache)[0]

    def transition(self, state, action):
        """
        Return the marginal probabilities of all next-state fluents given a
        current state and an action. Evaluates the circuit on every call
        (inspection helper; Value Iteration uses :meth:`structured_transition`).

        :param state: Evidence assignment for current-state fluents (t = 0).
        :type state: dict[problog.logic.Term, int]
        :param action: Evidence assignment for actions (one-hot).
        :type action: dict[problog.logic.Term, int]
        :return: List of (term, probability) pairs for next-state fluents.
        :rtype: list[tuple[problog.logic.Term, float]]
        """
        return self._evaluate(state, action)[0]

    def reward(self, state, action, cache=None):
        """
        Return the expected immediate reward for executing an action in a state.

        If cache is provided, the result is memoized and shared with
        :meth:`structured_transition`.

        :param state: Evidence assignment for current-state fluents (t = 0).
        :type state: dict[problog.logic.Term, int]
        :param action: Evidence assignment for actions (one-hot).
        :type action: dict[problog.logic.Term, int]
        :param cache: Optional cache key.
        :type cache: object or None
        :return: Expected immediate reward.
        :rtype: float
        """
        return self._cached_eval(state, action, cache)[1]

    def _evaluate(self, state, action):
        """
        Evaluate the ProbLog circuit once for both transitions and reward.

        :return: Pair of flat transition list and scalar reward.
        :rtype: tuple[list[tuple[problog.logic.Term, float]], float]
        """
        evidence = {**state, **action}
        results = dict(self._engine.evaluate(self._compiled_nodes, evidence))

        flat_transitions = [(t, results[t]) for t in self._next_state_queries]
        reward = sum(results[t] * self._utilities[t].value for t in self._reward_queries)

        return flat_transitions, reward

    def _structure(self, flat_transitions):
        """
        Turn flat marginals into per-factor ``(local_index, probability)`` branches,
        dropping branches with probability <= :attr:`epsilon_thr`.

        :rtype: tuple[tuple[tuple[int, float], ...], ...]
        """
        prob_map = {str(term): prob for term, prob in flat_transitions}
        eps = self.epsilon_thr

        structured = []
        for keys in self._factor_keys:
            if len(keys) == 1:
                p_true = prob_map.get(keys[0], 0.0)
                branches = ((0, 1.0 - p_true), (1, p_true))
            else:
                branches = [(i, prob_map.get(key, 0.0)) for i, key in enumerate(keys)]
            structured.append(tuple((i, p) for i, p in branches if p > eps))
        return tuple(structured)

    def _cached_eval(self, state, action, cache):
        """
        Return ``(structured_transition, reward)``, evaluating the circuit at
        most once per cache key.

        :param cache: Hashable key for memoization, or None to skip caching.
        :type cache: object or None
        :rtype: tuple[tuple, float]
        """
        if cache is not None:
            hit = self._eval_cache.get(cache)
            if hit is not None:
                return hit

        flat_transitions, reward = self._evaluate(state, action)
        result = (self._structure(flat_transitions), reward)

        if cache is not None:
            self._eval_cache[cache] = result
        return result

    def transition_model(self):
        """
        Build the full transition model P(s'|s,a) by enumeration.
 
        This is primarily a debugging/inspection helper.
 
        :return: Mapping from ``(state_values, action_values)`` to next-state probabilities.
        :rtype: dict of ((tuple, tuple), list)
        """
        transitions = {}
        states  = StateSpace(self.state_schema)
        actions = ActionSpace(self.actions())
        for state in states:
            for action in actions:
                transitions[(tuple(state.values()), tuple(action.values()))] = self.transition(state, action)
        return transitions
    
    def reward_model(self):
        """
        Return the reward model of all valid transitions.
 
        :rtype: dict of ((state, action), float)
        """
        rewards = {}
        states  = StateSpace(self.state_schema)
        actions = ActionSpace(self.actions())
        for state in states:
            for action in actions:
                rewards[(tuple(state.values()), tuple(action.values()))] = self.reward(state, action)
        return rewards