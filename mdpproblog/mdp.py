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

import mdpproblog.engine as eng
from mdpproblog.fluent import Fluent, FluentSchema, StateSpace, ActionSpace
from mdpproblog.fluent import FluentClassifier

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

    def __init__(self, model, epsilon_thr=1e-6, backend=None):
        self._model = model
        self.epsilon_thr = epsilon_thr

        self._engine = eng.Engine(model, backend=backend)

        self._eval_cache = {}

        self._prepare()

    def _prepare(self):
        """ Prepare the mdp-problog knowledge database to accept queries. """

        # classify fluents and build schema
        classifier = FluentClassifier(self._engine)
        self.state_schema = classifier.classify()

        # templates for next-state factors (t = 1) used by structured_transition
        self._next_state_factors = self.state_schema.get_factors_at(1)

        # add dummy current state fluents probabilistic facts
        for factor in self.state_schema.factors:
            if len(factor) == 1:
                for term in factor:
                    fluent_term = Fluent.create_fluent(term, 0)
                    self._engine.add_fact(fluent_term, 0.5)
            else:
                current_fluents =[]
                for term in factor: 
                    current_fluents.append(Fluent.create_fluent(term, 0)) #t = 0
                self._engine.add_annotated_disjunction(current_fluents, [1.0 / len(current_fluents)] * len(current_fluents))

        # add dummy actions annotated disjunction
        actions = self.actions()
        self._engine.add_annotated_disjunction(actions, [1.0 / len(actions)] * len(actions))

        # utility assignments are used to compute expected immediate rewards. 
        self._utilities = self._engine.assignments('utility')

        # ground only what is relevant to rewards, transitions and evidence.
        current_state_fluents = self.current_state_fluents()
        next_state_fluents = self.next_state_fluents()
        queries = list(set(self._utilities) | set(next_state_fluents) | set(actions) | set(current_state_fluents))
        
        self._engine.relevant_ground(queries)

        # compile query database once; map both transition and reward terms.
        self._compiled_nodes = self._engine.compile(next_state_fluents, list(self._utilities))
        self._next_state_queries = {t: self._compiled_nodes[t] for t in next_state_fluents}
        self._reward_queries = {t: self._compiled_nodes[t] for t in self._utilities}

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

        Transforms the flat transition list into a factorized representation
        aligned with :attr:`state_schema`.

        Factor semantics:
    
        - Boolean factor: includes an explicit false branch (term is None) with
            probability 1 - p_true.
        - Multivalued factor: returns only branches whose probability exceeds
            :attr:`epsilon_thr`.
 
        :param state: Evidence assignment for current-state fluents (t = 0).
        :type state: dict[problog.logic.Term, int]
        :param action: Evidence assignment for actions (one-hot).
        :type action: dict[problog.logic.Term, int]
        :param cache: Optional cache key shared with :meth:`transition` and :meth:`reward`.
        :type cache: object or None
        :return: List of factors; each factor is a list of (term, probability) pairs.
            For Boolean factors, term may be None to denote the false branch.
        :rtype: list[list[tuple[problog.logic.Term | None, float]]]
        """
        flat_transitions, _ = self._cached_eval(state, action, cache)
        prob_map = {str(term): prob for term, prob in flat_transitions}
 
        structured_result = []
        for factor_template in self._next_state_factors:
            group_data = []
            if len(factor_template) == 1:
                term = factor_template[0]
                p_true = prob_map.get(str(term), 0.0)
                p_false = 1.0 - p_true
                if p_false > self.epsilon_thr:
                    group_data.append((None, p_false))
                if p_true > self.epsilon_thr:
                    group_data.append((term, p_true))
            else:
                for term in factor_template:
                    p = prob_map.get(str(term), 0.0)
                    if p > self.epsilon_thr:
                        group_data.append((term, p))
            structured_result.append(group_data)
 
        return structured_result

    def transition(self, state, action, cache=None):
        """
        Return the marginal probabilities of all next-state fluents given a
        current state and an action.

        If cache is provided, results are memoized and subsequent calls with
        the same key reuse the stored evaluation without re-running the circuit.

        :param state: Evidence assignment for current-state fluents (t = 0).
        :type state: dict[problog.logic.Term, int]
        :param action: Evidence assignment for actions (one-hot).
        :type action: dict[problog.logic.Term, int]
        :param cache: Optional cache key shared with :meth:`reward`.
        :type cache: object or None
        :return: List of (term, probability) pairs for next-state fluents.
        :rtype: list[tuple[problog.logic.Term, float]]
        """
        flat_transitions, _ = self._cached_eval(state, action, cache)
        return flat_transitions

    def reward(self, state, action, cache=None):
        """
        Return the expected immediate reward for executing an action in a state.
 
        If cache is provided, results are memoized and subsequent calls with
        the same cache key will not re-evaluate the ProbLog circuit.
 
        :param state: Evidence assignment for current-state fluents (t = 0).
        :type state: dict[problog.logic.Term, int]
        :param action: Evidence assignment for actions (one-hot).
        :type action: dict[problog.logic.Term, int]
        :param cache: Optional cache key.
        :type cache: object or None
        :return: Expected immediate reward.
        :rtype: float
        """
        _, reward = self._cached_eval(state, action, cache)
        return reward

    def _transition_and_reward(self, state, action):
        """
        Evaluate the ProbLog circuit once for both transitions and reward.
 
        :param state: Evidence assignment for current-state fluents (t = 0).
        :type state: dict[problog.logic.Term, int]
        :param action: Evidence assignment for actions (one-hot).
        :type action: dict[problog.logic.Term, int]
        :return: Pair of flat transition list and scalar reward.
        :rtype: tuple[list[tuple[problog.logic.Term, float]], float]
        """
        evidence = {**state, **action}
        query_nodes = self._compiled_nodes
        results = dict(self._engine.evaluate(query_nodes, evidence))
 
        flat_transitions = []
        for t in self._next_state_queries:
            flat_transitions.append((t, results[t]))

        reward = 0.0
        for t in self._reward_queries:
            reward += results[t] * self._utilities[t].value
        
        return flat_transitions, reward
    
    def _cached_eval(self, state, action, cache):
        """
        Return cached evaluation result or compute and store it.
 
        :param state: Evidence assignment for current-state fluents (t = 0).
        :type state: dict[problog.logic.Term, int]
        :param action: Evidence assignment for actions (one-hot).
        :type action: dict[problog.logic.Term, int]
        :param cache: Hashable key for memoization, or None to skip caching.
        :type cache: object or None
        :return: Pair of flat transition list and scalar reward.
        :rtype: tuple[list[tuple[problog.logic.Term, float]], float]
        """
        if cache is None:
            return self._transition_and_reward(state, action)
        result = self._eval_cache.get(cache)
        if result is None:
            result = self._transition_and_reward(state, action)
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
    
