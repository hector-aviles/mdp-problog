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
    :param backend: ProbLog compilation backend. Use None for the default (d-DNNF).
    :type backend: str or None
    """

    def __init__(self, model, epsilon_thr=1e-6, backend=None):
        self._model = model
        self.epsilon_thr = epsilon_thr
        self.backend = backend

        self._engine = eng.Engine(model)

        self.__transition_cache = {}
        self.__reward_cache = {}

        self.__prepare()

    def __prepare(self):
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
        self.__utilities = self._engine.assignments('utility')

        # ground only what is relevant to rewards, transitions and evidence.
        current_state_fluents = self.current_state_fluents()
        next_state_fluents = self.next_state_fluents()
        queries = list(set(self.__utilities) | set(next_state_fluents) | set(actions) | set(current_state_fluents))
        self._engine.relevant_ground(queries)

        # compile query database
        self.__next_state_queries = self._engine.compile(next_state_fluents)
        self.__reward_queries = self._engine.compile(self.__utilities)

    def state_fluents(self):
        """
        Return the ordered list of atemporal state fluent terms.

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

        This method converts the flat list returned by :meth:`transition` into a
        factorized representation that matches :attr:`state_schema`.

        Behavior per factor:

            * Boolean factor (single term): injects the implicit false branch
              (represented as ``None``) with probability ``1 - p_true``.
            * Multi-valued factor (AD group): returns only terms with probability
              greater than :attr:`epsilon_thr` (sparse filter).

        :param state: Evidence assignment for current-state fluents (t = 0).
        :type state: dict[problog.logic.Term, int]
        :param action: Evidence assignment for actions (one-hot).
        :type action: dict[problog.logic.Term, int]
        :param cache: Optional cache key shared with :meth:`transition` and :meth:`reward`.
        :type cache: object or None
        :return: List of factors, each factor is a list of ``(term, probability)`` pairs.
            For boolean factors, ``term`` may be None to denote the false branch.
        :rtype: list[list[tuple[problog.logic.Term | None, float]]]
        """
        flat_transitions = self.transition(state, action, cache)
        prob_map = {str(term): prob for term, prob in flat_transitions}
        
        structured_result = []
        
        for factor_template in self._next_state_factors:
            group_data = []
            
           # Boolean factor: 1 term -> {false, true}
            if len(factor_template) == 1:
                term = factor_template[0]
                p_true = prob_map.get(str(term), 0.0)
                p_false = 1.0 - p_true
                
                if p_false > self.epsilon_thr:
                    group_data.append((None, p_false))
                if p_true > self.epsilon_thr:
                    group_data.append((term, p_true))
            
            # Multi-valued factor (ADS group): keep only non-negligible branches
            else:
                for term in factor_template:
                    p = prob_map.get(str(term), 0.0)
                    if p > self.epsilon_thr:
                        group_data.append((term, p))
                        
            structured_result.append(group_data)
            
        return structured_result

    def transition(self, state, action, cache=None):
        """
        Return probabilities for all next-state fluents given `state` and `action.

        If ``cache`` is provided, results are memoized and subsequent calls with
        the same cache key will not re-evaluate the ProbLog circuit.

        :param state: state vector representation of current state fluents
        :type state: list of 0/1 according to state fluents order
        :param action: action vector representation
        :type action: one-hot vector encoding of action as a list of 0/1
        :param cache: key to cache results
        :type cache: immutable, hashable object
        :rtype: list of pairs (problog.logic.Term, float)
        """
        if cache is None:
            return self.__transition(state, action)

        transition = self.__transition_cache.get(cache, None)
        if transition is None:
            transition = self.__transition(state, action)
            self.__transition_cache[cache] = transition
        return transition

    def __transition(self, state, action):
        """
        Return the probabilities of next state fluents given current
        `state` and `action`.

        :param state: state vector representation of current state fluents
        :type state: list of 0/1 according to state fluents order
        :param action: action vector representation
        :type action: one-hot vector encoding of action as a list of 0/1
        :rtype: list of pairs (problog.logic.Term, float)
        """
        evidence = state.copy()
        evidence.update(action)
        return self._engine.evaluate(self.__next_state_queries, evidence)

    def transition_model(self):
        """
        Build the full transition model P(s'|s,a) by enumeration.

        This is primarily a debugging/inspection helper. It enumerates every
        state and every action and calls :meth:`transition`.

        :return: Mapping from ``(state_values, action_values)`` to next-state probabilities.
        :rtype: dict of ((tuple, tuple), list)
        """
        transitions = {}
        states  = StateSpace(self.current_state_fluents())
        actions = ActionSpace(self.actions())
        for state in states:
            for action in actions:
                probabilities = self.transition(state, action)
                transitions[(tuple(state.values()), tuple(action.values()))] = probabilities
        return transitions

    def reward(self, state, action, cache=None):
        """
        Return the expected immediate reward for executing an action in a state.

        If ``cache`` is provided, results are memoized and subsequent calls with
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
        if cache is None:
            return self.__reward(state, action)

        value = self.__reward_cache.get(cache, None)
        if value is None:
            value = self.__reward(state, action)
            self.__reward_cache[cache] = value
        return value

    def __reward(self, state, action):
        """
        Return the immediate reward value of the transition
        induced by applying `action` to the given `state`.

        :param state: state vector representation of current state fluents (t = 0).
        :type state: list of 0/1 according to state fluents order
        :param action: action vector representation
        :type action: one-hot vector encoding of action as a list of 0/1
        :rtype: float
        """
        evidence = state.copy()
        evidence.update(action)
        total = 0
        for term, prob in self._engine.evaluate(self.__reward_queries, evidence):
            total += prob * self.__utilities[term].value
        return total

    def reward_model(self):
        """
        Return the reward model of all valid transitions.

        :rtype: dict of ((state,action), float)
        """
        rewards = {}
        states  = StateSpace(self.current_state_fluents())
        actions = ActionSpace(self.actions())
        for state in states:
            for action in actions:
                reward = self.reward(state, action)
                rewards[(tuple(state.values()), tuple(action.values()))] = reward
        return rewards
