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
import sys

from mdpproblog.fluent import StateSpace, ActionSpace
from mdpproblog.util import Timer

logger = logging.getLogger("mdpproblog")


class ValueIteration(object):
    """
    Implementation of the enumerative Value Iteration algorithm.
    It performs successive, synchronous Bellman backups until
    convergence is achieved for the given error epsilon for the
    infinite-horizon MDP with discount factor gamma.

    :param mdp: MDP representation
    :type mdp: mdpproblog.MDP
    """

    def __init__(self, mdp):
        self._mdp = mdp

    def run(self, gamma=0.9, epsilon=0.1):
        """
        Execute value iteration until convergence.
        Return optimal value function, greedy policy and number
        of iterations.

        :param gamma: discount factor
        :type gamma: float
        :param epsilon: maximum error
        :type epsilon: float
        :rtype: triple (dict(state, value), dict(policy, action), float)
        """
        V = {}
        policy = {}

        states = StateSpace(self._mdp.state_schema)
        actions = ActionSpace(self._mdp.actions())
        strides = self._mdp.state_schema.strides

        iteration = 0

        with Timer("ValueIteration"):
            while True:
                iteration += 1
                max_residual = -sys.maxsize
                for (i, state) in enumerate(states):
                    max_value = -sys.maxsize
                    greedy_action = None
                    for (j, action) in enumerate(actions):
                        transition_groups = self._mdp.structured_transition(state, action, (i, j))
                        reward = self._mdp.reward(state, action, (i, j))
                        Q = reward + gamma * self._expected_value(transition_groups, strides, V)
                        if Q >= max_value:
                            max_value = Q
                            greedy_action = actions[j]

                    residual = abs(V.get(i, 0) - max_value)
                    max_residual = max(max_residual, residual)
                    V[i] = max_value
                    policy[i] = greedy_action

                if max_residual <= 2 * epsilon * (1 - gamma) / gamma:
                    break

        V = { tuple(states[i].items()): value for i, value in V.items() }
        policy = { tuple(states[i].items()): action for i, action in policy.items() }

        return V, policy, iteration

    def _expected_value(self, transition_groups, strides, V, k=0, current_index=0, joint=1.0):
        """
        Compute the expected future value for a probabilistic transition.

        :param transition_groups: List of factors, where each factor is a list of ``(term, prob)`` pairs.
        :type transition_groups: list
        :param V: Current value function mapping integer state index to value.
        :type V: dict
        :param k: Recursion depth (index of the current factor being processed).
        :type k: int
        :param index: Accumulated integer state index for the current branch.
        :type index: int
        :param joint: Accumulated joint probability of the current branch.
        :type joint: float
        :rtype: float
        """

        if len(transition_groups) == k:
            return joint * V.get(current_index, 0.0)

        factor = transition_groups[k]
        stride = strides[k]
        expected_sum = 0.0
        
        for term, prob in factor:
            val = self._mdp.state_schema.get_local_index(k, term) 
            expected_sum += self._expected_value(transition_groups, strides, V, k + 1, current_index + val * stride, joint * prob)

        return expected_sum