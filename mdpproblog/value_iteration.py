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
import time
from dataclasses import dataclass, field

from mdpproblog.fluent import StateSpace, ActionSpace
from mdpproblog.util import Timer, TRACE

logger = logging.getLogger("mdpproblog")


@dataclass
class VIResult:
    """Data transfer object holding the output of Value Iteration.

    :param V: Optimal value function mapping state tuples to scalar values.
    :type V: dict
    :param policy: Optimal policy mapping state tuples to action dicts.
    :type policy: dict
    :param iterations: Number of Bellman backup iterations until convergence.
    :type iterations: int
    :param Q: Action-value function; ``None`` if not tracked.
    :type Q: dict or None
    :param history: Per-iteration value snapshots; ``None`` if not tracked.
    :type history: list or None
    """
    V: dict
    policy: dict
    iterations: int
    Q: dict = field(default=None)
    history: list = field(default=None)


class ValueIteration(object):
    """
    Implementation of the enumerative Value Iteration algorithm.
    It performs successive in-place Bellman backups (each state update
    is immediately visible to the states that follow it in the same sweep)
    until convergence is achieved for the given error epsilon for the
    infinite-horizon MDP with discount factor gamma.

    :param mdp: MDP representation
    :type mdp: mdpproblog.MDP
    """

    def __init__(self, mdp):
        self._mdp = mdp

    def run(self, gamma=0.9, epsilon=0.1, track_history=False, track_q=False):
        """Execute value iteration until convergence.

        :param gamma: discount factor
        :type gamma: float
        :param epsilon: maximum error bound for convergence
        :type epsilon: float
        :param track_history: if ``True``, record a snapshot of V after each full
            sweep over the state space. Stored in :attr:`VIResult.history` as a
            list of ``{state_tuple: value}`` dicts, one per iteration.
        :type track_history: bool
        :param track_q: if ``True``, compute Q*(s,a) for all state-action pairs after
            convergence using the final value function. Stored in :attr:`VIResult.Q`
            as a ``{(state_tuple, action_tuple): q_value}`` dict.
        :type track_q: bool
        :rtype: VIResult
        """
        V = {}
        policy = {}

        states  = StateSpace(self._mdp.state_schema)
        actions = ActionSpace(self._mdp.actions())
        strides = self._mdp.state_schema.strides

        num_states = len(states)

        iteration      = 0
        last_milestone = 0
        history        = [] if track_history else None
        threshold      = 2 * epsilon * (1 - gamma) / gamma
        trace          = logger.isEnabledFor(TRACE)

        with Timer("ValueIteration"):
            while True:
                iteration    += 1
                max_residual  = -sys.maxsize
                t_start       = time.perf_counter()

                for (i, state) in enumerate(states):
                    max_value     = -sys.maxsize
                    greedy_action = None
                    for (j, action) in enumerate(actions):
                        transition_groups = self._mdp.structured_transition(state, action, (i, j))
                        reward = self._mdp.reward(state, action, (i, j))
                        Q = reward + gamma * self._expected_value(transition_groups, strides, V)
                        if Q >= max_value:
                            max_value     = Q
                            greedy_action = actions[j]

                    if iteration == 1:
                        last_milestone = self._log_first_sweep_progress(
                            i + 1, num_states, last_milestone
                        )

                    residual     = abs(V.get(i, 0) - max_value)
                    max_residual = max(max_residual, residual)
                    V[i]      = max_value
                    policy[i] = greedy_action

                if track_history:
                    snapshot = {tuple(states[i].items()): V[i] for i in range(len(states))}
                    history.append(snapshot)

                if trace:
                    elapsed = time.perf_counter() - t_start
                    logger.log(TRACE, "Iteration %d: residual=%.6f  elapsed=%.4fs",
                               iteration, max_residual, elapsed)

                if max_residual <= threshold:
                    if trace:
                        logger.log(TRACE, "Converged at iteration %d (threshold=%f)",
                                   iteration, threshold)
                    break

        Q = None
        if track_q:
            Q = {}
            for i, state in enumerate(states):
                state_tuple = tuple(state.items())
                for j, action in enumerate(actions):
                    action_tuple = tuple(action.items())
                    transition_groups = self._mdp.structured_transition(state, action, (i, j))
                    reward = self._mdp.reward(state, action, (i, j))
                    q_val = reward + gamma * self._expected_value(transition_groups, strides, V)
                    Q[(state_tuple, action_tuple)] = q_val

        V = { tuple(states[i].items()): value for i, value in V.items() }
        policy = { tuple(states[i].items()): action for i, action in policy.items() }

        return VIResult(V=V, policy=policy, iterations=iteration, Q=Q, history=history)

    def _log_first_sweep_progress(self, states_done, num_states, last_milestone):
        """
        Log the progress of the first sweep every 10% of the state space.

        :return: The highest milestone (in tenths) logged so far.
        :rtype: int
        """
        milestone = states_done * 10 // num_states
        if milestone > last_milestone:
            logger.debug(
                "ValueIteration (iteration 1): %d%% (%d/%d states)",
                milestone * 10, states_done, num_states,
            )
        return max(milestone, last_milestone)

    def _expected_value(self, transition_groups, strides, V, k=0, current_index=0, joint=1.0):
        """
        Compute the expected future value for a factored transition.

        :param transition_groups: One tuple of ``(local_index, probability)``
            pairs per factor, as returned by :meth:`MDP.structured_transition`.
        :type transition_groups: tuple
        :param strides: Positional stride of each factor in the state index.
        :type strides: list of int
        :param V: Current value function mapping integer state index to value.
        :type V: dict
        :param k: Recursion depth (index of the current factor being processed).
        :type k: int
        :param current_index: Accumulated integer state index for the current branch.
        :type current_index: int
        :param joint: Accumulated joint probability of the current branch.
        :type joint: float
        :return: Expected value of V over the successor states of this branch.
        :rtype: float
        """

        if len(transition_groups) == k:
            return joint * V.get(current_index, 0.0)

        factor = transition_groups[k]
        stride = strides[k]
        expected_sum = 0.0
        
        for local_idx, prob in factor:
            expected_sum += self._expected_value(transition_groups, strides, V, k + 1, current_index + local_idx * stride, joint * prob)
 
        return expected_sum