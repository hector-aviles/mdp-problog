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

import random
from collections import OrderedDict
from mdpproblog.fluent import StateSpace, ActionSpace

class Simulator(object):
    """
    Simulator class for MDPs. Given an `mdp` and a `policy`,
    it generates histories and its corresponding
    expected cummulative discounted rewards.

    :param mdp: an MDP formulation
    :type mdp: mdpproblog.mdp.MDP object
    :param policy: mapping from state to action
    :type policy: dict of (tuple, OrderedDict)
    """

    def __init__(self, mdp, policy):
        self._mdp = mdp
        self._policy = policy

        self._schema = mdp.state_schema
        self._state_space = StateSpace(self._schema)
        self._action_space = ActionSpace(mdp.actions())
        self._current_state_factors = self._schema.get_factors_at(0) 

    def run(self, trials, horizon, start_state, gamma=0.9):
        """
        Simulate a number of `trials` of given `horizon` from `start_state`
        following its policy. Compute the discounted expected reward using
        `gamma` as discount factor. Return average reward over all trials,
        a list of rewards received at each trial and list of sampled states
        for each trial.
 
        :param trials: number of trials
        :type trials: int
        :param horizon: number of timesteps
        :type horizon: int
        :param start_state: state from which the simulation starts
        :param gamma: discount factor
        :type gamma: float
        :rtype: tuple (float, list of floats, list of list)
        """
        rewards = []
        paths = []
        for i in range(trials):
            total_reward, trial_path = self.run_trial(horizon, start_state, gamma)
            rewards.append(total_reward)
            paths.append(trial_path)
        avg = sum(rewards) / trials
        return avg, rewards, paths

    def run_trial(self, horizon, start_state, gamma=0.9):
        """
        Simulate a single trial of given `horizon` from `start_state`
        following its policy. Compute the discounted expected reward using
        `gamma` as discount factor. Return total discounted reward over all
        steps of the horizon and a list of sampled states in the trial.
 
        :param horizon: number of timesteps
        :type horizon: int
        :param start_state: state from which the simulation starts
        :param gamma: discount factor
        :type gamma: float
        :rtype: tuple (float, list)
        """
        state = start_state
        discount = 1.0
        total = 0.0
        path = [start_state]
        for step in range(horizon):
            action_val = self.__select_action(state)
 
            state_val = OrderedDict(state)
            cache = (self._state_space.index(state_val),
                     self._action_space.index(action_val))
 
            reward = self.__collect_reward(state_val, action_val, cache)
            state = self.__sample_next_state(state_val, action_val, cache)
 
            total += discount * reward
            path.extend([action_val, state])
            discount *= gamma
        return total, path

    def __select_action(self, state):
        """
        Return the action prescribed by the policy for the given `state`.
 
        :param state: state represented as a tuple of (Term, int) pairs
        :rtype: OrderedDict
        """
        return self._policy[state]

    def __collect_reward(self, state_val, action_val, cache):
        """
        Return the reward for applying `action_val` in `state_val`.
 
        :param state_val: state as OrderedDict evidence mapping
        :param action_val: action as one-hot OrderedDict evidence mapping
        :param cache: (state_index, action_index) for memoisation
        :rtype: float
        """
        return self._mdp.reward(state_val, action_val, cache)

    def __sample_next_state(self, state_val, action_val, cache):
        """
        Return next state sampled from the factored transition distribution
        given by applying `action_val` to `state_val`.
        Performs weighted categorical sampling per schema factor,
        handling both boolean and multivalued fluents.
 
        :param state_val: state as OrderedDict evidence mapping
        :param action_val: action as one-hot OrderedDict evidence mapping
        :param cache: (state_index, action_index) for memoisation
        :rtype: tuple of (Term, int) pairs
        """
        structured = self._mdp.structured_transition(state_val, action_val, cache)
        new_valuation = OrderedDict()
 
        for f_idx, group in enumerate(structured):
            factor_terms = self._current_state_factors[f_idx]
 
            if not group:
                for term in factor_terms:
                    new_valuation[term] = 0
                continue
 
            weights = [prob for _, prob in group]
            chosen_idx = random.choices(range(len(group)), weights=weights)[0]
            chosen_term = group[chosen_idx][0]
 
            local_idx = self._schema.get_local_index(f_idx, chosen_term)
 
            if len(factor_terms) == 1:
                new_valuation[factor_terms[0]] = local_idx
            else:
                for i, term in enumerate(factor_terms):
                    new_valuation[term] = 1 if i == local_idx else 0
 
        return tuple(new_valuation.items())
