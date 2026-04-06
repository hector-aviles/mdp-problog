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

"""
mdpproblog.csv_exporter - CSV export of MDP structures and Value Iteration results
------------------------------------------------------------------------------------

Provides :class:`CSVExporter`, which serialises transition matrices, reward
functions, value functions, policies, Q-tables, and convergence histories to
CSV files. File I/O is fully isolated from the mathematical engine.

..
    Part of the MDP-ProbLog distribution.
    Licensed under the GNU General Public License v3.
"""

import csv
import logging
import os
from datetime import datetime

from mdpproblog.fluent import StateSpace, ActionSpace
from mdpproblog.util import format_state

logger = logging.getLogger("mdpproblog")


class CSVExporter(object):
    """Handles the formatting and persistence of MDP and Value Iteration data to CSV files.

    Isolates file I/O operations from the core mathematical engine. Each export
    method writes a self-describing CSV file with a metadata header, a column row,
    and one data row per entry.

    :param mdp: The MDP-ProbLog instance containing the state and action schemas.
    :type mdp: mdpproblog.mdp.MDP
    :param output_dir: Path to the directory where CSV files will be saved.
    :type output_dir: str
    """

    _COLS_TRANSITIONS  = ["state_idx", "state_label", "action",
                           "next_state_idx", "next_state_label", "probability"]
    _COLS_REWARDS      = ["state_idx", "state_label", "action", "reward"]
    _COLS_VALUES       = ["state_idx", "state_label", "value"]
    _COLS_POLICY       = ["state_idx", "state_label", "action"]
    _COLS_Q_TABLE      = ["state_idx", "state_label", "action", "q_value"]
    _COLS_CONVERGENCE  = ["iteration", "state_idx", "state_label", "value"]
    _COLS_EVAL_METRICS = ["iteration", "state_index", "action_index", "eval_time_seconds"]

    def __init__(self, mdp, output_dir="output"):
        self.mdp = mdp
        self.output_dir = output_dir
        self._ensure_directory()

    def _ensure_directory(self):
        """Create the output directory if it does not already exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _write_header(self, filepath, description):
        """Write a metadata header to the specified file using CSV comment lines.

        :param filepath: Full path to the target file.
        :type filepath: str
        :param description: A brief description of the file's contents.
        :type description: str
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filepath, mode='w', encoding='utf-8') as f:
            f.write("# MDP-ProbLog Data Export\n")
            f.write(f"# Description: {description}\n")
            f.write(f"# Generated on: {timestamp}\n")

    def _format_state_label(self, item_tuple):
        """Format a state or action tuple as a human-readable string.

        Delegates to :func:`mdpproblog.util.format_state`.

        :param item_tuple: sequence of ``(fluent, value)`` pairs
        :type item_tuple: tuple
        :rtype: str
        """
        return format_state(item_tuple)

    def export_transition_matrix(self, filename="transitions.csv"):
        """Export the full state-action transition probability matrix P(s'|s,a).

        Expands the factored ProbLog distributions into a flat list of edges.
        Only transitions with probability greater than zero are written.

        :param filename: Target filename within the output directory.
        :type filename: str
        """
        filepath = os.path.join(self.output_dir, filename)
        logger.info("Exporting transition matrix \u2192 %s", filepath)
        self._write_header(filepath, "Transition Probability Matrix P(s'|s,a)")

        states = StateSpace(self.mdp.state_schema)
        actions = ActionSpace(self.mdp.actions())
        strides = self.mdp.state_schema.strides

        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self._COLS_TRANSITIONS)

            for i, state in enumerate(states):
                state_lbl = self._format_state_label(tuple(state.items()))
                for j, action in enumerate(actions):
                    action_lbl = self._format_state_label(tuple(action.items()))
                    transition_groups = self.mdp.structured_transition(state, action, (i, j))

                    for next_idx, prob in self._expand_transitions(transition_groups, strides):
                        if prob > 0.0:
                            next_state_lbl = self._format_state_label(tuple(states[next_idx].items()))
                            writer.writerow([
                                i, state_lbl, action_lbl,
                                next_idx, next_state_lbl, f"{prob:.6f}"
                            ])

    def _expand_transitions(self, transition_groups, strides, k=0, current_index=0, joint=1.0):
        """Recursively compute the Cartesian product of independent factored probabilities.

        Reconstructs flat next-state indices from a factored transition representation.

        :param transition_groups: Factored transition as returned by :meth:`mdpproblog.mdp.MDP.structured_transition`.
        :type transition_groups: list
        :param strides: Stride vector for indexing into the flat state space.
        :type strides: list
        :param k: Current recursion depth (factor index).
        :type k: int
        :param current_index: Accumulated flat state index for the current branch.
        :type current_index: int
        :param joint: Accumulated joint probability for the current branch.
        :type joint: float
        :return: Generator of ``(flat_state_index, joint_probability)`` pairs.
        :rtype: generator of tuple(int, float)
        """
        if k == len(transition_groups):
            yield current_index, joint
            return

        factor = transition_groups[k]
        stride = strides[k]

        for term, prob in factor:
            val = self.mdp.state_schema.get_local_index(k, term)
            yield from self._expand_transitions(
                transition_groups, strides,
                k + 1, current_index + val * stride, joint * prob
            )

    def export_reward_matrix(self, filename="rewards.csv"):
        """Export the immediate reward function R(s,a) for all state-action pairs.

        :param filename: Target filename within the output directory.
        :type filename: str
        """
        filepath = os.path.join(self.output_dir, filename)
        logger.info("Exporting reward matrix \u2192 %s", filepath)
        self._write_header(filepath, "Immediate Reward Function R(s,a)")

        states = StateSpace(self.mdp.state_schema)
        actions = ActionSpace(self.mdp.actions())

        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self._COLS_REWARDS)

            for i, state in enumerate(states):
                state_lbl = self._format_state_label(tuple(state.items()))
                for j, action in enumerate(actions):
                    action_lbl = self._format_state_label(tuple(action.items()))
                    reward_val = self.mdp.reward(state, action, (i, j))
                    writer.writerow([i, state_lbl, action_lbl, f"{reward_val:.6f}"])

    def export_value_function(self, vi_result, filename="values.csv"):
        """Export the optimal value function V*(s) to a CSV file.

        :param vi_result: The data transfer object containing the results of Value Iteration.
        :type vi_result: mdpproblog.value_iteration.VIResult
        :param filename: Target filename within the output directory.
        :type filename: str
        """
        filepath = os.path.join(self.output_dir, filename)
        logger.info("Exporting value function \u2192 %s", filepath)
        self._write_header(filepath, "Optimal Value Function V*(s)")

        states = StateSpace(self.mdp.state_schema)

        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self._COLS_VALUES)

            for i, state in enumerate(states):
                state_tuple = tuple(state.items())
                state_lbl = self._format_state_label(state_tuple)
                val = vi_result.V.get(state_tuple, 0.0)
                writer.writerow([i, state_lbl, f"{val:.6f}"])

    def export_policy(self, vi_result, filename="policy.csv"):
        """Export the optimal policy pi*(s) to a CSV file.

        :param vi_result: The data transfer object containing the results of Value Iteration.
        :type vi_result: mdpproblog.value_iteration.VIResult
        :param filename: Target filename within the output directory.
        :type filename: str
        """
        filepath = os.path.join(self.output_dir, filename)
        logger.info("Exporting policy \u2192 %s", filepath)
        self._write_header(filepath, "Optimal Policy pi*(s)")

        states = StateSpace(self.mdp.state_schema)

        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self._COLS_POLICY)

            for i, state in enumerate(states):
                state_tuple = tuple(state.items())
                state_lbl = self._format_state_label(state_tuple)
                action_tuple = vi_result.policy.get(state_tuple)
                action_lbl = next(term for term, val in action_tuple.items() if val == 1)
                writer.writerow([i, state_lbl, action_lbl])

    def export_q_table(self, vi_result, filename="q_values.csv"):
        """Export the action-value function Q*(s,a) if it was computed during Value Iteration.

        This method is a no-op when ``vi_result.Q`` is ``None``.

        :param vi_result: The data transfer object containing the results of Value Iteration.
        :type vi_result: mdpproblog.value_iteration.VIResult
        :param filename: Target filename within the output directory.
        :type filename: str
        """
        if vi_result.Q is None:
            return

        filepath = os.path.join(self.output_dir, filename)
        logger.info("Exporting Q-table \u2192 %s", filepath)
        self._write_header(filepath, "Action-Value Function Q*(s,a)")

        states = StateSpace(self.mdp.state_schema)
        state_to_idx = {tuple(s.items()): i for i, s in enumerate(states)}

        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self._COLS_Q_TABLE)

            for (state_tuple, action_tuple), q_val in vi_result.Q.items():
                idx = state_to_idx.get(state_tuple, "")
                state_lbl = self._format_state_label(state_tuple)
                action_lbl = self._format_state_label(action_tuple)
                writer.writerow([idx, state_lbl, action_lbl, f"{q_val:.6f}"])

    def export_convergence(self, vi_result, filename="convergence.csv"):
        """Export the value function history V_k(s) across all iterations.

        This method is a no-op when ``vi_result.history`` is ``None``.

        :param vi_result: The data transfer object containing the results of Value Iteration.
        :type vi_result: mdpproblog.value_iteration.VIResult
        :param filename: Target filename within the output directory.
        :type filename: str
        """
        if vi_result.history is None:
            return

        filepath = os.path.join(self.output_dir, filename)
        logger.info("Exporting convergence history \u2192 %s", filepath)
        self._write_header(filepath, "Convergence History V_k(s)")

        states = StateSpace(self.mdp.state_schema)
        state_to_idx = {tuple(s.items()): i for i, s in enumerate(states)}

        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self._COLS_CONVERGENCE)

            for k, v_dict in enumerate(vi_result.history):
                for state_tuple, val in v_dict.items():
                    idx = state_to_idx.get(state_tuple, "")
                    state_lbl = self._format_state_label(state_tuple)
                    writer.writerow([k, idx, state_lbl, f"{val:.6f}"])

    def export_all(self, vi_result):
        """Export all available MDP structures and Value Iteration results.

        Automatically skips optional extensions (Q-table, convergence history)
        when they were not computed during Value Iteration.

        :param vi_result: The data transfer object containing the results of Value Iteration.
        :type vi_result: mdpproblog.value_iteration.VIResult
        """
        logger.info("Exporting all MDP data to: %s", self.output_dir)

        self.export_transition_matrix()
        self.export_reward_matrix()
        self.export_value_function(vi_result)
        self.export_policy(vi_result)

        if vi_result.Q is not None:
            self.export_q_table(vi_result)

        if vi_result.history is not None:
            self.export_convergence(vi_result)

    def open_evaluate_metrics(self, filename="evaluate_metrics.csv"):
        """Open a CSV writer for per-``evaluate()`` timing metrics.

        The caller is responsible for closing the returned file handle when done::

            f, writer = exporter.open_evaluate_metrics()
            try:
                writer.writerow([iteration, state_idx, action_idx, elapsed])
            finally:
                f.close()

        :param filename: Target filename within the output directory.
        :type filename: str
        :return: ``(file_handle, csv_writer)`` pair
        :rtype: tuple(io.TextIOWrapper, csv.writer)
        """
        filepath = os.path.join(self.output_dir, filename)
        self._write_header(filepath, "Per-evaluate() timing metrics during Value Iteration")
        f = open(filepath, mode='a', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(self._COLS_EVAL_METRICS)
        return f, writer
