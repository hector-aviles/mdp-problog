#! /usr/bin/env python3

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
Tests for the Darwiche two-pass d-DNNF evaluator.

Coverage:
  - Numerical equivalence against SimpleDDNNFEvaluator on three models.
  - Defensive test: non-atomic query node raises ValueError.
  - VI regression: converged policy and values on sysadmin model.
"""

import unittest

from problog.evaluator import SemiringProbability
from problog.logic import Term

from mdpproblog.darwiche import DarwicheDDNNFEvaluator
from mdpproblog.mdp import MDP
from mdpproblog.fluent import StateSpace, ActionSpace
from mdpproblog.value_iteration import ValueIteration


# ── Model definitions ────────────────────────────────────────────────

SYSADMIN_3 = """
computer(c1). computer(c2). computer(c3).
connected(c1,[c2,c3]). connected(c2,[c1]). connected(c3,[c1]).

accTotal([],A,A).
accTotal([_|T],A,X) :- B is A+1, accTotal(T,B,X).
total(L,T) :- accTotal(L,0,T).
total_connected(C,T) :- connected(C,L), total(L,T).

accAlive([],A,A).
accAlive([H|T],A,X) :- running(H,0), B is A+1, accAlive(T,B,X).
accAlive([H|T],A,X) :- not(running(H,0)), B is A, accAlive(T,B,X).
alive(L,A) :- accAlive(L,0,A).
total_running(C,R) :- connected(C,L), alive(L,R).

state_fluent(running(C)) :- computer(C).

action(reboot(none)).
action(reboot(C)) :- computer(C).

1.00::running(C,1) :- reboot(C).
0.05::running(C,1) :- not(reboot(C)), not(running(C,0)).
P::running(C,1)    :- not(reboot(C)), running(C,0),
                      total_connected(C,T), total_running(C,R),
                      P is 0.45+0.50*R/T.

utility(running(C,0),  1.00) :- computer(C).
utility(reboot(C), -0.75) :- computer(C).
utility(reboot(none), 0.00).
"""

SYSADMIN_2 = """
computer(c1). computer(c2).
connected(c1,[c2]). connected(c2,[c1]).

accTotal([],A,A).
accTotal([_|T],A,X) :- B is A+1, accTotal(T,B,X).
total(L,T) :- accTotal(L,0,T).
total_connected(C,T) :- connected(C,L), total(L,T).

accAlive([],A,A).
accAlive([H|T],A,X) :- running(H,0), B is A+1, accAlive(T,B,X).
accAlive([H|T],A,X) :- not(running(H,0)), B is A, accAlive(T,B,X).
alive(L,A) :- accAlive(L,0,A).
total_running(C,R) :- connected(C,L), alive(L,R).

state_fluent(running(C)) :- computer(C).

action(reboot(none)).
action(reboot(C)) :- computer(C).

1.00::running(C,1) :- reboot(C).
0.05::running(C,1) :- not(reboot(C)), not(running(C,0)).
P::running(C,1)    :- not(reboot(C)), running(C,0),
                      total_connected(C,T), total_running(C,R),
                      P is 0.45+0.50*R/T.

utility(running(C,0),  1.00) :- computer(C).
utility(reboot(C), -0.75) :- computer(C).
utility(reboot(none), 0.00).
"""

GRID_2x2 = """
row(1). row(2).
col(1). col(2).

state_fluent(x(X), multivalued) :- row(X).
state_fluent(y(Y), multivalued) :- col(Y).

action(left). action(right). action(up). action(down). action(stay).

utility(goal, 100).
goal :- x(1, 0), y(1, 0), right.
goal :- x(2, 0), y(2, 0), up.

terminal :- x(1, 0), y(2, 0).

1.0::y(Y_new, 1) :- y(Y, 0), right, Y_new is Y + 1, col(Y_new), not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), right, Y_new is Y + 1, not(col(Y_new)), not(terminal).

1.0::y(Y_new, 1) :- y(Y, 0), left, Y_new is Y - 1, col(Y_new), not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), left, first_col(Y), not(terminal).

1.0::y(Y, 1)     :- y(Y, 0), up, not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), down, not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), stay, not(terminal).

1.0::x(X_new, 1) :- x(X, 0), down, X_new is X + 1, row(X_new), not(terminal).
1.0::x(X, 1)     :- x(X, 0), down, X_new is X + 1, not(row(X_new)), not(terminal).

1.0::x(X_new, 1) :- x(X, 0), up, X_new is X - 1, row(X_new), not(terminal).
1.0::x(X, 1)     :- x(X, 0), up, first_row(X), not(terminal).

1.0::x(X, 1)     :- x(X, 0), left, not(terminal).
1.0::x(X, 1)     :- x(X, 0), right, not(terminal).
1.0::x(X, 1)     :- x(X, 0), stay, not(terminal).

1.0::x(X, 1) :- x(X, 0), terminal.
1.0::y(Y, 1) :- y(Y, 0), terminal.

first_col(1).
first_row(1).
"""

TOLERANCE = 1e-10


# ── Test class ────────────────────────────────────────────────────────


class TestDarwicheEquivalence(unittest.TestCase):
    """Numerical equivalence of DarwicheDDNNFEvaluator vs SimpleDDNNFEvaluator."""

    def _compare_evaluators(self, model):
        """Run every (state, action) pair through both evaluators and return max diff."""
        mdp_d = MDP(model, darwiche=True)
        mdp_s = MDP(model, darwiche=False)

        states = StateSpace(mdp_d.state_schema)
        actions = ActionSpace(mdp_d.actions())

        max_diff = 0.0
        n_checks = 0

        for state in states:
            for action in actions:
                evidence = {**state, **action}
                darw = dict(mdp_d._engine.evaluate(mdp_d._compiled_nodes, evidence))
                simp = dict(mdp_s._engine.evaluate(mdp_s._compiled_nodes, evidence))

                for q in darw:
                    diff = abs(darw[q] - simp[q])
                    if diff > max_diff:
                        max_diff = diff
                    n_checks += 1

        return max_diff, n_checks

    def test_sysadmin_3_equivalence(self):
        """Sysadmin 3-computer model: Boolean state fluents."""
        max_diff, n = self._compare_evaluators(SYSADMIN_3)
        self.assertGreater(n, 0)
        self.assertLess(max_diff, TOLERANCE,
                        "Darwiche diverges from Simple on sysadmin-3: %.2e" % max_diff)

    def test_sysadmin_2_equivalence(self):
        """Sysadmin 2-computer model: smaller Boolean model."""
        max_diff, n = self._compare_evaluators(SYSADMIN_2)
        self.assertGreater(n, 0)
        self.assertLess(max_diff, TOLERANCE,
                        "Darwiche diverges from Simple on sysadmin-2: %.2e" % max_diff)

    def test_grid_2x2_equivalence(self):
        """Mitchell 2x2 grid: multivalued state fluents."""
        max_diff, n = self._compare_evaluators(GRID_2x2)
        self.assertGreater(n, 0)
        self.assertLess(max_diff, TOLERANCE,
                        "Darwiche diverges from Simple on grid-2x2: %.2e" % max_diff)


class TestDarwicheNonAtomicQuery(unittest.TestCase):
    """Defensive test: querying a non-atomic node raises ValueError."""

    def test_non_atomic_query_raises(self):
        """evaluate_all_queries rejects non-atom query nodes."""
        mdp = MDP(SYSADMIN_2, darwiche=True)
        formula = mdp._engine._knowledge

        # Find a non-atomic node (conj or disj) in the compiled circuit.
        non_atom_id = None
        for i in range(1, len(formula) + 1):
            node = formula.get_node(i)
            if type(node).__name__ in ('conj', 'disj'):
                non_atom_id = i
                break

        self.assertIsNotNone(non_atom_id,
                             "Could not find a non-atomic node in the circuit")

        # Build a fake query that points to the non-atomic node.
        fake_query = {Term('fake_query'): non_atom_id}

        evaluator = DarwicheDDNNFEvaluator(
            formula, SemiringProbability(), None
        )
        evaluator.propagate()

        with self.assertRaises(ValueError):
            evaluator.evaluate_all_queries(fake_query)


class TestDarwicheVIRegression(unittest.TestCase):
    """VI regression: converged policy and values on sysadmin-3."""

    # Reference values obtained from both Simple and Darwiche evaluators
    # before the refactoring (pre-refactoring snapshot).
    REFERENCE_POLICY = [
        'reboot(c1)', 'reboot(c1)', 'reboot(c1)', 'reboot(c1)',
        'reboot(c3)', 'reboot(c2)', 'reboot(c3)', 'reboot(none)',
    ]

    REFERENCE_V = [
        16.8291913331,
        19.2055288839,
        19.2046264509,
        21.3915199374,
        19.1711351246,
        23.0291765768,
        23.0282696007,
        25.6066576673,
    ]

    def test_vi_darwiche_policy(self):
        """Darwiche VI produces the reference optimal policy."""
        mdp = MDP(SYSADMIN_3, darwiche=True)
        vi = ValueIteration(mdp)
        result = vi.run()

        actual_policy = []
        for s, action_dict in sorted(result.policy.items(), key=str):
            for a, v in action_dict.items():
                if v == 1:
                    actual_policy.append(str(a))
                    break

        self.assertEqual(actual_policy, self.REFERENCE_POLICY)

    def test_vi_darwiche_values(self):
        """Darwiche VI produces the reference V* values."""
        mdp = MDP(SYSADMIN_3, darwiche=True)
        vi = ValueIteration(mdp)
        result = vi.run()

        actual_V = [v for _, v in sorted(result.V.items(), key=str)]

        self.assertEqual(len(actual_V), len(self.REFERENCE_V))
        for actual, expected in zip(actual_V, self.REFERENCE_V):
            self.assertAlmostEqual(actual, expected, places=6,
                                   msg="V* mismatch: got %.10f, expected %.10f"
                                       % (actual, expected))

    def test_vi_darwiche_matches_simple(self):
        """Darwiche VI matches Simple VI exactly."""
        mdp_d = MDP(SYSADMIN_3, darwiche=True)
        mdp_s = MDP(SYSADMIN_3, darwiche=False)

        result_d = ValueIteration(mdp_d).run()
        result_s = ValueIteration(mdp_s).run()

        self.assertEqual(result_d.iterations, result_s.iterations)

        max_diff = max(
            abs(result_d.V[s] - result_s.V[s]) for s in result_d.V
        )
        self.assertLess(max_diff, TOLERANCE,
                        "VI V* diff: %.2e" % max_diff)


if __name__ == '__main__':
    unittest.main(verbosity=2)
