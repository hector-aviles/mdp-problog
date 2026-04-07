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
mdpproblog.darwiche - Efficient marginal evaluation via circuit differentiation
-------------------------------------------------------------------------------

Implements Darwiche's two-pass algorithm for computing all marginals of a
d-DNNF circuit in O(|circuit|) time, independent of the number of queries.

Reference:
    Darwiche, A. (2003). A Differential Approach to Inference in Bayesian
    Networks. Journal of the ACM, 50(3), 280-305. Section 5.

The algorithm proceeds in two phases:
    Phase 1 (bottom-up): compute val(i) for each node i.
    Phase 2 (top-down): compute pd(i) = dF/dV_i for each node i.

For any query atom q, the marginal is then (Darwiche, Corollary 1):
    Pr(q | e) = (w_q * pd(q)) / val(root)

By multilinearity:  F|_{w_q^- = 0} = w_q^+ * dF/dw_q^+
which is exactly what SimpleDDNNFEvaluator.evaluate() computes, but requires
two full circuit traversals per query. The two-pass algorithm amortises this
cost across all queries in O(|circuit|) total.
"""

from problog.constraint import ConstraintAD
from problog.errors import InconsistentEvidenceError


class DarwicheEvaluator:
    """
    Evaluates all marginals over a compiled d-DNNF circuit in a single
    bottom-up / top-down pass (Darwiche, 2003, Section 5).

    The circuit topology is parsed once at construction time. Each call to
    :meth:`evaluate_all` performs exactly two linear traversals of the circuit
    regardless of the number of queries.

    :param formula: compiled d-DNNF circuit (a DDNNF instance)
    :param semiring: semiring for arithmetic operations
    """

    def __init__(self, formula, semiring):
        self._formula = formula
        self._semiring = semiring
        self._n = len(formula)

        # --- Parse circuit topology ---
        # _node_types[i]: 'atom' | 'conj' | 'disj'  for i in 1..n
        # _children[i]:   list of signed child indices (only for conj/disj)
        self._node_types = {}
        self._children = {}

        for i in range(1, self._n + 1):
            node = formula.get_node(i)
            ntype = type(node).__name__
            self._node_types[i] = ntype
            if ntype != 'atom':
                self._children[i] = list(node.children)

        # --- Verify topological order ---
        # ProbLog compiles nodes in topological order (leaves first, root last).
        # Assert |child| < i for every child reference so that [1..n] is a
        # valid topological order for the bottom-up pass.
        for i in range(1, self._n + 1):
            if self._node_types[i] != 'atom':
                for c in self._children[i]:
                    assert abs(c) < i, (
                        "Topological order violated: node %d has child %d. "
                        "ProbLog may have changed its compilation order. "
                        "Build a Kahn-sorted order as fallback." % (i, c)
                    )

        # --- Normalisation flag ---
        # Replicate the condition in SimpleDDNNFEvaluator.evaluate():
        #   normalise if has_evidence() or is_nsp() or
        #              has_constraints(ignore_type={ConstraintAD})
        # In MDP-ProbLog: no ProbLog-level evidence, no NSP, only ConstraintAD
        # → self._normalize is False for standard MDP models.
        self._normalize = any(
            not isinstance(c, ConstraintAD)
            for c in formula.constraints()
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate_all(self, queries, evidence_weights):
        """
        Compute Pr(q | e) for every query simultaneously.

        :param queries: mapping of query terms to their circuit node ids
        :type queries: dict[problog.logic.Term, int]
        :param evidence_weights: evidence as weights (term -> 0 or 1)
        :type evidence_weights: dict[problog.logic.Term, int]
        :return: list of (term, probability) pairs in sorted-by-str order
        :rtype: list[tuple[problog.logic.Term, float]]
        """
        semiring = self._semiring

        # Step 1.  prepare weights (evidence overrides model weights)
        weights = self._formula.extract_weights(semiring, evidence_weights)

        # Step 2.  Phase 1: bottom-up val-messages
        val, val_root = self._bottom_up(weights)

        # Step 3.  consistency check
        if semiring.is_zero(val_root):
            raise InconsistentEvidenceError(
                context=" during DarwicheEvaluator.evaluate_all"
            )

        # Step 4.  Phase 2: top-down pd-messages
        pd_pos, pd_neg = self._top_down(val, weights)

        # Step 5.  extract marginals
        results = []
        for query in sorted(queries, key=str):
            node_id = queries[query]
            prob = self._extract_marginal(node_id, val_root, pd_pos, pd_neg, weights)
            results.append((query, prob))

        return results

    # ------------------------------------------------------------------
    # Phase 1: bottom-up val-messages  (Darwiche 2003, Sec. 5, Figure 7)
    # ------------------------------------------------------------------

    def _bottom_up(self, weights):
        """
        Compute val(i) for every node, traversing leaves to root.

        For atoms:  val(i) = positive weight  w_i^+
        For AND:    val(i) = times over children
        For OR:     val(i) = plus  over children

        Returns (val, val_root) where val_root includes the weights[0] factor
        (the TRUE node weight) to match SimpleDDNNFEvaluator.get_root_weight().
        """
        semiring = self._semiring
        val = {}

        for i in range(1, self._n + 1):
            if self._node_types[i] == 'atom':
                w = weights.get(i)
                val[i] = w[0] if w is not None else semiring.one()
            else:
                child_vals = [
                    self._get_child_val(c, val, weights)
                    for c in self._children[i]
                ]
                if self._node_types[i] == 'conj':
                    v = semiring.one()
                    for cv in child_vals:
                        v = semiring.times(v, cv)
                else:  # 'disj'
                    v = semiring.zero()
                    for cv in child_vals:
                        v = semiring.plus(v, cv)
                val[i] = v

        # Apply the TRUE-node weight factor (weights[0]) — mirrors get_root_weight()
        root_val = val[self._n]
        w0 = weights.get(0)
        if w0 is not None:
            root_val = semiring.times(root_val, w0[0])

        return val, root_val

    # ------------------------------------------------------------------
    # Phase 2: top-down pd-messages  (Darwiche 2003, Sec. 5, Figure 8)
    # ------------------------------------------------------------------

    def _top_down(self, val, weights):
        """
        Compute pd(i) = dF/dV_i for every node, traversing root to leaves.

        Initialisation:  pd(root) = weights[0][0]  (or semiring.one())
        OR  parent → child j:   message = pd(parent)
        AND parent → child j:   message = pd(parent) * product_{siblings} val

        For a DAG (shared nodes) messages from all parents are accumulated
        via semiring.plus (chain rule for multi-parent nodes).

        Returns (pd_pos, pd_neg) where:
            pd_pos[i] = dF/dV_i  for node i referenced positively (+i)
            pd_neg[i] = dF/dV_i  for node i referenced negatively (-i)
        """
        sr = self._semiring
        pd_pos = {i: sr.zero() for i in range(1, self._n + 1)}
        pd_neg = {i: sr.zero() for i in range(1, self._n + 1)}

        # Initialise the root derivative.
        # F = times(val[n], weights[0][0])  →  dF/d val[n] = weights[0][0]
        # If weights[0] is absent:  F = val[n]  →  dF/d val[n] = one()
        w0 = weights.get(0)
        pd_pos[self._n] = w0[0] if w0 is not None else sr.one()

        for i in range(self._n, 0, -1):  # root → leaves
            if self._node_types[i] == 'atom':
                continue  # atoms only receive; they do not distribute further

            pd_i = pd_pos[i]
            children = self._children[i]
            k = len(children)

            if self._node_types[i] == 'disj':
                # dval(OR)/dval(c_j) = 1  →  message to each child = pd(i)
                for c in children:
                    self._accumulate(c, pd_i, pd_pos, pd_neg)

            else:  # 'conj'
                # dval(AND)/dval(c_j) = product of sibling values
                # message to c_j = pd(i) * product_{m != j} val(c_m)
                if k == 1:
                    self._accumulate(children[0], pd_i, pd_pos, pd_neg)

                elif k == 2:
                    v0 = self._get_child_val(children[0], val, weights)
                    v1 = self._get_child_val(children[1], val, weights)
                    self._accumulate(children[0], sr.times(pd_i, v1), pd_pos, pd_neg)
                    self._accumulate(children[1], sr.times(pd_i, v0), pd_pos, pd_neg)

                else:
                    # k >= 3: prefix/suffix products — O(k), no division,
                    # correct when any child value is zero.
                    cvs = [self._get_child_val(c, val, weights) for c in children]

                    prefix = [None] * k
                    prefix[0] = cvs[0]
                    for m in range(1, k):
                        prefix[m] = sr.times(prefix[m - 1], cvs[m])

                    suffix = [None] * k
                    suffix[k - 1] = cvs[k - 1]
                    for m in range(k - 2, -1, -1):
                        suffix[m] = sr.times(suffix[m + 1], cvs[m])

                    for j, c in enumerate(children):
                        if j == 0:
                            sib = suffix[1]
                        elif j == k - 1:
                            sib = prefix[k - 2]
                        else:
                            sib = sr.times(prefix[j - 1], suffix[j + 1])
                        self._accumulate(c, sr.times(pd_i, sib), pd_pos, pd_neg)

        return pd_pos, pd_neg

    # ------------------------------------------------------------------
    # Marginal extraction  (Darwiche 2003, Corollary 1)
    # ------------------------------------------------------------------

    def _extract_marginal(self, node_id, val_root, pd_pos, pd_neg, weights):
        """
        Compute the marginal probability for a single query node.

        By multilinearity (Darwiche 2003, proof of Corollary 1):
            F|_{w_q^- = 0} = w_q^+ * dF/dw_q^+
        so the unnormalised result is  w_q * pd[q].

        This matches SimpleDDNNFEvaluator.get_root_weight() after zeroing w_q^-.
        """
        sr = self._semiring

        if node_id == 0:      # query is deterministically TRUE
            return sr.result(sr.one(), self._formula)
        if node_id is None:   # query is deterministically FALSE
            return sr.result(sr.zero(), self._formula)

        abs_id = abs(node_id)
        w = weights.get(abs_id)

        if node_id > 0:
            w_q = w[0] if w is not None else sr.one()
            numerator = sr.times(w_q, pd_pos[abs_id])
        else:
            w_q = w[1] if w is not None else sr.one()
            numerator = sr.times(w_q, pd_neg[abs_id])

        if self._normalize:
            result = sr.normalize(numerator, val_root)
        else:
            result = numerator

        return sr.result(result, self._formula)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_child_val(self, c, val, weights):
        """
        Return the value of a signed child reference c.

        For atoms (+i or -i): read the appropriate weight directly.
        For internal nodes (+i only, per the NNF property): read val[i].
        """
        abs_c = abs(c)
        if self._node_types[abs_c] == 'atom':
            w = weights.get(abs_c)
            if w is None:
                return self._semiring.one()  # neutral (unlisted) atom
            return w[0] if c > 0 else w[1]
        else:
            assert c > 0, (
                "Negation of an internal node in d-DNNF (node %d): "
                "violates the NNF property." % c
            )
            return val[abs_c]

    def _accumulate(self, c, msg, pd_pos, pd_neg):
        """
        Accumulate a pd-message `msg` into node c (signed).

        Uses semiring.plus so that multiple-parent DAG nodes accumulate
        correctly (sum of incoming messages = chain rule).
        """
        sr = self._semiring
        if c > 0:
            pd_pos[c] = sr.plus(pd_pos[c], msg)
        else:
            pd_neg[-c] = sr.plus(pd_neg[-c], msg)
