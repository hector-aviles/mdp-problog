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
mdpproblog.darwiche - Efficient d-DNNF evaluation via circuit differentiation
-----------------------------------------------------------------------------

Implements Darwiche's two-pass algorithm (Darwiche 2003, Section 5) for
computing all marginals of a d-DNNF circuit.

Inherits the bottom-up evaluation (weight extraction, evidence handling,
cache management) from :class:`problog.ddnnf_formula.SimpleDDNNFEvaluator`.
Adds a top-down pass that computes partial derivatives for all nodes,
enabling O(1) marginal extraction per query via :meth:`evaluate_all_queries`.

Reference:
    Darwiche, A. (2003). A Differential Approach to Inference in Bayesian
    Networks. Journal of the ACM
"""

from problog.constraint import ConstraintAD
from problog.ddnnf_formula import SimpleDDNNFEvaluator
from problog.errors import InconsistentEvidenceError


class DarwicheDDNNFEvaluator(SimpleDDNNFEvaluator):
    """Evaluator for d-DNNFs using Darwiche's differentiation algorithm.

    Inherits the bottom-up evaluation from :class:`SimpleDDNNFEvaluator`
    (via ``get_root_weight()`` which fills ``cache_intermediate``).
    Adds a top-down pass to compute partial derivatives for all nodes.

    :param formula: compiled d-DNNF circuit.
    :type formula: problog.ddnnf_formula.DDNNF
    :param semiring: semiring for weight arithmetic.
    :type semiring: problog.evaluator.Semiring
    :param weights: external weight overrides (evidence dict).
    """

    def __init__(self, formula, semiring, weights=None, **kwargs):
        super().__init__(formula, semiring, weights, **kwargs)
        self._val_root = None
        self._pd_pos = None
        self._pd_neg = None

    # ------------------------------------------------------------------
    # Batch query (fast-path for engine)
    # ------------------------------------------------------------------

    def evaluate_all_queries(self, queries):
        """Evaluate all queries in O(|circuit|) + O(Q) via two-pass differentiation.

        Requires :meth:`propagate` to have been called first (which
        initialises weights, applies evidence, and fills the cache).

        :param queries: mapping of query terms to compiled node ids.
        :type queries: dict[Term, int]
        :return: list of (term, probability) pairs sorted by str(term).
        :rtype: list[tuple[Term, float]]
        """
        # Validate: all query nodes must be atoms (or 0 / None).
        for q, node_id in queries.items():
            if node_id is not None and node_id != 0:
                node = self.formula.get_node(abs(node_id))
                if type(node).__name__ != 'atom':
                    raise ValueError(
                        "Query '%s' maps to non-atomic node %d (type: %s). "
                        "Darwiche two-pass marginals are only valid for "
                        "indicator variables (atoms)."
                        % (q, node_id, type(node).__name__)
                    )

        # Phase 1: bottom-up reads from cache_intermediate
        self._val_root = self.get_root_weight()

        if self.semiring.is_zero(self._val_root):
            raise InconsistentEvidenceError(
                context=" during two-pass evaluation"
            )

        # Phase 2: top-down compute partial derivatives
        self._compute_pd()

        # Phase 3: O(1) extraction per query
        return [
            (q, self._extract_marginal(queries[q]))
            for q in sorted(queries, key=str)
        ]

    # ------------------------------------------------------------------
    # Phase 2: top-down
    # ------------------------------------------------------------------

    def _compute_pd(self):
        """Compute pd(i) = dF/dV_i for every node via reverse topological traversal.

        pd-messages propagate root -> leaves:
          Root:            pd(root) = weights[0][0]  (TRUE-node factor)
          Addition parent: mes(i->j) = pd(i)
          Multiply parent: mes(i->j) = pd(i) * product_{k!=j} val(c_k)

        For DAG nodes with multiple parents, pd accumulates via semiring.plus.
        """
        sr = self.semiring
        n = len(self.formula)

        pd_pos = [sr.zero()] * (n + 1)
        pd_neg = [sr.zero()] * (n + 1)

        # Seed: dF/d val(root) = weights[0][0] if present, else one().
        # Because F = val(root) * weights[0][0].
        w0 = self.weights.get(0)
        pd_pos[n] = w0[0] if w0 is not None else sr.one()

        for i in range(n, 0, -1):
            node = self.formula.get_node(i)
            ntype = type(node).__name__
            if ntype == 'atom':
                continue

            pd_i = pd_pos[i]
            kids = node.children
            k = len(kids)

            if ntype == 'disj':
                for c in kids:
                    self._accumulate(c, pd_i, pd_pos, pd_neg)
            else:  # conj
                if k == 0:
                    continue
                elif k == 1:
                    self._accumulate(kids[0], pd_i, pd_pos, pd_neg)
                elif k == 2:
                    v0 = self._get_weight(kids[0])
                    v1 = self._get_weight(kids[1])
                    self._accumulate(
                        kids[0], sr.times(pd_i, v1), pd_pos, pd_neg
                    )
                    self._accumulate(
                        kids[1], sr.times(pd_i, v0), pd_pos, pd_neg
                    )
                else:
                    self._conj_distribute(pd_i, kids, pd_pos, pd_neg)

        self._pd_pos = pd_pos
        self._pd_neg = pd_neg

    def _conj_distribute(self, pd_i, kids, pd_pos, pd_neg):
        """Distribute pd-messages through an AND node with k >= 3 children.

        Uses prefix/suffix products to compute each sibling product in O(k)
        total without division — correct even when a child value is zero.
        """
        sr = self.semiring
        k = len(kids)
        cvs = [self._get_weight(c) for c in kids]

        prefix = [None] * k
        prefix[0] = cvs[0]
        for m in range(1, k):
            prefix[m] = sr.times(prefix[m - 1], cvs[m])

        suffix = [None] * k
        suffix[-1] = cvs[-1]
        for m in range(k - 2, -1, -1):
            suffix[m] = sr.times(suffix[m + 1], cvs[m])

        for j, c in enumerate(kids):
            if j == 0:
                sib = suffix[1]
            elif j == k - 1:
                sib = prefix[k - 2]
            else:
                sib = sr.times(prefix[j - 1], suffix[j + 1])
            self._accumulate(c, sr.times(pd_i, sib), pd_pos, pd_neg)

    # ------------------------------------------------------------------
    # Marginal extraction
    # ------------------------------------------------------------------

    def _extract_marginal(self, node_id):
        """Compute the marginal for a single query node from precomputed pd/val.

        For query atom q with indicator lambda_q:
          Pr(q | e) = w_q * pd(q) / F(e)   (when normalisation applies)

        :param node_id: compiled node index (positive, negative, 0, or None).
        :return: marginal as semiring result value.
        """
        sr = self.semiring

        if node_id == 0:
            if not sr.is_nsp():
                return sr.result(sr.one(), self.formula)
            result = sr.normalize(self._val_root, self._val_root)
            return sr.result(result, self.formula)

        if node_id is None:
            return sr.result(sr.zero(), self.formula)

        abs_id = abs(node_id)
        w = self.weights.get(abs_id)

        if node_id > 0:
            w_q = w[0] if w is not None else sr.one()
            numerator = sr.times(w_q, self._pd_pos[abs_id])
        else:
            w_q = w[1] if w is not None else sr.one()
            numerator = sr.times(w_q, self._pd_neg[abs_id])

        if self._should_normalize():
            result = sr.normalize(numerator, self._val_root)
        else:
            result = numerator

        return sr.result(result, self.formula)

    def _should_normalize(self):
        """Check whether marginals require normalisation.

        Replicates the condition from ``SimpleDDNNFEvaluator.evaluate()``:
        normalise if evidence is active, the semiring is NSP, or there
        are non-AD constraints.
        """
        return (
            self.has_evidence()
            or self.semiring.is_nsp()
            or self.has_constraints(ignore_type={ConstraintAD})
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _accumulate(self, c, msg, pd_pos, pd_neg):
        """Accumulate a pd-message into the target node.

        :param c: signed child index.
        :param msg: pd-message to accumulate.
        :param pd_pos: positive pd array (mutated).
        :param pd_neg: negative pd array (mutated).
        """
        if c > 0:
            pd_pos[c] = self.semiring.plus(pd_pos[c], msg)
        else:
            pd_neg[-c] = self.semiring.plus(pd_neg[-c], msg)
