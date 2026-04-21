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
computing all marginals of a d-DNNF circuit in O(|circuit|) time.

The module provides two components:

- :class:`DDNNFTopology`: immutable cache of the circuit's DAG structure,
  computed once per compiled formula.
- :class:`DarwicheDDNNFEvaluator`: a :class:`problog.evaluator.Evaluator`
  subclass that uses the cached topology to evaluate queries via two linear
  traversals (bottom-up val-messages, top-down pd-messages).

Reference:
    Darwiche, A. (2003). A Differential Approach to Inference in Bayesian
    Networks. *Journal of the ACM*, 50(3), 280--305.
"""

from problog.constraint import ConstraintAD
from problog.errors import InconsistentEvidenceError
from problog.evaluator import Evaluator


class DDNNFTopology(object):
    """Immutable structural cache of a compiled d-DNNF circuit.

    Parses node types and children once, verifies topological order,
    and precomputes the normalisation flag.  Intended to be stored on
    the ``DDNNF`` instance and shared across evaluator lifetimes.

    :param formula: compiled d-DNNF circuit.
    :type formula: problog.ddnnf_formula.DDNNF
    """

    __slots__ = ('n', 'node_types', 'children', 'normalize')

    def __init__(self, formula):
        n = len(formula)
        self.n = n

        # Pre-allocate 1-indexed lists (index 0 unused).
        node_types = [None] * (n + 1)
        children = [None] * (n + 1)

        for i in range(1, n + 1):
            node = formula.get_node(i)
            ntype = type(node).__name__
            node_types[i] = ntype
            if ntype != 'atom':
                kids = tuple(node.children)
                children[i] = kids
                # Verify topological order: every child index < parent index.
                # ProbLog compilers (dsharp, c2d) guarantee this.
                for c in kids:
                    if abs(c) >= i:
                        raise ValueError(
                            "Topological order violated: node %d references "
                            "child %d (expected |child| < parent)." % (i, c)
                        )

        self.node_types = node_types
        self.children = children

        # Normalise iff there are non-AD constraints.
        # Mirrors the condition in SimpleDDNNFEvaluator.evaluate():
        #   has_evidence() or is_nsp() or has_constraints(ignore={ConstraintAD})
        # The first two are evaluated at query time; this flag covers the third.
        self.normalize = any(
            type(c) is not ConstraintAD for c in formula.constraints()
        )


class DarwicheDDNNFEvaluator(Evaluator):
    """Evaluator for d-DNNFs using Darwiche's differentiation algorithm.

    After :meth:`propagate`, every :meth:`evaluate` call is O(1) —
    the marginal is read from precomputed partial derivatives.

    :param formula: compiled d-DNNF circuit.
    :type formula: problog.ddnnf_formula.DDNNF
    :param semiring: semiring for weight arithmetic.
    :type semiring: problog.evaluator.Semiring
    :param weights: external weight overrides (evidence dict).
    :param topology: precomputed circuit topology.
    :type topology: DDNNFTopology
    """

    def __init__(self, formula, semiring, weights, topology, **kwargs):
        Evaluator.__init__(self, formula, semiring, weights, **kwargs)
        self._topo = topology

        # Populated by _run_two_pass() during propagate().
        self._val = None
        self._val_root = None
        self._pd_pos = None
        self._pd_neg = None

    # ------------------------------------------------------------------
    # Evaluator interface
    # ------------------------------------------------------------------

    def propagate(self):
        """Initialize weights, apply evidence, and run the two-pass algorithm."""
        self._initialize()
        self._run_two_pass()

    def evaluate(self, node):
        """Compute the marginal probability of a single query node.

        :param node: node index (positive or negative), 0 for TRUE, None for FALSE.
        :return: marginal probability as semiring result value.
        """
        return self._extract_marginal(node)

    def evaluate_fact(self, node):
        """Evaluate fact.

        :param node: fact to evaluate.
        :return: weight of the fact (as semiring result value).
        """
        return self.evaluate(node)

    def set_weight(self, index, pos, neg):
        """Set weight of a node.

        :param index: index of node.
        :param pos: positive weight (semiring internal value).
        :param neg: negative weight (semiring internal value).
        """
        self.weights[index] = (pos, neg)

    def set_evidence(self, index, value):
        """Set value for evidence node.

        :param index: index of evidence node.
        :param value: True if positive evidence, False otherwise.
        """
        curr_pos, curr_neg = self.weights.get(index)
        pos, neg = self.semiring.to_evidence(curr_pos, curr_neg, sign=value)

        if (value and self.semiring.is_zero(curr_pos)) or (
            not value and self.semiring.is_zero(curr_neg)
        ):
            raise InconsistentEvidenceError(self.formula.get_node(index).name)

        self.set_weight(index, pos, neg)

    def get_root_weight(self):
        """Get the WMC of the root node.

        :return: WMC of the circuit root, including the TRUE-node weight factor.
        """
        return self._val_root

    def has_constraints(self, ignore_type=None):
        """Check whether the formula has constraints not in *ignore_type*.

        :param ignore_type: constraint classes to ignore.
        :type ignore_type: set or None
        """
        ignore_type = ignore_type or set()
        return any(
            type(c) not in ignore_type for c in self.formula.constraints()
        )

    # ------------------------------------------------------------------
    # Batch query (fast-path for engine)
    # ------------------------------------------------------------------

    def evaluate_all_queries(self, queries):
        """Evaluate all queries in O(Q) after propagation.

        :param queries: mapping of query terms to compiled node ids.
        :type queries: dict[Term, int]
        :return: list of (term, probability) pairs sorted by str(term).
        :rtype: list[tuple[Term, float]]
        """
        return [
            (q, self._extract_marginal(queries[q]))
            for q in sorted(queries, key=str)
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _initialize(self):
        """Extract model weights and apply ProbLog-level evidence.

        Mirrors ``SimpleDDNNFEvaluator._initialize(with_evidence=True)``.
        """
        self.weights.clear()
        self.weights = self.formula.extract_weights(
            self.semiring, self.given_weights
        ).copy()

        for ev in self.evidence():
            self.set_evidence(abs(ev), ev > 0)

    def _run_two_pass(self):
        """Execute Darwiche's two-pass message scheme (Section 5).

        Phase 1 (bottom-up): val-messages set val(i) for every node.
        Phase 2 (top-down):  pd-messages set pd(i) = dF/dV_i for every node.

        After this method, ``_extract_marginal`` can answer any query in O(1).
        """
        self._val, self._val_root = self._bottom_up()

        if self.semiring.is_zero(self._val_root):
            raise InconsistentEvidenceError(
                context=" during DarwicheEvaluator two-pass evaluation"
            )

        self._pd_pos, self._pd_neg = self._top_down()

    # ------------------------------------------------------------------
    # Phase 1: bottom-up 
    # ------------------------------------------------------------------
    #
    # val-messages propagate leaf → root:
    #   Leaf (indicator/parameter): val(l) = weight of l
    #   Addition  (+) node:         val(i) = Σ_j val(c_j)
    #   Multiply  (×) node:         val(i) = Π_j val(c_j)
    #
    # In ProbLog's d-DNNF the addition nodes are OR (disj) and the
    # multiplication nodes are AND (conj).  Leaf atoms without an
    # explicit weight entry receive semiring.one() (neutral element).
    # ------------------------------------------------------------------

    def _bottom_up(self):
        """Compute val(i) for every node i in topological order.

        :return: (val, val_root) where val_root includes the TRUE-node factor.
        :rtype: tuple[list, any]
        """
        sr = self.semiring
        topo = self._topo
        n = topo.n
        weights = self.weights
        node_types = topo.node_types
        children = topo.children

        val = [None] * (n + 1)

        for i in range(1, n + 1):
            ntype = node_types[i]
            if ntype == 'atom':
                w = weights.get(i)
                val[i] = w[0] if w is not None else sr.one()
            elif ntype == 'conj':
                v = sr.one()
                for c in children[i]:
                    v = sr.times(v, self._child_val(c, val))
                val[i] = v
            else:  # 'disj'
                v = sr.zero()
                for c in children[i]:
                    v = sr.plus(v, self._child_val(c, val))
                val[i] = v

        # Apply TRUE-node weight (mirrors SimpleDDNNFEvaluator.get_root_weight).
        root = val[n]
        w0 = weights.get(0)
        if w0 is not None:
            root = sr.times(root, w0[0])

        return val, root

    # ------------------------------------------------------------------
    # Phase 2: top-down 
    # ------------------------------------------------------------------
    #
    # pd-messages propagate root → leaves:
    #   Root:            pd(root) = 1   (adjusted by TRUE-node weight)
    #   Addition parent: mes(i→j) = pd(i)              for each child j
    #   Multiply parent: mes(i→j) = pd(i) × Π_{k≠j} val(c_k)
    #
    # For DAG nodes with multiple parents, pd accumulates via semiring.plus
    # (chain rule: ∂F/∂V_i = Σ_{parents k} ∂F/∂V_k · ∂V_k/∂V_i).
    # ------------------------------------------------------------------

    def _top_down(self):
        """Compute pd(i) = dF/dV_i for every node i in reverse topological order.

        :return: (pd_pos, pd_neg) indexed by absolute node id.
        :rtype: tuple[list, list]
        """
        sr = self.semiring
        topo = self._topo
        n = topo.n
        val = self._val
        node_types = topo.node_types
        children = topo.children

        zero = sr.zero()
        pd_pos = [zero] * (n + 1)
        pd_neg = [zero] * (n + 1)

        # dF/d val(root) = weights[0][0] if present, else one().
        # Because F = val(root) × weights[0][0].
        w0 = self.weights.get(0)
        pd_pos[n] = w0[0] if w0 is not None else sr.one()

        for i in range(n, 0, -1):
            ntype = node_types[i]
            if ntype == 'atom':
                continue

            pd_i = pd_pos[i]
            kids = children[i]
            k = len(kids)

            if ntype == 'disj':
                # ∂val(OR)/∂val(c_j) = 1, so message = pd(i).
                for c in kids:
                    self._accumulate(c, pd_i, pd_pos, pd_neg)

            else:  # 'conj'
                # ∂val(AND)/∂val(c_j) = Π_{m≠j} val(c_m)
                # message = pd(i) × sibling product.
                if k == 1:
                    self._accumulate(kids[0], pd_i, pd_pos, pd_neg)
                elif k == 2:
                    v0 = self._child_val(kids[0], val)
                    v1 = self._child_val(kids[1], val)
                    self._accumulate(kids[0], sr.times(pd_i, v1), pd_pos, pd_neg)
                    self._accumulate(kids[1], sr.times(pd_i, v0), pd_pos, pd_neg)
                else:
                    self._conj_distribute(pd_i, kids, val, pd_pos, pd_neg)

        return pd_pos, pd_neg

    def _conj_distribute(self, pd_i, kids, val, pd_pos, pd_neg):
        """Distribute pd-messages through an AND node with k >= 3 children.

        Uses prefix/suffix products to compute each sibling product in O(k)
        total without division — correct even when a child value is zero.

        :param pd_i: partial derivative of the parent AND node.
        :param kids: tuple of signed child indices.
        :param val: val array from the bottom-up pass.
        :param pd_pos: positive pd accumulator (mutated).
        :param pd_neg: negative pd accumulator (mutated).
        """
        sr = self.semiring
        k = len(kids)
        cvs = [self._child_val(c, val) for c in kids]

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
    #
    # For query atom q with indicator variable λ_q:
    #   Pr(q | e) = ∂F/∂λ_q  /  F(e)
    #
    # In ProbLog's weighted setting this becomes:
    #   unnormalized = w_q × pd(q)
    # which equals F evaluated with w_q^- = 0 (by multilinearity).
    # Normalisation applies when evidence, NSP, or non-AD constraints
    # are present.
    # ------------------------------------------------------------------

    def _extract_marginal(self, node_id):
        """Compute the marginal for a single query node from precomputed pd/val.

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
            or self._topo.normalize
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _child_val(self, c, val):
        """Return the value of a signed child reference.

        :param c: signed child index (positive or negative).
        :param val: val array from the bottom-up pass.
        :return: semiring internal value.
        """
        abs_c = abs(c)
        ntype = self._topo.node_types[abs_c]
        if ntype == 'atom':
            w = self.weights.get(abs_c)
            if w is None:
                return self.semiring.one()
            return w[0] if c > 0 else w[1]
        else:
            # Internal nodes appear only positively in NNF.
            return val[abs_c]

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