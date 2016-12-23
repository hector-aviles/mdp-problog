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

from problog.program import PrologString
from problog.engine  import DefaultEngine
from problog.logic   import Term, Constant
from problog         import get_evaluatable

class Engine(object):
	"""
	Adapter class to Problog grounding and query engine.

	:param program: a valid MDP-ProbLog program
	:type program: str
	"""

	def __init__(self, program):
		self._engine = DefaultEngine()
		self._db = self._engine.prepare(PrologString(program))
		self._gp = None
		self._knowledge = None

	def declarations(self, declaration_type):
		"""
		Return a list of all terms of type `declaration_type`.

		:param declaration_type: declaration type.
		:type declaration_type: str
		:rtype: list of problog.logic.Term
		"""
		return [t[0] for t in self._engine.query(self._db, Term(declaration_type, None))]

	def assignments(self, assignment_type):
		"""
		Return a dictionary of assignments of type `assignment_type`.

		:param assignment_type: assignment type.
		:type assignment_type: str
		:rtype: dict of (problog.logic.Term, problog.logic.Constant) items.
		"""
		return dict(self._engine.query(self._db, Term(assignment_type, None, None)))

	def get_instructions_table(self):
		"""
		Return the table of instructions separated by instruction type
		as described in problog.engine.ClauseDB.

		:rtype: dict of (str, list of (node,namedtuple))
		"""
		instructions = {}
		for node, instruction in enumerate(self._db._ClauseDB__nodes):
			instruction_type = str(instruction)
			instruction_type = instruction_type[:instruction_type.find('(')]
			if not instruction_type in instructions:
				instructions[instruction_type] = []
			assert(self._db.get_node(node) == instruction) # sanity check
			instructions[instruction_type].append((node,instruction))
		return instructions

	def add_fact(self, term, probability=None):
		"""
		Add a new `term` with a given `probability` to the program database.
		Return the corresponding node number.

		:param term: a predicate
		:type term: problog.logic.Term
		:param probability: a number in [0,1]
		:type probability: float
		:rtype: int
		"""
		return self._db.add_fact(term.with_probability(Constant(probability)))

	def get_fact(self, node):
		"""
		Return the fact in the table of instructions corresponding to `node`.

		:param node: identifier of fact in table of instructions
		:type node: int
		:rtype: problog.engine.fact
		"""
		fact = self._db.get_node(node)
		if not str(fact).startswith('fact'):
			raise IndexError('Node `%d` is not a fact.' % node)
		return fact

	def add_rule(self, head, body):
		"""
		Add a new rule defined by a `head` and `body` arguments
		to the program database. Return the corresponding node number.

		:param head: a predicate
		:type head: problog.logic.Term
		:param body: a list of literals
		:type body: list of problog.logic.Term or problog.logic.Not
		:rtype: int
		"""
		b = body[0]
		for term in body[1:]:
			b = b & term
		rule = head << b
		return self._db.add_clause(rule)

	def get_rule(self, node):
		"""
		Return the rule in the table of instructions corresponding to `node`.

		:param node: identifier of rule in table of instructions
		:type node: int
		:rtype: problog.engine.clause
		"""
		rule = self._db.get_node(node)
		if not str(rule).startswith('clause'):
			raise IndexError('Node `%d` is not a rule.' % node)
		return rule

	def add_assignment(self, term, value):
		"""
		Add a new utility assignment of `value` to `term` in the program database.
		Return the corresponding node number.

		:param term: a predicate
		:type term: problog.logic.Term
		:param value: a numeric value
		:type value: float
		:rtype: int
		"""
		args = (term.with_probability(None), Constant(1.0*value))
		utility = Term('utility', *args)
		return self._db.add_fact(utility)

	def get_assignment(self, node):
		"""
		Return the assignment in the table of instructions corresponding to `node`.

		:param node: identifier of assignment in table of instructions
		:type node: int
		:rtype: pair of (problog.logic.Term, problog.logic.Constant)
		"""
		fact = self._db.get_node(node)
		if not (str(fact).startswith('fact') and fact.functor == 'utility'):
			raise IndexError('Node `%d` is not an assignment.' % node)
		return (fact.args[0], fact.args[1])

	def relevant_ground(self, queries):
		"""
		Create ground program with respect to `queries`.

		:param queries: list of predicates
		:type queries: list of problog.logic.Term
		"""
		self._gp = self._engine.ground_all(self._db, queries=queries)

	def compile(self):
		""" Create compiled knowledge database from ground program. """
		self._knowledge = get_evaluatable(None).create_from(self._gp)

	def evaluate(self, queries, evidence):
		"""
		Compute probabilities of `queries` given `evidence`.

		:param queries: list of predicates
		:type queries: list of problog.logic.Term
		:param evidence: mapping of predicate and evidence weight
		:type evidence: dictionary of (problog.logic.Term, {0, 1})
		:rtype: dictionary of (problog.logic.Term, [0.0, 1.0])
		"""
		evaluator = self._knowledge.get_evaluator(semiring=None, evidence=None, weights=evidence)
		result = { query: evaluator.evaluate(node) for query, node in queries.items() }
		return result
