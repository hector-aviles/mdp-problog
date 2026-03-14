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
from problog.logic   import Term, Constant, AnnotatedDisjunction
from problog         import get_evaluatable
from collections import defaultdict

class Engine(object):
    """
    Adapter class to Problog grounding and query engine.

    The adapter stores three representations of the program:

        * ``self._db``: ClauseDB prepared from the input program
        * ``self._gp``: ground program 
        * ``self._knowledge``: compiled knowledge base

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
        Return all declared terms for a predicate of arity 1.

        This queries the ClauseDB for facts of the form `declaration_type`

        :param declaration_type: Predicate name to query
        :type declaration_type: str
        :return: List of matching terms.
        :rtype: list[problog.logic.Term]
        """
        return [t[0] for t in self._engine.query(self._db, Term(declaration_type, None))]

    def assignments(self, assignment_type):
        """
        Return assignments for a predicate of arity 2 as a dictionary.

        :param assignment_type: Predicate name to query.
        :type assignment_type: str
        :return: Mapping from assigned term to its value.
        :rtype: dict[problog.logic.Term, problog.logic.Constant]
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
            if instruction_type not in instructions:
                instructions[instruction_type] = []
            assert(self._db.get_node(node) == instruction)  # sanity check
            instructions[instruction_type].append((node, instruction))
        return instructions

    def add_fact(self, term, probability=None):
        """
        Insert a fact into the ClauseDB.

        :param term: Fact term to add.
        :type term: problog.logic.Term
        :param probability: Probability in [0, 1] for a probabilistic fact.
        :type probability: float or None
        :return: ClauseDB node id for the inserted fact.
        :rtype: int
        """
        return self._db.add_fact(term.with_probability(Constant(probability)))

    def get_fact(self, node):
        """
        Return the fact in the ClauseDB corresponding to `node`.

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
        Insert a new rule defined by a `head` and `body` arguments into the ClauseDB.
        Return the corresponding node number.

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
        Return the ClauseDB rule corresponding to `node`.

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
        Add a new utility assignment of `value` to `term` in the ClauseDB.
        Return the corresponding node number.

        :param term: a predicate
        :type term: problog.logic.Term
        :param value: a numeric value
        :type value: float
        :rtype: int
        """
        args = (term.with_probability(None), Constant(1.0 * value))
        utility = Term('utility', *args)
        return self._db.add_fact(utility)

    def get_assignment(self, node):
        """
        Return the assignment from the ClauseDB corresponding to `node`.

        :param node: ClauseDB node id.
        :type node: int
        :return: The (term, value) pair.
        :rtype: tuple of (problog.logic.Term, problog.logic.Constant)
        :raises IndexError: If ``node`` does not correspond to a utility assignment.
        """
        fact = self._db.get_node(node)
        if not (str(fact).startswith('fact') and fact.functor == 'utility'):
            raise IndexError('Node `%d` is not an assignment.' % node)
        return (fact.args[0], fact.args[1])

    def add_annotated_disjunction(self, facts, probabilities):
        """
        Add a new annotated disjunction to the program database from
        a list of `facts` and its `probabilities`.
        Return a list of choice nodes.

        :param facts: list of probabilistic facts
        :type  facts: list of problog.logic.Term
        :param probabilities: list of valid individual probabilities
                              such that the total probability is less
                              than or equal to 1.0
        :type probabilities: list of float in [0.0, 1.0]
        :rtype: list of int
        """
        disjunction = [ f.with_probability(Constant(p)) for f, p in zip(facts, probabilities) ]
        self._db += AnnotatedDisjunction(heads=disjunction, body=Constant('true'))

        choices = []
        for node, term in enumerate(self._db._ClauseDB__nodes):
            if str(term).startswith('choice'):
                choices.append((term, node))

        nodes = []
        for term in disjunction:
            term = term.with_probability(None)
            for choice, node in choices:
                if term in choice.functor.args:
                    nodes.append(node)
        return nodes

    def get_annotated_disjunction(self, nodes):
        """
        Return the ProbLog ``choice`` nodes referenced by ``nodes``.

        :param nodes: ClauseDB node ids.
        :type nodes: list of int
        :return: Choice instructions.
        :rtype: list of problog.engine.choice
        :raises IndexError: If any node id is not a choice node.
        """
        choices = [ self._db.get_node(node) for node in nodes ]
        for choice in choices:
            if not str(choice).startswith('choice'):
                raise IndexError('Node `%d` is not a choice node.' % choice)
        return choices

    def relevant_ground(self, queries):
        """
        Ground the program with respect to a set of query terms.

        After calling this method, :meth:`compile` can be used.

        :param queries: Terms that define the relevant portion of the program.
        :type queries: list of problog.logic.Term
        """
        self._gp = self._engine.ground_all(self._db, queries=queries)

    def compile(self, terms=[]):
        """
        Compile the grounded program into an evaluatable knowledge base.

        :param terms: list of predicates
        :type terms: list of problog.logic.Term or None
        :param backend: Evaluatable backend name. Use None for the default (d-DNNF).
        :type backend: str or None
        :return: Mapping of each provided term to its compiled node id.
        :rtype: dict of (problog.logic.Term, int)
        """
        self._knowledge = get_evaluatable(None).create_from(self._gp)
        term2node = {}
        for term in terms:
            term2node[term] = self._knowledge.get_node_by_name(term)
        return term2node

    def evaluate(self, queries, evidence):
        """
        Evaluate compiled query nodes under evidence.

        :param queries: mapping of predicates to nodes
        :type queries: dict of (problog.logic.Term, int)
        :param evidence: mapping of predicate and evidence weight
        :type evidence: dictionary of (problog.logic.Term, {0, 1})
        :rtype: list of (problog.logic.Term, [0.0, 1.0])
        """
        evaluator = self._knowledge.get_evaluator(semiring=None, evidence=None, weights=evidence)
        return [ (query, evaluator.evaluate(queries[query])) for query in sorted(queries, key=str) ]


    def get_ads_metadata(self):
            """
            Build an inverted index of AD-generated values in the ClauseDB.

            The index maps each concrete value (as string) to the set of annotated
            disjunction group ids that generated it.

            :return: Inverted index: value -> set of AD group ids.
            :rtype: dict of (str, set of int)
            """
            inverted_index = defaultdict(set)
            node_index = 0
            
            while True:
                try:
                    node = self._db.get_node(node_index)
                    
                    if type(node).__name__ == 'choice':
                        parent_id = node.group
                        fact_term = node.functor.args[2] 
                        
                        if fact_term.args:
                            # Término con argumentos (ej. colores(rojo) -> mapea 'rojo')
                            for arg in fact_term.args:
                                if hasattr(arg, 'is_var') and not arg.is_var():
                                    inverted_index[str(arg)].add(parent_id)
                        else:
                            # Átomo sin argumentos (ej. television -> mapea 'television')
                            inverted_index[str(fact_term.functor)].add(parent_id)
                            
                except IndexError:
                    # Fin de la tabla de instrucciones
                    break
                    
                node_index += 1
                
            return dict(inverted_index)