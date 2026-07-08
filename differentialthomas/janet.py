r"""
Janet trees -- port of ``tree`` from the DifferentialThomas Maple package
(the involutive-division / multiplicative-variable machinery; see "Fast
Search for the Janet Divisor", Gerdt-Yanovich-Blinkov, for the theory).

Node / leaf representation
==========================

The reference models a tree as nested Maple tables: an *interior node* is a
table with keys ``Position`` (a list of ``nops(ivar)`` ints, ``-1`` = not yet
touched variable, else the derivative order fixed at this depth), ``Degree``
(child one derivative deeper in the current variable) and ``Variable`` (child
in the next independent variable); each child slot holds another node, a
*leaf*, or ``NULL``.  A leaf is a PolynomialObject stored directly in a child
slot -- leaf-ness is detected by the *absence* of the ``Degree`` / ``Variable``
keys (hence the reference's warning, ``polynomobjects:49``, never to add a
``Degree`` key to a PolynomialObject).  The root of a per-dvar tree may itself
be a leaf (after an order-zero insertion, ``tree:393-399``).

The port maps this onto:

- :class:`JanetNode` -- interior node with ``position`` / ``degree`` /
  ``variable`` slots (``None`` plays ``NULL``) and the transient ``delete``
  flag of ``RemoveElementsInSubtree``;
- a **leaf is a bare** :class:`~differentialthomas.polyobj.PolynomialObject`
  (``isinstance`` replaces the key-presence test -- no key collision is
  possible by construction);
- :class:`JanetTreesObject` -- the per-ranking table of trees, keyed by
  dependent-variable head, sharing the Ranking (``NoDeepCopy``).

``MultiplicativeVariables`` lives on the leaf PolynomialObject as a mutable
list with entries ``0`` (classically non-multiplicative) or ``math.inf``
(classically multiplicative); the Hilbert-series code also admits finite
positive bounds, but the insertion machinery only ever writes ``0`` /
``infinity``.

Engine invariants (ported as ``assert``, mirroring the reference ASSERTs)
=========================================================================

- only elements whose leader is a differential variable are inserted
  (``tree:391``);
- the insertion walk never passes *through* a leaf: the engine only inserts
  Janet-irreducible elements, so a leaf on a strictly shorter path would have
  been found by ``JanetDivisorInTrees`` and reduced first.  (The reference
  ASSERTs at ``tree:479/498`` additionally tolerate an equal-leader leaf, but
  recursing into one would dereference the leaf's unassigned ``Position`` --
  the equal-leader *replacement* is reachable only via the ``togo=[1,0..]``
  branch, ``tree:488-494``, or the order-zero branch, which both handle it.);
- a divisor returned by the tree search must be a genuine *involutive*
  divisor (``tree:79-81``, a hard ``error`` in the reference, ported as
  :class:`RuntimeError`).

Deferred (combinatorial outputs, not needed for the decomposition):
``HilbertSeriesFromTree(s)``, ``FactorModuleBasisFromTree(s)(Recursive)``,
``PrincipalDerivativesFromTree(s)``, ``HilbertSamuel/HilbertPolynomialFromTree``,
``CombinatoricInfoFromTree(s)``.  ``print_janet_tree(s)`` is kept as a minimal
debugging pretty-printer.

Note: ``RemoveElementsInSubtree`` (``tree:430``) is ported for completeness
but has **no callers anywhere in the reference package** (dead code; its
transient ``Delete`` flag is set and never consumed).
"""

import math

from .polyobj import PolynomialObject
from .derivation import partial_derivative

INFINITY = math.inf


class JanetNode(object):
    """An interior Janet-tree node (reference ``tree:40-44``)."""

    __slots__ = ("position", "degree", "variable", "delete")

    def __init__(self, position):
        self.position = list(position)
        self.degree = None              # deeper derivative in current ivar
        self.variable = None            # next ivar
        self.delete = False             # transient RemoveElementsInSubtree flag

    def __repr__(self):
        def slot(v):
            if v is None:
                return "-"
            if isinstance(v, JanetNode):
                return "node"
            return "leaf"
        return "JanetNode(%s, D=%s, V=%s)" % (
            self.position, slot(self.degree), slot(self.variable))

    def deep_copy(self):
        from .general import deep_copy as _dc
        n = JanetNode(self.position)
        n.degree = _dc(self.degree)
        n.variable = _dc(self.variable)
        n.delete = self.delete
        return n


class JanetTreesObject(object):
    """`DifferentialThomas` JanetTreesObject (``tree:24-27``): the shared
    Ranking plus one tree per dependent variable, keyed by head name.
    Build via :func:`create_janet_trees_object`."""

    __slots__ = ("ranking", "trees")

    def __init__(self, ranking):
        # `DifferentialThomas/CreateJanetTreesObject` (tree:61-70)
        self.ranking = ranking
        n = len(ranking.ivar)
        self.trees = {u: JanetNode([0] + [-1] * (n - 1))
                      for u in ranking.dvar}

    def __getitem__(self, head):
        return self.trees[head]

    def __setitem__(self, head, value):
        self.trees[head] = value

    def deep_copy(self):
        """Trees copied, Ranking shared (``NoDeepCopy``)."""
        from .general import deep_copy as _dc
        new = JanetTreesObject.__new__(JanetTreesObject)
        new.ranking = self.ranking
        new.trees = {u: _dc(t) for u, t in self.trees.items()}
        return new


def create_janet_trees_object(ranking):
    """`DifferentialThomas/CreateJanetTreesObject` (``tree:61``)."""
    return JanetTreesObject(ranking)


def current_var(l):
    """`DifferentialThomas/CurrentVar` (``tree:48-58``): the largest 1-based
    position of ``l`` with a nonnegative entry (where in the tree we are);
    1 if all entries are negative."""
    for i in range(len(l), 0, -1):
        if l[i - 1] >= 0:
            return i
    return 1


# ---------------------------------------------------------------------------
# Divisor search
# ---------------------------------------------------------------------------

def janet_divisor_in_trees(trees, p):
    """`DifferentialThomas/JanetDivisorInTrees` (``tree:73-83``): the leaf of
    ``trees[LeadingFunction(p)]`` reached by descending along ``p``'s leading
    derivation, or ``None``.  A found leaf is verified to be an *involutive*
    divisor: ``divisor.MultiplicativeVariables - (deriv(p) - deriv(divisor))``
    must be componentwise nonnegative-or-infinity (a hard error otherwise,
    as in the reference)."""
    if p.leader() == 1:
        return None
    tree = trees[p.leading_function()]
    divisor = janet_divisor_in_tree(tree, p.leading_derivation())
    if divisor is not None:
        mv = divisor.f["MultiplicativeVariables"]
        dp = p.leading_derivation()
        dd = divisor.leading_derivation()
        if any(m - (a - b) < 0 for m, a, b in zip(mv, dp, dd)):
            raise RuntimeError("wrong multiplicative variables")
    return divisor


def janet_divisor_in_tree(node, derivations):
    """`DifferentialThomas/JanetDivisorInTree` (``tree:90-106``): recursion
    anchor; handles the root-is-already-a-leaf case."""
    if isinstance(node, JanetNode):
        return janet_divisor_in_tree_rek(node, list(derivations))
    if isinstance(node, PolynomialObject):
        # tree:95: the root leaf must have a dependent-variable leader
        if node.leading_function() in node.ranking.dvar:
            return node
    return None


def janet_divisor_in_tree_rek(node, derivations):
    """`DifferentialThomas/JanetDivisorInTreeRek` (``tree:112-135``): descend
    ``Degree`` while the current variable's remaining order is positive (or
    fall over to ``Variable`` when no ``Degree`` child exists), ``Variable``
    when it is zero; a reached leaf is the candidate divisor."""
    if isinstance(node, PolynomialObject):        # we have a polynom here
        return node
    if not derivations:
        return None
    if derivations[0] == 0:
        if node.variable is None:
            return None
        return janet_divisor_in_tree_rek(node.variable, derivations[1:])
    if node.degree is None:
        if node.variable is None:
            return None
        return janet_divisor_in_tree_rek(node.variable, derivations[1:])
    return janet_divisor_in_tree_rek(
        node.degree, [derivations[0] - 1] + derivations[1:])


# ---------------------------------------------------------------------------
# Leaf enumeration
# ---------------------------------------------------------------------------

def janet_tree_leafs(node):
    """`DifferentialThomas/JanetTreeLeafs` (``tree:139-157``): all leaves of a
    (sub)tree, Degree subtree before Variable subtree."""
    if isinstance(node, PolynomialObject):
        assert node.f.get("Polynom") is not None, "wrong in tree 1"
        return [node]
    assert isinstance(node, JanetNode), "wrong in tree 2/3"
    result = []
    if node.degree is not None:
        result.extend(janet_tree_leafs(node.degree))
    if node.variable is not None:
        result.extend(janet_tree_leafs(node.variable))
    return result


def janet_trees_leafs(trees):
    """`DifferentialThomas/JanetTreesLeafs` (``tree:161-165``): all leaves of
    a trees object, flattened in ranking ``dvar`` order."""
    result = []
    for u in trees.ranking.dvar:
        result.extend(janet_tree_leafs(trees[u]))
    return result


# ---------------------------------------------------------------------------
# Insertion (with multiplicative-variable side effects)
# ---------------------------------------------------------------------------

def insert_into_janet_trees(trees, p):
    """`DifferentialThomas/InsertIntoJanetTrees` (``tree:389-403``): insert
    ``p`` into the tree of its leading function.  As a side effect the
    multiplicative variables are made to fit; returns the list of all
    non-multiplicative prolongations that come into existence by adding this
    element, plus all elements it renders obsolete (which are removed from
    the tree)."""
    assert trees.ranking.is_differential_variable(p.leader()), \
        "no differential variable as leader"
    assert isinstance(p, PolynomialObject), "p not a table"
    head = p.leading_function()
    ld = p.leading_derivation()
    if ld == [0] * len(ld):
        # order-zero leader: p replaces the whole tree (tree:393-399)
        leafs = janet_tree_leafs(trees[head])
        if p.f.get("MultiplicativeVariables") is None:
            p.f["MultiplicativeVariables"] = \
                [INFINITY] * len(p.ranking.ivar)
        trees[head] = p
        return leafs
    root = trees[head]
    # engine invariant: an order-zero leaf at the root involutively divides
    # every jet of this head, so a positive-order p would have been reduced,
    # never inserted (the reference would dereference the leaf's unassigned
    # Position here)
    assert isinstance(root, JanetNode), \
        "insertion into a tree whose root is an order-zero leaf"
    return insert_into_janet_tree(root, p)


def insert_into_janet_tree(node, p, togo=None):
    """`DifferentialThomas/InsertIntoJanetTree` (``tree:455-506``): walk the
    ``Degree``/``Variable`` chain following ``p``'s remaining derivation
    index ``togo``, building nodes as needed; changes the tree (and
    multiplicative variables) with side effects and returns all elements
    divided by ``p`` plus all newly required prolongations."""
    assert p.leader() != 1
    assert isinstance(p, PolynomialObject), "p not a table"
    result = []
    if togo is None:
        togo = list(p.leading_derivation())
        if p.f.get("MultiplicativeVariables") is None:
            p.f["MultiplicativeVariables"] = \
                [INFINITY] * len(p.ranking.ivar)
    # tree:479/498 invariant -- the walk must not pass through a leaf (see
    # module docstring)
    assert isinstance(node, JanetNode), "some strange object in tree"
    currentvar = current_var(node.position)
    ivar = p.ranking.ivar
    if togo[0] == 0:
        # we want to go into the Variable node; don't touch node.degree
        if node.degree is not None:
            # p has lower order in currentvar than an existing Degree
            # sibling: currentvar is non-multiplicative for p
            p.f["MultiplicativeVariables"][currentvar - 1] = 0
        if node.variable is not None:
            result.extend(insert_into_janet_tree(node.variable, p, togo[1:]))
        else:
            result.extend(
                insert_variable_into_empty_janet_tree(node, p, togo[1:]))
    else:
        # we want to go into the Degree node
        if node.variable is not None:
            # correct the multiplicative variables in the side tree: its
            # elements have lower order in currentvar than p will
            result.extend(remove_multiplicative_variable_in_subtree(
                node.variable, currentvar, ivar))
        if togo[0] == 1 and not any(togo[1:]):
            # just one step to go: p becomes the Degree child; an element or
            # whole subtree already there is replaced (its leaves returned)
            results = node.degree
            if results is not None:
                result.extend(janet_tree_leafs(results))
            node.degree = p
            result.extend(complete_element_in_janet_tree(p))
        else:
            if node.degree is not None:
                result.extend(insert_into_janet_tree(
                    node.degree, p, [togo[0] - 1] + togo[1:]))
            else:
                result.extend(insert_degree_into_empty_janet_tree(
                    node, p, [togo[0] - 1] + togo[1:]))
    return result


def insert_degree_into_empty_janet_tree(node, p, togo):
    """`DifferentialThomas/InsertDegreeIntoEmptyJanetTree` (``tree:509-525``):
    build the ``Degree`` chain below ``node`` for the remaining index
    ``togo`` and place ``p`` at its end."""
    assert isinstance(p, PolynomialObject), "p not a table"
    currentvar = current_var(node.position)
    if not any(togo):
        node.degree = p
        return complete_element_in_janet_tree(p)
    newpos = list(node.position)
    newpos[currentvar - 1] += 1
    newnode = JanetNode(newpos)
    node.degree = newnode
    if togo[0] == 0:
        return insert_variable_into_empty_janet_tree(newnode, p, togo[1:])
    return insert_degree_into_empty_janet_tree(
        newnode, p, [togo[0] - 1] + togo[1:])


def insert_variable_into_empty_janet_tree(node, p, togo):
    """`DifferentialThomas/InsertVariableIntoEmptyJanetTree`
    (``tree:528-543``): like the above for the ``Variable`` direction (the
    new node fixes order 0 in the next independent variable)."""
    currentvar = current_var(node.position)
    if not any(togo):
        node.variable = p
        return complete_element_in_janet_tree(p)
    newpos = list(node.position)
    newpos[currentvar] = 0          # entry currentvar+1 (1-based) := 0
    newnode = JanetNode(newpos)
    node.variable = newnode
    if togo[0] == 0:
        return insert_variable_into_empty_janet_tree(newnode, p, togo[1:])
    return insert_degree_into_empty_janet_tree(
        newnode, p, [togo[0] - 1] + togo[1:])


def remove_multiplicative_variable_in_subtree(node, indexofvar, ivar):
    """`DifferentialThomas/RemoveMultiplicativeVariableInSubtree`
    (``tree:407-427``): flip variable ``indexofvar`` (1-based) to
    non-multiplicative (``0``) on every leaf of the subtree; each leaf whose
    flag actually flips contributes its now-required prolongation
    ``PartialDerivative(leaf, ivar[indexofvar])`` to the returned list."""
    if isinstance(node, PolynomialObject):
        # reference leaf test: assigned(node['MultiplicativeVariables'])
        mv = node.f.get("MultiplicativeVariables")
        assert mv is not None, "leaf without multiplicative variables"
        if mv[indexofvar - 1] != 0:
            mv[indexofvar - 1] = 0
            return [partial_derivative(node, ivar[indexofvar - 1])]
        return []
    result = []
    if node.degree is not None:
        result.extend(remove_multiplicative_variable_in_subtree(
            node.degree, indexofvar, ivar))
    if node.variable is not None:
        result.extend(remove_multiplicative_variable_in_subtree(
            node.variable, indexofvar, ivar))
    return result


def remove_elements_in_subtree(node, togo):
    """`DifferentialThomas/RemoveElementsInSubtree` (``tree:430-451``): cut
    every ``Degree`` subtree hanging at derivation index ``togo`` (in any
    trailing-variable combination), returning the removed leaves; interior
    nodes left with no live children get the transient ``delete`` flag.

    NOTE: dead code in the reference -- no caller in the package, and the
    ``Delete`` flag is never consumed.  Ported for faithfulness, including a
    reference quirk (oracle-confirmed): cutting a subtree assigns
    ``node['Degree'] := NULL``, which in Maple *unassigns* the table entry,
    so the flag condition's ``node['Degree'] = NULL`` test (an unevaluated
    name vs NULL) is then false -- a node whose Degree was cut by this very
    call is never flagged deletable.  Only ``None``-since-construction
    children (and flagged interior children) count."""
    result = []
    if isinstance(node, PolynomialObject):
        return result
    degree_cut = False
    if togo[0] == 1 and not any(togo[1:]) and node.degree is not None:
        # remove the whole subtree node.degree
        result.extend(janet_tree_leafs(node.degree))
        node.degree = None
        degree_cut = True               # reference: entry now *unassigned*
    elif togo[0] > 0 and node.degree is not None:
        result.extend(remove_elements_in_subtree(
            node.degree, [togo[0] - 1] + togo[1:]))
    if len(togo) > 1 and node.variable is not None:
        result.extend(remove_elements_in_subtree(node.variable, togo[1:]))
    if ((not degree_cut)
            and (node.degree is None
                 or (isinstance(node.degree, JanetNode) and node.degree.delete))
            and (node.variable is None
                 or (isinstance(node.variable, JanetNode)
                     and node.variable.delete))):
        node.delete = True
    return result


def complete_element_in_janet_tree(p):
    """`DifferentialThomas/CompleteElementInJanetTree` (``tree:546-555``):
    emit the not-yet-considered non-multiplicative prolongations of ``p``
    (marking them considered via the ``ProlongationConsidered`` side
    effect)."""
    mv = p.f["MultiplicativeVariables"]
    ivar = p.ranking.ivar
    result = []
    for a in range(len(mv)):
        if mv[a] < INFINITY and not p.prolongation_considered(ivar[a]):
            result.append(partial_derivative(p, ivar[a]))
    return result


# ---------------------------------------------------------------------------
# Debug pretty-printer (minimal PrintJanetTree(s), tree:560-590)
# ---------------------------------------------------------------------------

def print_janet_tree(rootnode, combinatorics=False, out=None):
    import sys
    out = out or sys.stdout
    for leaf in janet_tree_leafs(rootnode):
        mv = ",".join("inf" if m == INFINITY else "%3d" % m
                      for m in leaf.f.get("MultiplicativeVariables", []))
        line = "%s, [%s], %s" % (leaf.leader(), mv, leaf.rank())
        if not combinatorics:
            line += ", %s" % (leaf.standard_form(),)
        out.write(line + "\n")


def print_janet_trees(trees, combinatorics=False, out=None):
    import sys
    out = out or sys.stdout
    for u in trees.ranking.dvar:
        out.write("%s:\n" % u)
        print_janet_tree(trees[u], combinatorics, out)
