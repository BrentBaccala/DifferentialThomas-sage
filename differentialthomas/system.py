r"""
DifferentialSystem -- port of the container from ``differentialsystems`` in the
DifferentialThomas Maple package (the minimal Phase-4 subset the split
operators need to copy and spawn).

A :class:`DifferentialSystem` mirrors the reference table (``differentialsystems:24``):

- ``Q``               -- work queue: a list of :class:`PolynomialObject`
                         (equations and inequations not yet treated);
- ``JanetTrees``      -- a :class:`~differentialthomas.janet.JanetTreesObject`
                         holding the equations already put into involutive form;
- ``Inequations``     -- list of inequation :class:`PolynomialObject`;
- ``Ranking``         -- the SHARED ranking (never deep-copied);
- ``Inconsistent``    -- bool;
- ``Finished``        -- bool;
- ``MaxOrderInSystem``-- highest differential order ever seen in this system.

The deep-copy / share-ranking boundary (``general.deep_copy`` +
``NoDeepCopy``) is the load-bearing invariant: on a split the system, its Q,
its trees and its inequations are DEEP-copied so the child's mutations do not
alias the parent's, but the ranking (and its caches / ring) is SHARED.  Getting
this wrong is either an aliased-mutation correctness bug or a memory blow-up
(cloned rankings).

Ported here (Phase 4): the container + ``CreateDifferentialSystem``, the
equation/inequation accessors, ``deep_copy``, ``DifferentialSystemInequationImplied``
(needed by the split operators), and ``ReduceQListInSystem`` (needed by the
Phase-5 main loop; short and self-contained).  The full finish logic /
tail-reduction / the main loop remain Phase 5.
"""

from .general import deep_copy as _deep_copy
from .polyobj import (create_polynomial_object, is_differential_field_element,
                      inconsistent_polynom)
from .sorting import sort_qlist, insert_into_qlist
from .reduction import (reduce_wrt_janet_trees,
                        reduce_nonlinear_tail_wrt_janet_trees)
from .janet import (create_janet_trees_object, janet_trees_leafs,
                    janet_divisor_in_trees)


class DifferentialSystem(object):
    """See module docstring.  Build via :func:`create_differential_system`."""

    __slots__ = ("Ranking", "Inequations", "Q", "Finished", "JanetTrees",
                 "Inconsistent", "MaxOrderInSystem")

    def __init__(self):
        self.Ranking = None
        self.Inequations = []
        self.Q = []
        self.Finished = False
        self.JanetTrees = None
        self.Inconsistent = False
        self.MaxOrderInSystem = 0

    # -- `DifferentialThomas/DeepCopy` (share-ranking) -----------------------

    def deep_copy(self):
        """Deep-copy the system; SHARE the ranking (``NoDeepCopy``).  Q,
        trees and inequations are deep-copied (their elements' mutations must
        not alias the parent)."""
        new = DifferentialSystem()
        new.Ranking = self.Ranking                 # shared (no_deep_copy)
        new.Q = [_deep_copy(q) for q in self.Q]
        new.Inequations = [_deep_copy(q) for q in self.Inequations]
        new.JanetTrees = self.JanetTrees.deep_copy()
        new.Inconsistent = self.Inconsistent
        new.Finished = self.Finished
        new.MaxOrderInSystem = self.MaxOrderInSystem
        return new

    def __repr__(self):
        return ("DifferentialSystem(|Q|=%d, |Ineq|=%d, inconsistent=%s)"
                % (len(self.Q), len(self.Inequations), self.Inconsistent))


# ---------------------------------------------------------------------------
# `DifferentialThomas/CreateDifferentialSystem` (differentialsystems:36)
# ---------------------------------------------------------------------------

def create_differential_system(q2, ranking, trees=None):
    """`DifferentialThomas/CreateDifferentialSystem`: a system with sorted work
    queue ``q2`` (list of PolynomialObjects), the given ``ranking`` (shared),
    and either the supplied Janet ``trees`` or a fresh empty trees object."""
    ds = DifferentialSystem()
    ds.Ranking = ranking
    ds.Inequations = []
    ds.Q = sort_qlist(list(q2))
    ds.Finished = False
    if trees is not None:
        ds.JanetTrees = trees
    else:
        ds.JanetTrees = create_janet_trees_object(ranking)
    ds.MaxOrderInSystem = max((q.max_order() for q in ds.Q), default=0)
    return ds


# ---------------------------------------------------------------------------
# Accessors (differentialsystems:97-161)
# ---------------------------------------------------------------------------

def differential_system_janet_trees(ds):
    """`DifferentialThomas/DifferentialSystemJanetTrees`."""
    return ds.JanetTrees


def differential_system_equations(ds):
    """`DifferentialThomas/DifferentialSystemEquations`: standard forms of the
    tree leaves plus the equation-typed elements of ``Q``."""
    trees = ds.JanetTrees
    out = [leaf.standard_form() for leaf in janet_trees_leafs(trees)]
    out += [q.standard_form() for q in ds.Q if q.equation()]
    return out


def differential_system_inequations(ds):
    """`DifferentialThomas/DifferentialSystemInequations`: standard forms of the
    ``Inequations`` plus the inequation-typed elements of ``Q``."""
    out = [q.standard_form() for q in ds.Inequations]
    out += [q.standard_form() for q in ds.Q if q.inequation()]
    return out


# ---------------------------------------------------------------------------
# reduction wrt the system (the piece of DifferentialSystemReduce the split /
# implied-inequation logic needs)
# ---------------------------------------------------------------------------

def differential_system_reduce_object(ds, q):
    """The object form of `DifferentialThomas/DifferentialSystemReduce`:
    simplify ``q``'s standard form, then head+initial reduce w.r.t. the trees.
    Returns a fresh PolynomialObject (the reference keeps the ``Inequation``
    flag via SimplifyPolynom / ReduceWRTJanetTrees)."""
    rk = ds.Ranking
    pp = create_polynomial_object(q.standard_form(), rk,
                                  inequation=q.inequation())
    pp.simplify_polynom()
    return reduce_wrt_janet_trees(ds.JanetTrees, pp)


# ---------------------------------------------------------------------------
# `DifferentialThomas/DifferentialSystemInequationImplied` (diffsystems:411)
# ---------------------------------------------------------------------------

def _factor_stdforms(p):
    """The primitive factors of a substrate polynomial (Maple
    ``factors(p)[2]`` -> the ``[factor, mult]`` first components)."""
    F = p.factor()
    return [g for g, _m in F]


def _in_up_to_sign(p, l):
    """``p in l`` up to a nonzero scalar (sign): inequations/equations are
    defined up to a nonzero constant, so membership is tested modulo sign --
    the reference's exact structural ``in`` on content-simplified primitives
    coincides with this on all gated inputs."""
    for m in l:
        if (p - m).is_zero() or (p + m).is_zero():
            return True
    return False


def differential_system_inequation_implied(ds, q):
    """`DifferentialThomas/DifferentialSystemInequationImplied`: True iff the
    non-vanishing of ``q`` is already implied by the system's inequations (so a
    split on ``q`` would be redundant).

    Ported faithfully: zero -> False; field element -> True; direct membership
    (up to sign); reduce and re-test; and the all-factors-are-inequations
    tests for both ``q`` and its reduced form."""
    if q.standard_form().is_zero():
        return False
    if is_differential_field_element(q):
        return True
    l = [obj.standard_form() for obj in ds.Inequations]
    l += [obj.standard_form() for obj in ds.Q if obj.inequation()]
    qsf = q.standard_form()
    if _in_up_to_sign(qsf, l):
        return True
    p = differential_system_reduce_object(ds, q)
    psf = p.standard_form()
    if psf.is_zero():
        return False
    if is_differential_field_element(p):
        return True
    if _in_up_to_sign(psf, l):
        return True
    fak = _factor_stdforms(qsf)
    if all(_in_up_to_sign(g, l) for g in fak):
        return True
    if not (qsf - psf).is_zero():
        fak = _factor_stdforms(psf)
        if all(_in_up_to_sign(g, l) for g in fak):
            return True
    return False


# ---------------------------------------------------------------------------
# `DifferentialThomas/ReduceQListInSystem` (differentialsystems:451)
# ---------------------------------------------------------------------------

def reduce_qlist_in_system(ds):
    """`DifferentialThomas/ReduceQListInSystem`: reduce a slice of ``Q``
    w.r.t. the trees (default: the inequations), setting ``Inconsistent`` on a
    vanished inequation or a nonzero field-element equation, then re-insert.

    (Provided for the Phase-5 loop; not exercised by the Phase-4 gate.)"""
    mode = ds.Ranking.reduce_qlist_in_system
    if mode == "Inequations":
        l = [q for q in ds.Q if q.inequation()]
        ds.Q = [q for q in ds.Q if not q.inequation()]
    elif mode != 0:
        import math
        k = math.floor(len(ds.Q) * mode)
        l = ds.Q[:k]
        ds.Q = ds.Q[k:]
    else:
        l = []
    l = [reduce_wrt_janet_trees(ds.JanetTrees, q) for q in l]
    for a in l:
        if a.inequation() and a.standard_form().is_zero():
            ds.Inconsistent = True
        if (a.equation() and not a.standard_form().is_zero()
                and is_differential_field_element(a)):
            ds.Inconsistent = True
    ds.Q = insert_into_qlist(l, ds.Q)


# ---------------------------------------------------------------------------
# `DifferentialThomas/DifferentialSystemReduce` (differentialsystems:180)
# ---------------------------------------------------------------------------

def differential_system_reduce(ds, p):
    """`DifferentialThomas/DifferentialSystemReduce`: head+initial reduce ``p``
    (a polynomial or PolynomialObject) w.r.t. the system's trees; returns the
    reduced *standard form* (a substrate polynomial).  The object form (used by
    the implied-inequation heuristic and the split operators) is
    :func:`differential_system_reduce_object`."""
    return differential_system_reduce_object(ds, p).standard_form()


# ---------------------------------------------------------------------------
# `DifferentialThomas/DifferentialSystemNormalForm` (differentialsystems:214)
# ---------------------------------------------------------------------------

def differential_system_normal_form(ds, p):
    """`DifferentialThomas/DifferentialSystemNormalForm`: full head+tail
    (non-linear) reduction of ``p`` w.r.t. the system, returned as a normalised
    rational function ``standard_form(r) / u`` (``u`` the reduced denominator
    multiplier).  For the polynomial inputs of ex1-ex3 the multiplier is a
    field element, so this reduces to the tail-reduced standard form."""
    rk = ds.Ranking
    pp = create_polynomial_object(p, rk)
    pp.simplify_polynom()
    r, u = reduce_nonlinear_tail_wrt_janet_trees(
        ds.JanetTrees, pp, "denominator")
    rr = r.standard_form()
    if is_differential_field_element(create_polynomial_object(u, rk)):
        return rr.exquo(rk.ring(u)) if not (u - rk.ring.one()).is_zero() else rr
    # non-trivial denominator: reduce it too (recursion, as the reference)
    uu = differential_system_normal_form(ds, create_polynomial_object(u, rk))
    return rr.exquo(uu)


# ---------------------------------------------------------------------------
# `DifferentialThomas/DifferentialSystemTailReduce` (differentialsystems:275)
# ---------------------------------------------------------------------------

def differential_system_tail_reduce(ds, p, *extra):
    """`DifferentialThomas/DifferentialSystemTailReduce`: (non-linear) tail
    reduction of ``p`` w.r.t. the system's trees.  ``p`` may be a
    PolynomialObject (reduced/substituted in place, as the reference does) or a
    polynomial (wrapped first).  ``extra`` forwards the mode strings
    (``"final"`` / ``"denominator"`` / ...) to
    :func:`reduce_nonlinear_tail_wrt_janet_trees`."""
    pp = create_polynomial_object(p, ds.Ranking)
    return reduce_nonlinear_tail_wrt_janet_trees(ds.JanetTrees, pp, *extra)


# ---------------------------------------------------------------------------
# `DifferentialThomas/DifferentialSystemTailReduction` (differentialsystems:294)
# ---------------------------------------------------------------------------

def differential_system_tail_reduction(ds):
    """`DifferentialThomas/DifferentialSystemTailReduction`: tail-reduce every
    equation (tree leaf) of the system.  If any leader changed as a result, the
    involutive basis is invalidated: re-queue all leaves and inequations, reset
    the trees, and clear ``Finished`` (so the main loop re-completes)."""
    trees = ds.JanetTrees
    leaves = list(janet_trees_leafs(trees))
    oldleaders = [leaf.leader() for leaf in leaves]
    for a in leaves:
        r = differential_system_tail_reduce(ds, a, "final")
        r.simplify_polynom()
    newleaders = [leaf.leader() for leaf in janet_trees_leafs(trees)]
    if newleaders != oldleaders:
        ds.Q = insert_into_qlist(list(janet_trees_leafs(trees)), ds.Q)
        ds.Q = insert_into_qlist(list(ds.Inequations), ds.Q)
        ds.Inequations = []
        ds.JanetTrees = create_janet_trees_object(ds.Ranking)
        ds.Finished = False


# ---------------------------------------------------------------------------
# `DifferentialThomas/ReduceInequationsInDifferentialSystem`
# (differentialsystems:322)
# ---------------------------------------------------------------------------

def reduce_inequations_in_differential_system(ds, q):
    """`DifferentialThomas/ReduceInequationsInDifferentialSystem`: after ``q``
    is inserted as a new equation (leader ``Leader(q)``), move any inequation
    that mentions that leader, or whose own leader is now Janet-divisible by the
    trees, back into ``Q`` for re-treatment."""
    trees = ds.JanetTrees
    lq = q.leader()
    l1 = [a for a in ds.Inequations if lq in a.diff_var_list()]
    l2 = [a for a in ds.Inequations if lq not in a.diff_var_list()]
    ds.Inequations = l2
    ds.Q = insert_into_qlist(l1, ds.Q)
    l1 = [a for a in ds.Inequations
          if janet_divisor_in_trees(trees, a) is not None]
    l2 = [a for a in ds.Inequations
          if janet_divisor_in_trees(trees, a) is None]
    ds.Inequations = l2
    ds.Q = insert_into_qlist(l1, ds.Q)
