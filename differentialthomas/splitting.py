r"""
Splitting -- port of the split operators of ``algebraic`` plus the
system-level ``ReduceWithSideEffects`` and ``Reduction`` loop of ``reduction``
from the DifferentialThomas Maple package.

The subresultant / resultant PRS *engine* is provided by the substrate
(``DifferentialPolynomial.subresultants`` -- BLAD's Ducos chain, keyed by degree
in the leader, sign-up-to-parity).  What is ported here is the split *logic*
layered on top:

- :class:`ResultantData` + :func:`initialize_resultant` / :func:`sub_resultant`
  / :func:`prs_gcd` / :func:`co_factor`: the ``InitializeResultantPRS`` /
  ``SubResultantPRS`` / ``PRSGCD`` / ``CoFactorPRS`` hooks (the default
  ``ResultantStrategy = "PRS"``), re-expressed over the substrate chain.
  ``SubResultant(i)`` is the *principal subresultant coefficient* (leading
  coefficient in the leader of the degree-``i`` subresultant); ``PRSGCD(i)`` is
  the degree-``i`` subresultant polynomial (the conditional gcd); ``CoFactor(i)``
  is the pseudo-quotient of ``p`` by that gcd.  All are used either in a
  zero-modulo-system test (scalar-invariant) or become equations that are
  content-simplified (defined up to sign), so the substrate's up-to-parity sign
  is immaterial -- the *zero-pattern* of the chain (which fixes the split index)
  is a scalar-invariant.

- :func:`split_by_initial` (``algebraic:309``): the initial-vanishing split --
  no PRS.  ``initial = 0`` spawns a child with the leading term of ``q``
  dropped (as an equation) alongside the reduced initial; the original gets the
  initial as an inequation.

- :func:`split_by_squarefree` / :func:`split_by_squarefree_old`
  (``algebraic:18/54``): the discriminant/subresultant split making the leader
  squarefree.

- :func:`divide_by_inequation` / :func:`divide_by_inequation_old`
  (``algebraic:206/247``): divide an equation by an inequation via
  subresultants, spawning a child where the subresultant vanishes.

- :func:`inequation_lcm` (``algebraic:116``): coprime-ise / combine inequations.

- :func:`reduce_with_side_effects` (``reduction:452``): conditional-gcd
  reduction of an equation against the trees, spawning a child at the first
  subresultant degree where the gcd degree can jump.

- :func:`reduction` (``reduction:560``): the reduce/``Factorize``/tail-reduce
  loop until the standard form stabilises; may set ``Inconsistent`` and emit
  child systems.

**Deferred**: the linear-algebra resultant path (``ResultantStrategy = "LA"``),
``FactorStrong`` (RootOf).
"""

import math

from .polyobj import (create_polynomial_object, is_differential_field_element,
                      inconsistent_polynom)
from .ranking import compare_componentwise
from .derivation import multiple_partial_derivative
from .janet import (janet_divisor_in_trees, insert_into_janet_trees)
from .reduction import (pseudo_remainder, reduce_wrt_janet_trees,
                        reduce_nonlinear_tail_wrt_janet_trees)
from .sorting import insert_into_qlist
from .system import (differential_system_janet_trees,
                     differential_system_inequation_implied)
from .factor import factorize

INFINITY = math.inf


# ---------------------------------------------------------------------------
# Resultant PRS adapter over the substrate subresultant chain
# ---------------------------------------------------------------------------

class ResultantData(object):
    """The ``ResultantData`` table of ``InitializeResultantPRS`` re-expressed
    over the substrate.  ``chain`` is ``p.subresultants(q, leader)`` -- a dict
    ``{degree_in_leader: subresultant}`` (sign-up-to-parity)."""

    __slots__ = ("rk", "p_obj", "q_obj", "x", "xb", "deg", "chain")

    def __init__(self, p, q):
        self.rk = p.ranking
        self.p_obj = p
        self.q_obj = q
        self.x = p.leader()
        self.xb = self.rk.blad_name(self.x)
        self.deg = p.rank()
        self.chain = p.standard_form().subresultants(q.standard_form(), self.xb)


def initialize_resultant(ds, p, q):
    """`DifferentialThomas/InitializeResultantPRS` (``algebraic:394``)."""
    return ResultantData(p, q)


def sub_resultant(res, i):
    """`DifferentialThomas/SubResultantPRS` (``algebraic:457``): the principal
    subresultant coefficient of degree ``i`` (1 at the top degree, 0 where the
    chain skips degree ``i``)."""
    R = res.rk.ring
    if i == res.deg:
        return R.one()
    s = res.chain.get(i)
    if s is None:
        return R.zero()
    return s.coefficient_in(res.xb, i)


def prs_gcd(res, i):
    """`DifferentialThomas/PRSGCD` (``algebraic:424``): the degree-``i``
    subresultant polynomial (the conditional gcd)."""
    return create_polynomial_object(res.chain[i], res.rk)


def co_factor(ds, res, i):
    """`DifferentialThomas/CoFactorPRS` (``algebraic:508``): the pseudo-quotient
    of ``p`` by the degree-``i`` conditional gcd (``p`` itself at ``i = 0``,
    ``1`` at the top degree)."""
    R = res.rk.ring
    if i == 0:
        return res.p_obj.standard_form()
    if i == res.deg:
        return R.one()
    s = res.chain[i]
    quo, _rem = res.p_obj.standard_form().pseudo_quo_rem(s, res.xb)
    return quo


# ---------------------------------------------------------------------------
# `DifferentialThomas/SplitByInitial` (algebraic:309)
# ---------------------------------------------------------------------------

def split_by_initial(ds, q):
    """`DifferentialThomas/SplitByInitial`: case-split on whether ``q``'s
    initial vanishes modulo the system.  Returns the list of spawned child
    systems (mutating ``q`` and ``ds`` in place)."""
    rk = q.ranking
    R = rk.ring
    x = q.leader()
    trees = ds.JanetTrees

    ini = reduce_wrt_janet_trees(
        trees, create_polynomial_object(q.initial(), rk))

    if ini.standard_form().is_zero():
        if q.standard_form().is_zero():
            q.nonzero_initial(True)
            return []
        # the reduced initial vanishes but q does not: drop the leading term
        # and recurse
        xb = rk.blad_name(x)
        q.substitute_polynom(R.tail(q.standard_form(), xb), keep=("Inequation",))
        q.simplify_polynom(force=True)
        return split_by_initial(ds, q)

    if differential_system_inequation_implied(ds, ini):
        q.simplify_polynom(force=True)
        q.nonzero_initial(True)
        return []

    # generic case: split initial = 0 vs initial != 0
    xb = rk.blad_name(x)
    q1 = create_polynomial_object(R.tail(q.standard_form(), xb), rk)
    q1.inequation(q.inequation())
    if inconsistent_polynom(q1):
        result = []
    else:
        child = ds.deep_copy()
        if ini.leader() != 1:
            child.Q = insert_into_qlist([ini], child.Q)
        child.Q = insert_into_qlist([q1], child.Q)
        result = [child]

    # initial != 0
    ini2 = ini.copy()
    ini2.inequation(True)
    ds.Q = insert_into_qlist([ini2], ds.Q)
    q.nonzero_initial(True)
    q.simplify_polynom(force=True)
    return result


# ---------------------------------------------------------------------------
# `DifferentialThomas/SplitBySquarefree(Old)` (algebraic:18 / 54)
# ---------------------------------------------------------------------------

def split_by_squarefree(ds, p):
    """`DifferentialThomas/SplitBySquarefree` (``algebraic:18``)."""
    if p.rank() == 1:
        p.squarefree(True)
        return []
    return split_by_squarefree_old(ds, p)


def split_by_squarefree_old(ds, p):
    """`DifferentialThomas/SplitBySquarefreeOld` (``algebraic:54``): make the
    leader of ``p`` squarefree via the discriminant subresultant chain."""
    rk = p.ranking
    result = []

    if rk.inequations_not_squarefree and p.inequation():
        p.squarefree(True)
        return []
    if is_differential_field_element(p):
        p.squarefree(True)
        return []
    if len(p.diff_var_list()) == 1 and rk.factor:
        p.squarefree(True)
        return []

    assert p.nonzero_initial(), "initial might be zero!"

    if p.rank() == 1:
        p.squarefree(True)
        return result

    sep = create_polynomial_object(p.separant(), rk)
    sep.simplify_polynom(force=True)
    res = initialize_resultant(ds, p, sep)
    i = 0
    s = None
    while i <= p.rank() - 1:
        s = reduce_wrt_janet_trees(
            ds.JanetTrees, create_polynomial_object(sub_resultant(res, i), rk))
        if not rk.reduction_system(s).standard_form().is_zero():
            break
        i += 1
    # no non-vanishing subresultant found -> inconsistent branch
    if i > p.rank() - 1:
        ds.Inconsistent = True
        return []

    if (not differential_system_inequation_implied(ds, s)
            and i < p.rank() - 1):
        child = ds.deep_copy()
        child.Q = insert_into_qlist([s, p.copy()], child.Q)
        s2 = s.copy()
        s2.inequation(True)
        ds.Q = insert_into_qlist([s2], ds.Q)
        result = [child]

    p.substitute_polynom(co_factor(ds, res, i),
                         keep=("Leader", "LeadingDerivation", "LeadingFunction",
                               "Inequation", "NonZeroInitial"))
    p.squarefree(True)
    return result


# ---------------------------------------------------------------------------
# `DifferentialThomas/DivideByInequation(Old)` (algebraic:206 / 247)
# ---------------------------------------------------------------------------

def divide_by_inequation(ds, p, q):
    """`DifferentialThomas/DivideByInequation` (``algebraic:206``)."""
    return divide_by_inequation_old(ds, p, q)


def divide_by_inequation_old(ds, p, q):
    """`DifferentialThomas/DivideByInequationOld` (``algebraic:247``): divide the
    equation ``p`` by the inequation ``q`` via subresultants, spawning a child
    where the subresultant vanishes."""
    rk = p.ranking
    result = []
    res = initialize_resultant(ds, p, q)
    lim = min(p.rank() - 1, q.rank())
    i = 0
    s = None
    while i <= lim:
        s = reduce_wrt_janet_trees(
            ds.JanetTrees, create_polynomial_object(sub_resultant(res, i), rk))
        if not rk.reduction_system(s).standard_form().is_zero():
            break
        i += 1

    if not differential_system_inequation_implied(ds, s):
        if i < lim:
            child = ds.deep_copy()
            child.Q = insert_into_qlist([q, s], child.Q)
            result = [child]
        s2 = s.copy()
        s2.inequation(True)
        ds.Q = insert_into_qlist([s2], ds.Q)

    p.substitute_polynom(co_factor(ds, res, i),
                         keep=("Leader", "LeadingDerivation", "LeadingFunction",
                               "Inequation", "Squarefree", "NonZeroInitial"))
    return result


# ---------------------------------------------------------------------------
# `DifferentialThomas/InequationLCM` (algebraic:116)
# ---------------------------------------------------------------------------

def inequation_lcm(ds, q, listp2):
    """`DifferentialThomas/InequationLCM`: coprime-ise the inequation ``q``
    against the list ``listp2`` of existing inequation objects (dividing ``q``
    by each via subresultants, spawning children where a subresultant vanishes),
    then merge ``q`` into the system's ``Inequations`` (as a product with any
    same-leader inequation unless ``FactorInequations``).

    Ported for completeness (a main-loop operator).  The optional
    non-generic-gcd pre-sort (``nops(listp) > 2``) is included; the
    ``InequationsNotCoprime`` short-circuit and ``FactorInequations`` merge
    follow the reference defaults."""
    rk = q.ranking
    R = rk.ring
    if rk.inequations_not_coprime:
        ds.Inequations = ds.Inequations + [q]
        return []
    listp = list(listp2)
    leaderq = q.leader()
    q2 = q.copy()
    result = []
    if leaderq == 1:
        return result

    # optional pre-sort of listp by the (reduced) subresultant index, treating
    # non-generic gcds first (reference algebraic:135-152)
    if len(listp) > 2:
        annotated = []
        for pj in listp:
            res = initialize_resultant(ds, q, pj)
            i = 0
            while i <= pj.rank():
                s = reduce_wrt_janet_trees(
                    ds.JanetTrees,
                    create_polynomial_object(sub_resultant(res, i), rk))
                if not rk.reduction_system(s).standard_form().is_zero():
                    break
                i += 1
            annotated.append((pj, i))
        annotated.sort(key=lambda a: a[1], reverse=True)
        listp = [a[0] for a in annotated]

    while listp != [] and leaderq == q.leader():
        p = listp[0]
        listp = listp[1:]
        res = initialize_resultant(ds, q, p)
        i = 0
        s = None
        while i <= p.rank():
            s = reduce_wrt_janet_trees(
                ds.JanetTrees,
                create_polynomial_object(sub_resultant(res, i), rk))
            if not rk.reduction_system(s).standard_form().is_zero():
                break
            i += 1
        q.substitute_polynom(co_factor(ds, res, i), keep=("Inequation",))
        if not differential_system_inequation_implied(ds, s):
            child = ds.deep_copy()
            child.Q = insert_into_qlist([q2.copy(), s], child.Q)
            s2 = s.copy()
            s2.inequation(True)
            ds.Q = insert_into_qlist([s2.copy()], ds.Q)
            result.append(child)

    q.squarefree(True)
    q.nonzero_initial(True)
    if leaderq == q.leader():
        l = [a for a in ds.Inequations if a.leader() == q.leader()]
        if rk.factor_inequations or l == []:
            ds.Inequations = ds.Inequations + [q]
        else:
            prod = q.standard_form()
            for a in l:
                prod = prod * a.standard_form()
            q = create_polynomial_object(prod, rk)
            q.squarefree(True)
            q.nonzero_initial(True)
            q.inequation(True)
            ds.Inequations = [a for a in ds.Inequations
                              if a.leader() != q.leader()] + [q]
    else:
        if not differential_system_inequation_implied(ds, q):
            ds.Q = insert_into_qlist([q], ds.Q)

    # two equal inequations -> inconsistent
    forms = [a.standard_form() for a in ds.Inequations]
    seen = []
    for f in forms:
        if any((f - g).is_zero() for g in seen):
            ds.Inconsistent = True
            break
        seen.append(f)
    return result


# ---------------------------------------------------------------------------
# `DifferentialThomas/ReduceWithSideEffects` (reduction:452)
# ---------------------------------------------------------------------------

def reduce_with_side_effects(ds, q):
    """`DifferentialThomas/ReduceWithSideEffects`: reduce the equation ``q``
    against the Janet trees using conditional-gcd (subresultant) reduction,
    spawning a child system at the first subresultant degree where the gcd
    degree can jump.  Mutates ``q`` and ``ds`` in place; returns child systems."""
    rk = ds.Ranking
    trees = ds.JanetTrees
    result = []
    divisor = janet_divisor_in_trees(trees, q)
    ge = []
    while divisor is not None:
        leader = q.leader()
        mv = divisor.f.get("MultiplicativeVariables")
        if mv is None:
            mv = [INFINITY] * len(rk.ivar)
        toderive = [a - b for a, b in zip(q.leading_derivation(),
                                          divisor.leading_derivation())]
        if (compare_componentwise(mv, toderive) < 0
                or compare_componentwise(toderive, [0] * len(toderive)) == -1):
            break
        reductor = multiple_partial_derivative(divisor, toderive)
        if reductor.rank() == 1:
            q = pseudo_remainder(q, reductor, False)
        else:
            prs = initialize_resultant(ds, reductor, q)
            qcopy = q.copy()
            q = create_polynomial_object(sub_resultant(prs, 0), rk)
            q.simplify_polynom(force=True)
            i = 1
            while i < prs.deg:
                r = reduce_wrt_janet_trees(
                    trees, create_polynomial_object(sub_resultant(prs, i), rk))
                if r.standard_form().is_zero():
                    i += 1
                elif is_differential_field_element(r):
                    ge.append(prs_gcd(prs, i))
                    g = ge[-1]
                    g.equation(True)
                    g.squarefree(True)
                    g.nonzero_initial(True)
                    g.simplify_polynom(force=True)
                    if g.leader() == leader:
                        ds.Q = insert_into_qlist(
                            [a for a in insert_into_janet_trees(trees, g)
                             if a.leader() != reductor.leader()],
                            ds.Q)
                    i = INFINITY
                else:
                    newsystem = ds.deep_copy()
                    ri = r.copy()
                    ri.inequation(True)
                    ds.Q = insert_into_qlist([ri], ds.Q)
                    ge.append(prs_gcd(prs, i))
                    g = ge[-1]
                    g.equation(True)
                    g.squarefree(True)
                    g.nonzero_initial(True)
                    g.simplify_polynom(force=True)
                    if g.leader() == leader:
                        ds.Q = insert_into_qlist(
                            [a for a in insert_into_janet_trees(trees, g)
                             if a.leader() != reductor.leader()],
                            ds.Q)
                    re = r.copy()
                    re.equation(True)
                    qe = q.copy()
                    qe.equation(True)
                    qcopy.equation(True)
                    newsystem.Q = insert_into_qlist(
                        [re, qe, qcopy], newsystem.Q)
                    result.append(newsystem)
                    i = INFINITY
        divisor = janet_divisor_in_trees(trees, q)
    # write the reduced q back into the caller's object
    _q = q
    return result, _q


# ---------------------------------------------------------------------------
# `DifferentialThomas/Reduction` (reduction:560)
# ---------------------------------------------------------------------------

def reduction(ds, q):
    """`DifferentialThomas/Reduction`: reduce/``Factorize``/tail-reduce ``q``
    against the system until its standard form stabilises.  Returns
    ``(children, q)`` where ``q`` is the fully reduced object (the reference
    mutates its argument in place; the port threads it back because
    ``ReduceWithSideEffects`` / ``ReduceWRTJanetTrees`` return fresh objects).
    May set ``ds.Inconsistent``."""
    rk = ds.Ranking
    R = rk.ring
    trees = ds.JanetTrees
    eq = q.equation()
    result = []

    q.simplify_polynom()
    result += factorize(ds, q)

    savedstdform = R.zero()
    inconsistent_return = False
    while not (savedstdform - q.standard_form()).is_zero():
        if is_differential_field_element(q):
            if inconsistent_polynom(q):
                ds.Inconsistent = True
            return result, q
        savedstdform = q.standard_form()
        if rk.reduction_old:
            q = reduce_wrt_janet_trees(trees, q)
        else:
            if eq:
                children, q = reduce_with_side_effects(ds, q)
                result += children
            else:
                q = reduce_wrt_janet_trees(trees, q)

        nresult = len(result)
        if rk.reduction_factor:
            result += factorize(ds, q)
        if nresult != len(result):
            continue
        if not (savedstdform - q.standard_form()).is_zero():
            q.nonzero_initial(False)
        q.simplify_polynom()
        if rk.tail_reduction_intermediate:
            q = reduce_nonlinear_tail_wrt_janet_trees(trees, q, INFINITY)
        # initial reduction: strip the leading term while the reduced initial
        # vanishes modulo the system
        while not q.standard_form().is_zero():
            init = create_polynomial_object(q.initial(), rk)
            init = reduce_wrt_janet_trees(trees, init)
            if rk.reduction_system(init).standard_form().is_zero():
                xb = rk.blad_name(q.leader())
                q = create_polynomial_object(R.tail(q.standard_form(), xb), rk)
                q.equation(eq)
            else:
                break

    if rk.tail_reduction:
        oldrank = q.rank()
        oldleader = q.leader()
        q = reduce_nonlinear_tail_wrt_janet_trees(trees, q, INFINITY)
        if q.leader() != oldleader or oldrank != q.rank():
            ds.Inconsistent = True
            return [], q
        q.simplify_polynom()
        result += factorize(ds, q)
    assert eq == q.equation(), "equation type changed"
    return result, q
