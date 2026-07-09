r"""
Reduction engine -- port of ``reduction`` from the DifferentialThomas Maple
package (the parts that operate on a polynomial w.r.t. Janet trees /
reductor sets; ``ReduceWithSideEffects`` and the top-level
``Reduction(DifferentialSystem, q)`` loop need a DifferentialSystem and are
Phase 4).

Ported procs
============

- :func:`pseudo_remainder` (``PseudoRemainder``, reduction:370) -- Euclidean
  pseudo-division in the leader of the reductor, with the reference's
  per-step division by ``gcd(lr, lv)`` (both leading coefficients).  The gcd
  normalisation follows the ORACLE stack's conventions (open-maple delegates
  gcd to a Sage subprocess), reproduced exactly by
  :mod:`differentialthomas.maplecas` -- see that module's docstring for the
  monic / primitive-positive rules and why this deliberately diverges from
  real Maple.  Optional cofactor (Maple's 4th-arg write-back) and the
  ``dononlineartail`` multiplier ``f`` (product of the ``lv/g`` factors),
  satisfying the exact invariant  ``f * uu == qq * vv + r``.

- :func:`differential_pseudo_reduction` (``DifferentialPseudoReduction``,
  reduction:26) -- differentiate the reductor up to the polynomial's leader
  (honouring the reductor's multiplicative variables), then pseudo-reduce;
  loops while the leading *function* is unchanged.  In ``nonlineartail``
  mode it returns ``(result, f)`` after the FIRST pseudo-division (the
  reference returns from inside the loop).

- :func:`reduce_wrt_janet_tree` (``ReduceWRTJanetTree``, reduction:320) --
  reduce w.r.t. one per-dvar tree while a Janet divisor of the leading
  derivation exists (rank-guarded at equal leaders).

- :func:`reduce_wrt_janet_trees` (``ReduceWRTJanetTrees``, reduction:108) --
  head reduction across all trees plus the *initial* reduction: if the
  (reduced) initial vanishes modulo the trees (and the ranking's
  ``ReductionSystem``), the leading term is stripped and the loop restarts.

- :func:`reduce_nonlinear_tail_wrt_janet_trees`
  (``ReduceNonLinearTailWRTJanetTrees``, reduction:219) -- full head+tail
  reduction sweeping the differential variables from the top down, with the
  ``denominator`` / ``linearcombination`` / ``final`` modes and the
  ``maxsizemultiplier`` size cap.  The result is substituted INTO the input
  object (reference ``SubstitutePolynom``), keeping the mode's flag set.

Return-value convention: the reference returns Maple sequences whose ``NULL``
members vanish (``return eval(result), f, lc``).  The port mirrors the
observable arities: plain calls return the object; ``nonlineartail`` /
``denominator`` calls return ``(object, f)``; ``linearcombination`` returns
``(object, f, lc)``.

Linear combinations: Maple builds an inert symbolic receipt
``1/f_i * (lc + q * D[..](divisor))``.  The port records the same data
structurally as a list of :class:`LinearCombinationStep` (cofactor ``q``,
derivation multi-index, divisor, reductor, multiplier ``f_i``), verifiable
via :func:`verify_linear_combination`:  ``result == (prod f_i) * p -
sum_i (f_{i+1}*...*f_n) * q_i * reductor_i``.  (Only the user-facing
``DifferentialSystemLinearCombination`` consumes this; the engine does not.)

Size cap: the reference compares Maple ``length`` of the intermediate result
against ``maxsizemultiplier * length(input)`` and abandons the tail
reduction (returning the input untouched) when exceeded.  open-maple has no
``length`` builtin (the condition is inert, hence always false), so the cap
CANNOT be oracle-gated; the port uses ``len(str(standard form))`` as the
size measure -- a documented approximation, unit-tested port-side only.
The default (``maxsizemultiplier = infinity``, DT option
``MaxSizeMultiplicator``) disables the cap, and all reference runs to date
(ex1-ex3, hydrogen) use the default.
"""

import math

from .general import set_max_order
from .janet import janet_divisor_in_tree, janet_divisor_in_trees
from .derivation import multiple_partial_derivative
from .polyobj import (PolynomialObject, create_polynomial_object,
                      is_differential_field_element)
from .ranking import compare_componentwise
from .maplecas import oracle_gcd_parts, oracle_div_gcd
from . import ctrace
from . import rtrace

INFINITY = math.inf


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pow(base, exp):
    out = None
    for _ in range(int(exp)):
        out = base if out is None else out * base
    return out


def _parse_tail_args(extra):
    """The reference's ``"tail"`` / ``"nonlineartail"`` argument parsing
    (accepts the bare string or a ``(name, value)`` pair for Maple's
    ``name = value``); returns ``(dotail, dononlineartail)``."""
    dotail = False
    dononlineartail = False
    for a in extra:
        if isinstance(a, tuple) and len(a) == 2:
            name, val = a
        else:
            name, val = a, True
        if name == "tail":
            dotail = val
        elif name == "nonlineartail":
            dononlineartail = val
        else:
            raise ValueError("unknown argument %r" % (a,))
    return dotail, dononlineartail


def _extra_args(dotail, dononlineartail):
    ea = []
    if dotail:
        ea.append("tail")
    if dononlineartail:
        ea.append("nonlineartail")
    return tuple(ea)


def maple_length(p):
    """Size measure for the ``maxsizemultiplier`` cap.  Approximates Maple's
    ``length`` by the printed length of the standard form (see module
    docstring -- the cap is not oracle-gatable, and disabled by default)."""
    return len(str(p))


# ---------------------------------------------------------------------------
# `DifferentialThomas/PseudoRemainder` (reduction:370)
# ---------------------------------------------------------------------------

def pseudo_remainder(uu, vv, dononlineartail=False, cofactor=None):
    """Pseudo-divide ``uu`` by ``vv`` w.r.t. ``vv``'s leader.

    ``cofactor`` -- optional empty list; the pseudo-quotient ``qq`` is
    appended (Maple's 4th-argument write-back), satisfying
    ``f * uu == qq * vv + r`` with ``f`` the full multiplier product.

    Returns the remainder as a fresh PolynomialObject, plus ``f`` when
    ``dononlineartail`` (mirroring the reference's return arity).
    """
    targs = [rtrace.desc_obj(uu), rtrace.desc_obj(vv),
             rtrace.desc_literal("true" if dononlineartail else "false")]
    if cofactor is not None:
        targs.append(rtrace.desc_literal("q"))

    rk = uu.ranking
    R = rk.ring
    f = R.one()

    x = vv.leader()
    v = vv.standard_form()
    dv = vv.rank()
    if v.is_zero():
        raise ZeroDivisionError("division by zero")
    xb = rk.blad_name(x) if x != 1 else None

    r = uu.standard_form()
    if x == uu.leader():
        dr = uu.rank()
    else:
        dr = r.degree_in(xb)
    if dv <= dr:
        lv = vv.initial()
        if dv == 0:
            v = R.zero()
        else:
            v = R.tail(v, xb)               # subs(x^dv = 0, v)
    else:
        lv = R.one()
    qq = R.zero() if cofactor is not None else None
    xgen = R.gen(xb) if xb is not None else None

    _ct = ctrace.pr_call(r, vv.standard_form(), lv) if ctrace.enabled else None
    _it = 0

    while dv <= dr and not r.is_zero():
        lr = r.coefficient_in(xb, dr)
        g0, unit = oracle_gcd_parts(lr, lv, rk)
        lr = oracle_div_gcd(lr, g0, unit, rk)       # MyNormal(lr / g)
        t = v * lr
        if dr != dv:
            t = t * _pow(xgen, dr - dv)             # expand(x^(dr-dv)*v*lr)
        if dr == 0:
            r = R.zero()
        else:
            r = R.tail(r, xb)                       # subs(x^dr = 0, r)
        newf = oracle_div_gcd(lv, g0, unit, rk)     # MyNormal(lv / g)
        r = newf * r - t
        if dononlineartail:
            f = f * newf
        if qq is not None:
            step = lr if dr == dv else lr * _pow(xgen, dr - dv)
            qq = newf * qq + step
        dr = r.degree_in(xb)
        if _ct is not None:
            _it += 1
            ctrace.pr_step(_ct, _it, dr, r, newf)

    if cofactor is not None:
        cofactor.append(qq)
    result = create_polynomial_object(r, rk)
    if dononlineartail:
        rtrace.record("PseudoRemainder", targs, rtrace.desc_seq_result(result))
        return result, f
    rtrace.record("PseudoRemainder", targs, rtrace.desc_obj(result))
    return result


# ---------------------------------------------------------------------------
# `DifferentialThomas/DifferentialPseudoReduction` (reduction:26)
# ---------------------------------------------------------------------------

def differential_pseudo_reduction(p, q, *extra):
    """Reduce ``p`` by ``q`` (and its prolongations, honouring ``q``'s
    multiplicative variables) while ``p``'s leading function equals ``q``'s.

    In ``nonlineartail`` mode returns ``(result, f)`` after the first
    pseudo-division (as the reference does, from inside the loop); if no
    division happens the single unreduced object is returned -- the same
    observable as the reference's collapsed 1-sequence."""
    targs = [rtrace.desc_obj(p), rtrace.desc_obj(q)]
    dotail, dononlineartail = _parse_tail_args(extra)
    for s in _extra_args(dotail, dononlineartail):
        targs.append(rtrace.desc_literal('"%s"' % s))

    set_max_order([p.max_order(), q.max_order()])

    result = p
    mv = q.f.get("MultiplicativeVariables")
    if mv is None:
        mv = [INFINITY] * len(p.ranking.ivar)

    while result.leading_function() == q.leading_function():
        toderive = [a - b for a, b in zip(result.leading_derivation(),
                                          q.leading_derivation())]
        if (compare_componentwise(mv, toderive) < 0
                or compare_componentwise(toderive, [0] * len(toderive)) == -1):
            break
        reductor = multiple_partial_derivative(q, toderive)
        if reductor.leader() != result.leader():
            raise RuntimeError("leader should be the same here")
        # cannot reduce u[[0,0]] by u[[0,0]]^2
        if result.rank() < reductor.rank():
            break
        if dononlineartail:
            result, f = pseudo_remainder(result, reductor, True)
            rtrace.record("DifferentialPseudoReduction", targs,
                          rtrace.desc_seq_result(result))
            return result, f
        result = pseudo_remainder(result, reductor, False)

    rtrace.record("DifferentialPseudoReduction", targs,
                  rtrace.desc_obj(result))
    return result


# ---------------------------------------------------------------------------
# `DifferentialThomas/ReduceWRTJanetTree` (reduction:320)
# ---------------------------------------------------------------------------

def reduce_wrt_janet_tree(rootnode, p, *extra):
    """Reduce ``p`` w.r.t. one per-dvar Janet tree while a divisor of the
    leading derivation exists."""
    dotail, dononlineartail = _parse_tail_args(extra)
    ea = _extra_args(dotail, dononlineartail)

    f = p.ranking.ring.one()
    result = p
    u = p.leading_function()
    q = janet_divisor_in_tree(rootnode, result.leading_derivation())
    while q is not None:
        if (result.leading_derivation() == q.leading_derivation()
                and result.rank() < q.rank()):
            break
        if dononlineartail:
            result, newf = differential_pseudo_reduction(result, q, *ea)
            f = newf * f
        else:
            result = differential_pseudo_reduction(result, q, *ea)
        if result.leading_function() != u:
            break
        q = janet_divisor_in_tree(rootnode, result.leading_derivation())
    if dononlineartail:
        return result, f
    return result


# ---------------------------------------------------------------------------
# `DifferentialThomas/ReduceWRTJanetTrees` (reduction:108)
# ---------------------------------------------------------------------------

def reduce_wrt_janet_trees(treeobject, p, *extra):
    """Head + initial reduction of ``p`` w.r.t. a whole trees object.

    Starts from ``p['PreReducedForm']`` when assigned (the strategy layer's
    cache, Phase 4+).  After the head is irreducible, the initial is
    (recursively) reduced: if it vanishes modulo the trees and the ranking's
    ``ReductionSystem``, the leading term is stripped and the loop restarts.
    The result's Inequation flag is taken from ``p``."""
    dotail, dononlineartail = _parse_tail_args(extra)
    ea = _extra_args(dotail, dononlineartail)

    rk = p.ranking
    R = rk.ring
    f = R.one()

    prf = p.f.get("PreReducedForm")
    result = prf if isinstance(prf, PolynomialObject) else p

    while True:
        old_u = None
        u = result.leading_function()
        while u != old_u:
            old_u = u
            if u != 1 and u in rk.dvar:
                if dononlineartail:
                    result, newf = reduce_wrt_janet_tree(
                        treeobject[u], result, *ea)
                    f = f * newf
                else:
                    result = reduce_wrt_janet_tree(treeobject[u], result, *ea)
            u = result.leading_function()

        result.inequation(p.inequation())

        if result.diff_var_list() == []:
            return (result, f) if dononlineartail else result

        # initial reduction
        oldinitial = create_polynomial_object(result.initial(), rk)
        if is_differential_field_element(oldinitial):
            return (result, f) if dononlineartail else result
        newinitial = reduce_wrt_janet_trees(treeobject, oldinitial)
        if not rk.reduction_system(newinitial).standard_form().is_zero():
            return (result, f) if dononlineartail else result
        # the initial vanishes modulo the system: strip the leading term
        lead = _pow(R.gen(rk.blad_name(result.leader())), result.rank())
        result = create_polynomial_object(
            result.standard_form() - result.initial() * lead, rk)
        result.inequation(p.inequation())


# ---------------------------------------------------------------------------
# `DifferentialThomas/ReduceNonLinearTailWRTJanetTrees` (reduction:219)
# ---------------------------------------------------------------------------

class LinearCombinationStep(object):
    """One pseudo-division step of a ``linearcombination`` tail reduction:
    ``newf * result_i == result_{i-1} * newf ... `` precisely,
    ``result_i == newf_i * result_{i-1} - q * reductor`` with ``reductor``
    the ``toderive``-prolongation of ``divisor``."""

    __slots__ = ("q", "toderive", "divisor", "reductor", "newf")

    def __init__(self, q, toderive, divisor, reductor, newf):
        self.q = q                      # pseudo-quotient (substrate)
        self.toderive = tuple(toderive)
        self.divisor = divisor          # divisor standard form (substrate)
        self.reductor = reductor        # derived divisor standard form
        self.newf = newf                # this step's multiplier (substrate)

    def __repr__(self):
        return ("LinearCombinationStep(D[%s](%s), q=%s, f=%s)"
                % (",".join(str(t) for t in self.toderive), self.divisor,
                   self.q, self.newf))


def verify_linear_combination(p0, result, f, lc, ring):
    """Check the exact certificate of a ``linearcombination`` reduction:
    iterating ``acc := newf_i * acc - q_i * reductor_i`` from ``acc = p0``
    reproduces ``result`` (and ``f == prod newf_i``)."""
    acc = p0
    ftot = ring.one()
    for step in lc:
        acc = step.newf * acc - step.q * step.reductor
        ftot = ftot * step.newf
    return (acc - result).is_zero() and (ftot - f).is_zero()


def reduce_nonlinear_tail_wrt_janet_trees(treeobject, p, *args):
    """Head + tail reduction of ``p`` w.r.t. the trees, sweeping the
    differential variables from the highest down.

    Modes (string args): ``"denominator"`` -- also return the multiplier
    product ``f``; ``"linearcombination"`` -- additionally return the
    :class:`LinearCombinationStep` receipt (implies ``denominator``);
    ``"final"`` -- do NOT reduce w.r.t. the leader (and keep
    ``MultiplicativeVariables`` through the final substitution).  A
    nonnegative numeric arg sets ``maxsizemultiplier`` (default infinity).

    The reduced polynomial is substituted into ``p`` itself (reference
    ``SubstitutePolynom``), keeping {NonZeroInitial, Squarefree, Inequation}
    (+ MultiplicativeVariables when ``final``)."""
    assert not isinstance(treeobject, PolynomialObject), \
        "treeobject wrong"        # reference: not assigned(treeobject['Polynom'])

    final = False
    denominator = False
    linearcombination = False
    have_f = False
    lc = None
    maxsizemultiplier = INFINITY
    for a in args:
        if a == "denominator":
            denominator = True
            have_f = True
        if a == "final":
            final = True
        if a == "linearcombination":
            linearcombination = True
            have_f = True
        if isinstance(a, (int, float)) and not isinstance(a, bool) and a >= 0:
            maxsizemultiplier = a

    if linearcombination:
        lc = []
        denominator = True

    rk = p.ranking
    R = rk.ring
    f = R.one() if have_f else None

    result = p.copy()                   # Maple copy(p) -- shallow
    v = p.leader()
    worked = [v]
    if final:                           # don't reduce w.r.t. the leader here
        v = _biggest_remaining(result, worked, rk)
        worked.append(v)
    while v != 1:
        divisor = janet_divisor_in_trees(
            treeobject,
            create_polynomial_object(R.gen(rk.blad_name(v)), rk))
        if divisor is not None:
            assert divisor.nonzero_initial()
            assert rk.is_differential_variable(divisor.leader()), \
                "no differential variable as leader"
            toderive = [a - b for a, b in
                        zip(v.exps, divisor.leading_derivation())]
            reductor = multiple_partial_derivative(divisor, toderive)

            if linearcombination:
                cof = []
                result, newf = pseudo_remainder(result, reductor, True,
                                                cofactor=cof)
                lc.append(LinearCombinationStep(
                    cof[0], toderive, divisor.standard_form(),
                    reductor.standard_form(), newf))
                f = f * newf
            elif denominator:
                result, newf = pseudo_remainder(result, reductor, True)
                f = f * newf
            else:
                result = pseudo_remainder(result, reductor, False)

            # Dongming's proposal: only tail-reduce while size stays bounded
            if (maxsizemultiplier != INFINITY
                    and maple_length(result.standard_form())
                    > maxsizemultiplier * maple_length(p.standard_form())):
                if denominator:
                    return p, R.one()
                return p

            worked.append(reductor.leader())
        else:
            worked.append(v)

        v = _biggest_remaining(result, worked, rk)

    if final:
        result = p.substitute_polynom(
            result.standard_form(),
            keep=("NonZeroInitial", "Squarefree", "Inequation",
                  "MultiplicativeVariables"))
    else:
        result = p.substitute_polynom(
            result.standard_form(),
            keep=("NonZeroInitial", "Squarefree", "Inequation"))

    # return eval(result), f, lc -- NULL members vanish from the sequence
    if lc is not None:
        return result, f, lc
    if f is not None:
        return result, f
    return result


def _biggest_remaining(result, worked, rk):
    """``BiggestDiffVar`` of ``DiffVarList(result) minus worked`` (the
    reference materialises the set difference as a sum polynomial; the
    maximum under the ranking is the same)."""
    remaining = set(result.diff_var_list()) - set(w for w in worked if w != 1)
    if not remaining:
        return 1
    it = iter(remaining)
    best = next(it)
    for v in it:
        if not rk.compare(best, v):
            best = v
    return best
