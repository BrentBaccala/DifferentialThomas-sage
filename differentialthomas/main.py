r"""
Main completion driver -- port of ``main`` from the DifferentialThomas Maple
package: the input handler (:func:`proc_input`), the single completion step
(:func:`do_next_step`), and the public work-queue loop
(:func:`differential_thomas_decomposition`).

This closes the loop begun in Phases 1-4.  Everything below the queue is
already ported (object model, ranking, Janet trees, reduction, split operators,
sorting/strategy, passivity, the DifferentialSystem container); ``main`` is the
control flow that drives them.

Ported procs
============

- :func:`proc_input` (``main:48``): resolve the ranking + option defaults (the
  subset from ``init`` that the implemented knobs cover -- see
  ``@@PACKAGE@@NewOptions``), wrap the equations / inequations into
  :class:`~differentialthomas.polyobj.PolynomialObject`\ s, detect the
  coefficient field (constant / rational -- selecting ``StandardFormSimplify``
  and the ``Linear`` short-circuit), and build the initial one-element
  ``SystemList``.  Also accepts ``eqs == "SystemList"`` to resume from a list of
  systems (the reference's ``Eq2="SystemList"`` branch), used by the lockstep
  test harness.

  **Guarded out** (documented, not ported): ``goon`` / ``store`` /
  ``stopafter`` / ``maxsteptime`` / ``history`` / ``TraceHistory`` /
  ``extrareductors`` / the ``CompareMatrix``/``SortMatrix`` embedding / the
  trig-exp (`simplify`) coefficient field.  The defaults
  (``maxsteptime = -1`` etc.) disable every guarded branch.

- :func:`do_next_step` (``main:382``): one completion step.  Full control flow
  in the module body; mirrors the reference statement-for-statement, including
  the three branch-count controls (eager ``Inconsistent`` prune via
  ``ReduceQListInSystem`` + the strategy field-element prune, the
  ``Criteria`` involutive skip, and ``DifferentialSystemInequationImplied``
  before every inequation split).

- :func:`differential_thomas_decomposition` (``main:253``): the
  ``while SystemList <> [] do`` loop.  Drops ``Inconsistent`` systems at the
  top; on a ``Finished`` system does the final ``DifferentialSystemTailReduction``
  (which may *un*-finish it, re-queuing its parts) before collecting it into the
  result.  ``FactorModuleBasis`` (combinatorial, side-effect-free) is skipped.

Return-value convention: the reference's ``DoNextStep`` returns a Maple sequence
``ReturnInDoNextStepAndSetOrders(DifferentialSystem, result)`` -- the mutated
current system (when it survives) followed by the spawned children, each with
``MaxOrderInSystem`` bumped.  The port returns the same list; the reference
``Reduction`` mutates its ``q`` in place, whereas the port threads ``q`` back
(the substrate returns fresh objects), so ``do_next_step`` rebinds ``q`` from
:func:`~differentialthomas.splitting.reduction`.
"""

import os
import sys
import time

from .general import MAX_ORDER, set_max_order, reset_max_order
from .polyobj import create_polynomial_object, is_differential_field_element
from .ranking import get_global_ranking
from .janet import (janet_divisor_in_trees, insert_into_janet_trees,
                    janet_trees_leafs)
from .sorting import insert_into_qlist
from .strategy import strategy
from .passivity import criteria
from .reduction import reduce_nonlinear_tail_wrt_janet_trees
from .splitting import (reduction, split_by_initial, split_by_squarefree,
                        divide_by_inequation, inequation_lcm)
from .factor import factorize  # noqa: F401  (used indirectly via reduction)
from .system import (
    DifferentialSystem, create_differential_system,
    differential_system_equations, differential_system_inequations,
    differential_system_janet_trees, differential_system_inequation_implied,
    reduce_qlist_in_system, reduce_inequations_in_differential_system,
    differential_system_tail_reduction,
)

INFINITY = float("inf")


# ---------------------------------------------------------------------------
# In-epoch BLAD arena GC (env-tunable; ON by default)
#
# The single-epoch BLAD arena never frees pseudo-division intermediates on its
# own, so a long decomposition retains every polynomial ever built for the life
# of the run (the 42 GB "operand swell" deaths, and the phase-6 RSS shortfall).
# The substrate's ``ring.gc()`` reclaims that arena wholesale in O(1) at a
# quiescent point; we trigger it at the DoNextStep boundary.
#
# - ``DT_ARENA_GC=0``       kill switch (default on).
# - ``DT_ARENA_GC_MB=N``    size trigger: GC when the reclaimable arena on
#                           BLAD's main stack exceeds N MB (default 256).
# - ``DT_ARENA_GC_EVERY=N`` step trigger: GC every N DoNextStep calls
#                           (default 0 = disabled, size trigger only).
# Whichever trigger fires first runs the GC.  Set both to 0 to disable (but
# leave ``DT_ARENA_GC`` unset) is equivalent to the kill switch for the size
# path while keeping the count path off.
# ---------------------------------------------------------------------------

def _arena_gc_config():
    on = os.environ.get("DT_ARENA_GC", "1") != "0"
    every = int(os.environ.get("DT_ARENA_GC_EVERY", "0") or "0")
    mb = float(os.environ.get("DT_ARENA_GC_MB", "256") or "256")
    return on, every, mb


# ---------------------------------------------------------------------------
# branch / RSS instrumentation (env-gated, off by default)
#
# ``DT_BRANCH_TRACE=1`` prints a periodic progress line and a final summary of
# the work-queue trajectory: systems dropped-inconsistent at the top of the
# loop, systems finished (cells collected), total DoNextStep calls, the peak
# work-queue length, and the process peak RSS.  ``DT_BRANCH_TRACE_EVERY=N``
# sets how often (in loop iterations) the progress line is emitted (default
# 500).  This is the acceptance instrumentation for the hydrogen profile.
# ---------------------------------------------------------------------------

def _peak_rss_mb():
    """Process peak RSS in MB (resource.ru_maxrss is KB on Linux)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


class _BranchStats(object):
    __slots__ = ("iters", "steps", "dropped_inconsistent", "finished",
                 "peak_queue", "t0")

    def __init__(self):
        self.iters = 0
        self.steps = 0
        self.dropped_inconsistent = 0
        self.finished = 0
        self.peak_queue = 0
        self.t0 = time.time()

    def line(self, qlen, ncells):
        return ("[dt-branch] iter=%d step=%d queue=%d peakq=%d finished=%d "
                "cells=%d dropped=%d rss=%.0fMB t=%.0fs"
                % (self.iters, self.steps, qlen, self.peak_queue,
                   self.finished, ncells, self.dropped_inconsistent,
                   _peak_rss_mb(), time.time() - self.t0))


# ---------------------------------------------------------------------------
# `DifferentialThomas/ReturnInDoNextStepAndSetOrders` (main:375)
# ---------------------------------------------------------------------------

def _return_and_set_orders(systems):
    """Bump every returned system's ``MaxOrderInSystem`` to at least the max
    order seen since the last reset, then return the list (mirrors
    ``ReturnInDoNextStepAndSetOrders``)."""
    for sys in systems:
        sys.MaxOrderInSystem = max(sys.MaxOrderInSystem,
                                   MAX_ORDER["since_reset"])
    return systems


# ---------------------------------------------------------------------------
# option defaults (`DifferentialThomas/@@PACKAGE@@NewOptions`, init:206)
# ---------------------------------------------------------------------------

def _install_option_defaults(rk):
    """Install the implemented subset of the reference option table
    (``init:206``) onto the ranking.  The strategy hooks (CompareStrategy /
    FillS / SelectionStrategy) are already the defaults on a fresh
    :class:`Ranking`; here we set the boolean / string / numeric knobs that
    ``ProcInput`` resolves from the options table."""
    rk.factor = True                         # Factor
    rk.factor_strong = False                 # FactorStrong
    rk.factor_inequations = True             # FactorInequations
    rk.inequations_not_coprime = False       # InequationsNotCoprime
    rk.inequations_not_squarefree = False    # InequationsNotSquarefree
    rk.tail_reduction = True                 # TailReduction
    rk.reduction_old = False                 # ReductionOld
    rk.reduction_factor = False              # ReductionFactor
    rk.tail_reduction_intermediate = False   # TailReductionIntermediate
    rk.reduce_qlist_in_system = "Inequations"  # ReduceQListInSystem
    rk.often_remove_content = True           # OftenRemoveContent


# ---------------------------------------------------------------------------
# `DifferentialThomas/ProcInput` (main:48)
# ---------------------------------------------------------------------------

def proc_input(eqs, ineqs, ranking=None):
    """`DifferentialThomas/ProcInput`: resolve the ranking + options, wrap the
    equations / inequations into PolynomialObjects, and return the initial
    one-element ``SystemList``.

    ``eqs`` / ``ineqs`` -- lists of substrate polynomials (or anything
    :func:`create_polynomial_object` coerces).  Input is expected already in
    jet form over ``ranking.ring`` (the ``diff`` -> jet conversion, reference
    ``Diff2JetList`` / ``ProcPolynom``, is the substrate ring's parser); the
    caller supplies ring elements.

    ``eqs == "SystemList"`` resumes from ``ineqs`` (a list of
    :class:`DifferentialSystem`), mirroring the reference's ``Eq2="SystemList"``
    branch -- used by the lockstep harness.

    Returns the ``SystemList`` (a list of :class:`DifferentialSystem`)."""
    if ranking is None:
        ranking = get_global_ranking()

    # resume-from-systems branch (main:169)
    if eqs == "SystemList":
        return list(ineqs)

    _install_option_defaults(ranking)

    eq_objs = [create_polynomial_object(p, ranking, inequation=False)
               for p in eqs]
    ineq_objs = [create_polynomial_object(p, ranking, inequation=True)
                 for p in ineqs]

    set_max_order(eq_objs)
    set_max_order(ineq_objs)

    # coefficient-field detection (main:210-223): if any independent variable
    # appears in a standard form -> `normal` (rational); else the coefficients
    # are constant.  The trig/exp (`simplify`) case cannot arise over the
    # substrate's QQ polynomial coefficients and is guarded out.  In both
    # representable cases the port's `standard_form_simplify` (clear coefficient
    # denominators) is the faithful action; we only need to set the flags the
    # engine consults.
    ivars_present = _ivars_appear(eq_objs + ineq_objs, ranking)
    ranking.constant_coefficients = not ivars_present

    # Linear short-circuit (main:205): only equations, every rank <= 1 and every
    # total degree in the differential variables <= 1.
    if ineq_objs == [] and _all_linear(eq_objs, ranking):
        ranking.linear = True
    else:
        ranking.linear = False

    ds = create_differential_system(eq_objs + ineq_objs, ranking)
    return [ds]


def _ivars_appear(objs, ranking):
    """True iff any independent-variable symbol appears in any standard form
    (the reference's ``indets(...) intersect IVar <> {}``)."""
    from sage_differential_polynomial import _blad
    ivar = set(ranking.ivar)
    for o in objs:
        p = o.standard_form()
        for _coeff, term in _blad.read_terms(p._h()):
            for nm, _deg in term:
                head = nm.split("[", 1)[0]
                if head in ivar:
                    return True
    # also: independent variables can appear as coefficient symbols; the
    # substrate holds them inside the coefficient, invisible to read_terms'
    # jet split, so scan the printed form as a fallback.
    for o in objs:
        s = str(o.standard_form())
        for x in ivar:
            import re
            if re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_\[])" % re.escape(x),
                         s):
                return True
    return False


def _all_linear(objs, ranking):
    """The ``Linear`` predicate (main:205): every element has ``Rank <= 1`` and
    total degree ``<= 1`` in its differential variables."""
    for o in objs:
        if max(1, o.rank()) != 1:
            return False
        p = o.standard_form()
        dvl = o.diff_var_list()
        if not dvl:
            continue
        # total degree in the differential variables
        deg = _total_degree_in(p, dvl, ranking)
        if deg > 1:
            return False
    return True


def _total_degree_in(p, dvl, ranking):
    from sage_differential_polynomial import _blad
    names = set(ranking.blad_name(v) for v in dvl)
    best = 0
    for _coeff, term in _blad.read_terms(p._h()):
        d = sum(int(deg) for nm, deg in term if nm in names)
        best = max(best, d)
    return best


# ---------------------------------------------------------------------------
# `DifferentialThomas/DoNextStep` (main:382)
# ---------------------------------------------------------------------------

def do_next_step(ds, extrareductors=()):
    """`DifferentialThomas/DoNextStep`: one completion step on ``ds``.

    Mutates ``ds`` in place and returns the list of systems replacing it in the
    work queue: ``[ds] + children`` when ``ds`` survives the step, or just the
    spawned children when ``ds`` becomes inconsistent mid-step.  See the module
    docstring for the return convention."""
    rk = ds.Ranking
    R = rk.ring
    result = []

    reset_max_order()
    set_max_order(ds.Q)

    reduce_qlist_in_system(ds)
    if ds.Inconsistent:
        return _return_and_set_orders(result)

    if ds.Q != []:
        # --- normal run: pick and treat one element --------------------------
        indexq = strategy(ds)                       # 1-based
        if indexq == 0:
            return _return_and_set_orders([ds] + result)
        q = ds.Q[indexq - 1]
        ds.Q = ds.Q[:indexq - 1] + ds.Q[indexq:]
        set_max_order(q)

        # equation criteria (involutive skip)
        if q.equation():
            divisor = janet_divisor_in_trees(
                differential_system_janet_trees(ds), q)
            if divisor is not None and criteria(q, ds.JanetTrees, divisor):
                q = create_polynomial_object(R.zero(), rk)

        # implied-inequation heuristic
        if q.inequation() and differential_system_inequation_implied(ds, q):
            q = create_polynomial_object(R.zero(), rk)

        children, q = reduction(ds, q)
        result += children
        set_max_order(q)

        if ds.Inconsistent:
            return _return_and_set_orders(result)

        # remove content; recheck nonzero-initial / squarefreeness
        q.simplify_polynom(force=True)
        if q.f.get("NonZeroInitial") is False:
            q.f.pop("NonZeroInitial", None)
        if q.f.get("Squarefree") is False:
            q.f.pop("Squarefree", None)

        if q.inequation() and differential_system_inequation_implied(ds, q):
            q = create_polynomial_object(R.zero(), rk)

        if q.equation():
            _handle_equation(ds, q, result, R, rk)
        else:
            _handle_inequation(ds, q, result, R, rk)

    else:
        # --- try to finish the system ----------------------------------------
        ineqs = ds.Inequations
        n = len(ineqs)
        i = 0
        while i < n and ineqs[i].nonzero_initial():
            i += 1
        if i < n:
            result += split_by_initial(ds, ineqs[i])
        else:
            i = 0
            while i < n and ineqs[i].squarefree():
                i += 1
            if i < n:
                result += split_by_squarefree(ds, ineqs[i])
            else:
                ds.Finished = True

    return _return_and_set_orders([ds] + result)


def _handle_equation(ds, q, result, R, rk):
    """The equation dispatch of ``DoNextStep`` (main:461-490)."""
    if q.standard_form().is_zero():
        return                                   # tautology 0 = 0; discard q
    if is_differential_field_element(q):
        ds.Inconsistent = True
        return
    if not q.nonzero_initial():
        result += split_by_initial(ds, q)
        ds.Q = insert_into_qlist([q], ds.Q)
    elif not q.squarefree():
        q.simplify_polynom(force=True)
        result += split_by_squarefree(ds, q)
        ds.Q = insert_into_qlist([q], ds.Q)
    else:
        l = insert_into_janet_trees(
            differential_system_janet_trees(ds), q)
        ds.Q = insert_into_qlist(l, ds.Q)
        reduce_inequations_in_differential_system(ds, q)


def _handle_inequation(ds, q, result, R, rk):
    """The inequation dispatch of ``DoNextStep`` (main:492-535)."""
    if q.standard_form().is_zero():
        ds.Inconsistent = True
        return
    if is_differential_field_element(q):
        return                                   # nonzero field element; drop q

    trees = differential_system_janet_trees(ds)
    p = janet_divisor_in_trees(trees, q)
    if p is not None:
        # divide the tree equation p by the inequation q
        p = p.copy()
        deg = p.rank()
        leader = p.leader()
        result += divide_by_inequation(ds, p, q)
        if deg != p.rank() and rk.tail_reduction:
            deg = p.rank()
            p = reduce_nonlinear_tail_wrt_janet_trees(trees, p)
        if deg != p.rank() or leader != p.leader():
            ds.Inconsistent = True
        else:
            insert_into_janet_trees(trees, p)
    elif not q.nonzero_initial():
        result += split_by_initial(ds, q)
        ds.Q = insert_into_qlist([q], ds.Q)
    elif not q.squarefree():
        result += split_by_squarefree(ds, q)
        ds.Q = insert_into_qlist([q], ds.Q)
    else:
        same = [a for a in ds.Inequations if a.leader() == q.leader()]
        if same != []:
            result += inequation_lcm(ds, q, same)
        else:
            ds.Inequations = ds.Inequations + [q]


# ---------------------------------------------------------------------------
# `DifferentialThomas/DifferentialThomasDecomposition` (main:253)
# ---------------------------------------------------------------------------

def differential_thomas_decomposition(eqs, ineqs=(), ranking=None,
                                      system_list=None, on_step=None):
    """`DifferentialThomas/DifferentialThomasDecomposition`: the public entry.

    Runs the ``while SystemList <> [] do`` work-queue loop and returns the list
    of finished, consistent :class:`DifferentialSystem` cells.

    ``eqs`` / ``ineqs`` -- input polynomials (see :func:`proc_input`).
    ``system_list`` -- optional pre-built ``SystemList`` (bypasses
    :func:`proc_input`; used by the lockstep harness so the port and the
    reference start from an identical queue).
    ``on_step`` -- optional callback ``on_step(system_list)`` invoked after each
    ``DoNextStep`` (before the next iteration's top-of-loop drop), for the
    lockstep parity check.

    Guarded out (see module docstring): history / store / goon / stopafter /
    timelimit.  With the defaults these branches are never taken."""
    if system_list is None:
        system_list = proc_input(eqs, ineqs, ranking)
    else:
        system_list = list(system_list)

    result = []

    trace = os.environ.get("DT_BRANCH_TRACE")
    stats = _BranchStats() if trace else None
    every = int(os.environ.get("DT_BRANCH_TRACE_EVERY", "500") or "500")

    gc_on, gc_every, gc_mb = _arena_gc_config()
    gc_steps = 0
    gc_ring = None
    if gc_on and system_list:
        gc_ring = system_list[0].Ranking.ring
        if gc_mb > 0:
            # Arm the ring-level size trigger: ring.maybe_gc() now fires from
            # the hot pseudo-division loop (reduction.pseudo_remainder) and the
            # tail-reduction endgame too, not only at the step boundary.
            gc_ring.set_gc_threshold(int(gc_mb * 1024 * 1024))

    while system_list != []:
        cur = system_list[0]

        if stats is not None:
            stats.iters += 1
            if len(system_list) > stats.peak_queue:
                stats.peak_queue = len(system_list)

        if cur.Inconsistent:
            system_list = system_list[1:]
            if stats is not None:
                stats.dropped_inconsistent += 1
        elif cur.Finished:
            # FactorModuleBasis is combinatorial and side-effect free -> skip.
            differential_system_tail_reduction(cur)
            if cur.Finished:            # tail reduction may have un-finished it
                if not cur.Inconsistent:
                    result.append(cur)
                    if stats is not None:
                        stats.finished += 1
                system_list = system_list[1:]
            # tail reduction allocates like a full step; GC its garbage too
            if gc_ring is not None:
                gc_ring.maybe_gc()
        else:
            new = do_next_step(cur)
            # SystemList := [op(new), op(2..nops(SystemList),SystemList)]
            system_list = list(new) + system_list[1:]
            if stats is not None:
                stats.steps += 1

            # --- in-epoch arena GC at the DoNextStep boundary (quiescent) ----
            if gc_on:
                if gc_ring is None:
                    gc_ring = cur.Ranking.ring
                gc_steps += 1
                if gc_every and gc_steps % gc_every == 0:
                    gc_ring.gc()
                elif gc_ring is not None:
                    gc_ring.maybe_gc()

        if stats is not None and (stats.iters % every == 0):
            sys.stderr.write(stats.line(len(system_list), len(result)) + "\n")
            sys.stderr.flush()

        if on_step is not None:
            on_step(system_list)

    if stats is not None:
        ngc = gc_ring.gc_count if gc_ring is not None else 0
        sys.stderr.write("[dt-branch] DONE " + stats.line(0, len(result))
                         + (" arena_gc=%d" % ngc) + "\n")
        sys.stderr.flush()

    return result


# ---------------------------------------------------------------------------
# public output wrappers (reference `Equations` / `Inequations`)
# ---------------------------------------------------------------------------

def equations(ds):
    """Public ``Equations``: the standard forms of a finished cell's equations
    (tree leaves + equation-typed ``Q`` elements)."""
    return differential_system_equations(ds)


def inequations(ds):
    """Public ``Inequations``: the standard forms of a finished cell's
    inequations."""
    return differential_system_inequations(ds)
