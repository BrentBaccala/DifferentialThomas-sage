r"""
Sorting -- port of ``sorting`` from the DifferentialThomas Maple package: the
work-queue ordering that, together with ``strategy``, forms the anti-explosion
layer.

Ported procs
============

- :func:`compare_polynomials_by_equation_then_ranking`
  (`DifferentialThomas/ComparePolynomialsByEquationThenRanking`, the default
  ``CompareStrategy``): the comparator ``a`` before ``b`` in ``Q``.  It enforces
  the strategy invariants (``strategy:18-21``):

    1. equations before inequations of the same leader (so an equation of a
       leader is always chosen before an inequation of that leader);
    2. between two equations, the one with the smaller leader is smaller (so no
       equation ranked below a chosen equation is left behind);

  then finer tie-breaks by rank.

- :func:`insert_into_qlist` (`DifferentialThomas/InsertIntoQList`): insert each
  element of the unsorted ``newQ`` into the sorted ``Q``.

- :func:`sort_qlist` (`DifferentialThomas/SortQList`) = ``insert_into_qlist(Q, [])``.

The ``length``/string tie-break subtree (reached only for two elements with the
SAME (inequation, leader, rank)) is DEGENERATE under the oracle stack:
open-maple has no ``length`` builtin, so ``length(A)=length(B)`` reduces to a
structural-equality test of the two standard forms (empirically verified
2026-07-08), and ``length(A)<length(B)`` / the string ``>`` comparison return
inert / ``FAIL``.  The port therefore models that subtree as: identical
standard forms -> ``True``; otherwise a documented port-side size tie-break
(:func:`_degenerate_tiebreak`) that is NOT oracle-gatable (open-maple cannot
produce a boolean there).  All meaningfully-gated branches -- distinct
(inequation, leader) or distinct rank -- return a clean boolean and match the
oracle exactly.

The ``"force?"`` third-argument protocol of ``InsertIntoQList`` is only
exercised by ``ComparePolynomialsByCompareMatrix`` (a non-default comparator,
deferred); the default comparator ignores the extra argument and returns a
plain boolean, so the force machinery collapses -- the port implements the
collapsed (default-comparator) form faithfully.
"""


# ---------------------------------------------------------------------------
# `DifferentialThomas/ComparePolynomialsByEquationThenRanking` (sorting:105)
# ---------------------------------------------------------------------------

def _ancestor_equals_leader(p):
    """``Ancestor(p) = Leader(p)`` as the reference evaluates it: ``Ancestor``
    is a derivation *list*, ``Leader`` is a jet (or the field sentinel ``1``),
    so the two are never structurally equal -- always ``False`` (mirroring
    open-maple's list-vs-name comparison)."""
    return p.ancestor() == p.leader()


def _degenerate_tiebreak(a, b):
    """Port-side size tie-break for the degenerate (same inequation, leader,
    rank, ancestor-class) / different-standard-form case.  NOT oracle-gatable
    (open-maple returns an inert ``length(...) < length(...)`` here).  Uses the
    canonical spaceless printed length then string order, deterministically."""
    sa = str(a.standard_form())
    sb = str(b.standard_form())
    if len(sa) != len(sb):
        return len(sa) < len(sb)
    return not (sa > sb)


def compare_polynomials_by_equation_then_ranking(a, b, *force):
    """`DifferentialThomas/ComparePolynomialsByEquationThenRanking`: True iff
    ``a`` should come before ``b`` in ``Q``.  A trailing ``"force?"`` argument
    (Maple's ``args[3]``) is accepted and ignored, as the reference does."""
    if a.inequation() == b.inequation():
        la, lb = a.leader(), b.leader()
        if la == lb:
            if a.rank() == b.rank():
                ea = _ancestor_equals_leader(a)
                eb = _ancestor_equals_leader(b)
                if (ea and eb) or ((not ea) and (not eb)):
                    # length / sort-string subtree (degenerate; see docstring)
                    if (a.standard_form() - b.standard_form()).is_zero():
                        return True
                    return _degenerate_tiebreak(a, b)
                # ancestor-class differs (dead branch: list != jet always)
                return not ea
            return a.rank() < b.rank()
        # different leaders: smaller leader first
        return not a.ranking.compare(la, lb)
    # equation vs inequation: the equation (Inequation(b) picks a<b when b is
    # the inequation) comes first
    return b.inequation()


# default CompareStrategy, matching the ranking table's default (init:211)
DEFAULT_COMPARE_STRATEGY = compare_polynomials_by_equation_then_ranking


# ---------------------------------------------------------------------------
# `DifferentialThomas/InsertIntoQList` (sorting:64)
# ---------------------------------------------------------------------------

def insert_into_qlist(new_q, q):
    """`DifferentialThomas/InsertIntoQList`: insert every element of the
    (unsorted) list ``new_q`` into the (sorted) list ``q``; return the new
    sorted list.  The comparator is the element's own
    ``Ranking['CompareStrategy']`` (default
    :func:`compare_polynomials_by_equation_then_ranking`).

    Faithful to the reference's non-force algorithm: for each new element,
    place it right after the last existing position it does NOT sort before."""
    result = list(q)
    for elem in new_q:
        assert hasattr(elem, "standard_form"), "table expected"
        if not result:
            result = [elem]
            continue
        cmp = elem.ranking.compare_strategy
        # res2[a] = comparator(elem, result[a], "force?")  (default: a bool)
        # the force machinery (res3, m1, m2) collapses for the default
        # comparator (never returns a (bool, "force") pair): m1 = 0,
        # m2 = len+1, so res2 is the whole list; keep the FALSE entries.
        false_positions = [i for i, other in enumerate(result)
                           if cmp(elem, other, "force?") is False]
        m3 = (false_positions[-1] + 1) if false_positions else 0
        result = result[:m3] + [elem] + result[m3:]
    return result


# ---------------------------------------------------------------------------
# `DifferentialThomas/SortQList` (sorting:291)
# ---------------------------------------------------------------------------

def sort_qlist(q):
    """`DifferentialThomas/SortQList`: sort an unsorted ``Q`` (= insert into an
    empty list)."""
    return insert_into_qlist(q, [])
