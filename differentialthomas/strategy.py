r"""
Strategy -- port of ``strategy`` from the DifferentialThomas Maple package.
Together with ``sorting`` this is the anti-explosion layer; its stated purpose
(``strategy:18-21``) is to always choose the next element under conditions that
prevent "newly created special cases [being] created again and again":

  1. there must not be equations ranking lower than a chosen equation;
  2. there must not be an equation of the same leader if we choose an
     inequation.

Ported procs
============

- :func:`remove_leading_field_elements`
  (`DifferentialThomas/RemoveLeadingFieldElements`, ``strategy:23``): the eager
  field-element inconsistency prune.  While the front of ``Q`` is a
  differential-field element: a nonzero equation or a zero inequation there
  makes the whole system inconsistent (``Q`` cleared); a zero equation / nonzero
  inequation is a tautology and is dropped.  Stops at the first genuine
  differential polynomial.

- :func:`fill_s_by_smallest_leader`
  (`DifferentialThomas/FillSBySmallestLeader`, the default ``FillS``): the
  prefix of ``Q`` (after the prune) sharing the smallest leader.

- :func:`strategy_smallest_element`
  (`DifferentialThomas/StrategySmallestElement`, the default
  ``SelectionStrategy``): always pick the first candidate.  (The
  factor/reduced-factor/reduced-element strategies are deferred.)

- :func:`strategy` (`DifferentialThomas/Strategy`): pick the index (1-based,
  into ``Q``) of the next element to treat, or ``0`` if ``Q`` is exhausted.

Indices are 1-based to match the reference (the main loop uses the return value
directly as a ``Q`` index).  ``S`` is always a prefix of ``Q``, so the index
into ``S`` equals the index into ``Q``.
"""

from .polyobj import is_differential_field_element


def remove_leading_field_elements(ds):
    """`DifferentialThomas/RemoveLeadingFieldElements` (``strategy:23``)."""
    leader = 1
    while leader == 1 and len(ds.Q) > 0:
        q0 = ds.Q[0]
        if is_differential_field_element(q0):
            sf = q0.standard_form()
            if ((q0.equation() and not sf.is_zero())
                    or (q0.inequation() and sf.is_zero())):
                ds.Inconsistent = True
                ds.Q = []
                return
            # tautology (0 = 0 equation or nonzero <> 0 inequation): drop it
            ds.Q = ds.Q[1:]
        else:
            leader = q0.leader()


def fill_s_by_smallest_leader(ds):
    """`DifferentialThomas/FillSBySmallestLeader` (``strategy:42``): the ``Q``
    prefix sharing the smallest leader (after the field-element prune)."""
    remove_leading_field_elements(ds)
    if ds.Q == []:
        return []
    leader = ds.Q[0].leader()
    s = []
    i = 0
    while i < len(ds.Q) and ds.Q[i].leader() == leader:
        s.append(ds.Q[i])
        i += 1
    return s


def strategy_smallest_element(ds, s):
    """`DifferentialThomas/StrategySmallestElement` (``strategy:120``): the
    default selection strategy -- always the first candidate (1-based)."""
    return 1


def strategy(ds):
    """`DifferentialThomas/Strategy` (``strategy:126``): the 1-based index into
    ``Q`` of the next element to treat, or ``0`` if ``Q`` is empty."""
    s = ds.Ranking.fill_s(ds)
    if s == []:
        return 0
    if len(s) == 1:
        return 1
    if is_differential_field_element(s[0]):
        return 1
    return ds.Ranking.selection_strategy(ds, s)
