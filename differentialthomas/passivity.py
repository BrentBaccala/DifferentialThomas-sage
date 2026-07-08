r"""
Passivity -- port of ``passivity`` from the DifferentialThomas Maple package.

A single proc, :func:`criteria` (`DifferentialThomas/Criteria`), implementing
the involutive (Janet) criteria that let the completion loop *skip* a
prolongation reduction that is provably redundant.  Only the second and third
criteria are supported in the reference (the first is commented out), so only
those are ported.

The two criteria (Gerdt/Blinkov involutive criteria):

- **crit2**: let ``l`` be the element-wise max ("lcm") of the leading
  derivations of the *ancestors* of ``q`` and the ``divisor``.  If ``l`` is
  not equal to the leading derivation of ``q``, the reduction is redundant.
- **crit3**: for some leaf of the divisor's tree whose leading derivation
  ``ld`` satisfies ``l`` strictly dominating both
  ``lcm(Ancestor(q), ld)`` and ``lcm(Ancestor(divisor), ld)`` componentwise,
  the reduction is redundant.

``Ancestor`` returns a derivation *list* (see :mod:`polyobj`); ``LCMList``
(`general.lcm_list`) is the element-wise max; ``CompareComponentwise`` returns
``1`` iff the first list dominates the second (some entry strictly, none less).
"""

from .general import lcm_list
from .ranking import compare_componentwise
from .janet import janet_tree_leafs


def criteria(q, tree_object, divisor):
    """`DifferentialThomas/Criteria` (``passivity:20``): True iff an
    involutive criterion (crit2 / crit3) shows the reduction of ``q`` by
    ``divisor`` w.r.t. ``tree_object`` is redundant."""
    # "lcm" of leading derivations of the ancestors of q and the divisor
    l = lcm_list(q.ancestor(), divisor.ancestor())

    # crit2
    if l != q.leading_derivation():
        return True

    # crit3
    leafs = janet_tree_leafs(tree_object[q.leading_function()])
    for leaf in leafs:
        if (compare_componentwise(
                l, lcm_list(q.ancestor(), leaf.leading_derivation())) > 0
                and compare_componentwise(
                    l, lcm_list(divisor.ancestor(),
                                leaf.leading_derivation())) > 0):
            return True

    return False
