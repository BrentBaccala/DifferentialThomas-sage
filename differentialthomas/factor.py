r"""
Factorize -- port of ``factor`` from the DifferentialThomas Maple package
(the weak-factorization split).

`DifferentialThomas/Factorize`(DS, q) factors ``q`` and, for a non-trivial
factorization ``q = f1 * f2`` (``f2`` the product of all but the largest-leader
factor), spawns one child system per case of the resulting disjunction:

- **equation** ``q = 0``: the original ``q`` becomes ``f1`` (``= 0``) and a
  child system gets ``f2 = 0`` together with ``f1 <> 0`` (or just ``f2 = 0``
  when ``q`` is squarefree and ``f1``/``f2`` share a leader);
- **inequation** ``q <> 0``: the original keeps ``f1 <> 0`` and ``f2 <> 0`` is
  pushed into its own ``Q`` (no child).

Denominators (only arising with rational factors) are handled by the
denominator-nonzero split: if the product of factor denominators is a genuine
differential polynomial after reduction, a child requires it ``= 0`` while the
original requires it ``<> 0``.  For polynomial inputs the denominator is a
field element and the split is a no-op.

**Deferred**: ``RootOf`` / strong factorization (``FactorStrong``,
``GetRootOfSubsLists``) -- the reference's algebraic-extension path.  Only weak
factorization via the substrate ``.factor()`` is ported; the ``FactorStrong``
branch is documented and not taken (the ranking default is ``false``).
"""

import functools

from .polyobj import create_polynomial_object, is_differential_field_element
from .sorting import insert_into_qlist


def _factor_sorter_cmp(a_obj, b_obj, ds):
    """`DifferentialThomas/FactorSorter` as a cmp: negative iff ``a`` sorts
    before ``b``.  Distinct leaders order by the ranking (LARGER leader first,
    reference ``Compare(Leader(a),Leader(b))``); equal leaders fall back to a
    canonical string order (a port-side stand-in for Maple's ``sort``/string
    -- only reached for same-leader factors, which the gate avoids)."""
    la, lb = a_obj.leader(), b_obj.leader()
    if la != lb:
        # Compare(la, lb) True (a before b) iff la >= lb -> larger leader first
        return -1 if ds.Ranking.compare(la, lb) else 1
    sa, sb = str(a_obj.standard_form()), str(b_obj.standard_form())
    if sa == sb:
        return 0
    return -1 if sa < sb else 1


def factorize(ds, q):
    """`DifferentialThomas/Factorize` (``factor:79``): return the list of child
    systems spawned by factoring ``q`` (mutating ``q`` and ``ds`` in place)."""
    result = []
    rk = ds.Ranking
    R = rk.ring

    if is_differential_field_element(q):
        return result

    # never factorise prolongations (rank-1 derived elements) -- only
    # inequations or "primary" elements (ancestor == leading derivation)
    if not (rk.factor
            and (q.inequation() or q.ancestor() == q.leading_derivation())):
        return result

    if rk.factor_strong:
        # RootOf / strong factorization -- deferred (default is False)
        raise NotImplementedError("FactorStrong (RootOf factorization) is "
                                  "deferred to a later phase")

    # weak factorization: fak = [unit, [[factor, mult], ...]]
    fak = q.factors()
    fak1 = fak[0]
    fak_pairs = [] if fak1 == 0 else list(fak[1])

    # denominator = product of (factor if mult<0 else denom(factor)); for
    # polynomial factors every denominator is 1 -> denominator = 1.
    denom = R.one()
    for g, m in fak_pairs:
        if m < 0:
            denom = denom * g            # (does not occur for polynomials)
    denom_obj = create_polynomial_object(denom, rk)
    from .system import differential_system_janet_trees
    from .reduction import reduce_wrt_janet_trees
    denom_red = reduce_wrt_janet_trees(differential_system_janet_trees(ds),
                                       denom_obj)
    if not is_differential_field_element(denom_red):
        denomsystem = ds.deep_copy()
        denomsystem.Q = insert_into_qlist(
            [denom_red, q.copy()], denomsystem.Q)
        ds.Q = insert_into_qlist(
            [denom_red.copy().inequation(True)], ds.Q)
        result.append(denomsystem)

    # positive-multiplicity factors (numerators), sorted by FactorSorter
    facs = [g for g, m in fak_pairs if m > 0]
    facs_obj = [create_polynomial_object(g, rk) for g in facs]
    facs_obj.sort(key=functools.cmp_to_key(
        lambda a, b: _factor_sorter_cmp(a, b, ds)))

    if len(facs_obj) > 1:
        rest = R.one()
        for o in facs_obj[1:]:
            rest = rest * o.standard_form()
        fak_objs = [create_polynomial_object(facs_obj[0].standard_form(), rk),
                    create_polynomial_object(rest, rk)]
    elif len(facs_obj) == 1:
        fak_objs = [create_polynomial_object(facs_obj[0].standard_form(), rk)]
    else:
        fak_objs = [create_polynomial_object(
            R.zero() if fak1 == 0 else R.one(), rk)]

    # substitute the first factor into q, preserving the reference's keep-set
    # (factor:207-211: {Leader, Squarefree, NonZeroInitial, Inequation} when
    # the leader is unchanged, else only {Inequation})
    if fak_objs[0].leader() == q.leader():
        q.substitute_polynom(fak_objs[0].standard_form(),
                             keep=("Leader", "Squarefree",
                                   "NonZeroInitial", "Inequation"))
    else:
        q.substitute_polynom(fak_objs[0].standard_form(), keep=("Inequation",))

    if len(fak_objs) == 2:
        if q.equation():
            newsystem = ds.deep_copy()
            fak_objs[1].equation(True)
            if (q.squarefree()
                    and fak_objs[0].leader() == fak_objs[1].leader()):
                newsystem.Q = insert_into_qlist([fak_objs[1]], newsystem.Q)
            else:
                fak_objs[0].inequation(True)
                newsystem.Q = insert_into_qlist(
                    [fak_objs[1], fak_objs[0]], newsystem.Q)
            result.append(newsystem)
        else:
            fak_objs[1].inequation(True)
            ds.Q = insert_into_qlist([fak_objs[1]], ds.Q)

    return result
