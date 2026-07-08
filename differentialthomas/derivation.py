r"""
Derivations -- port of ``derivation`` from the DifferentialThomas Maple
package.

The raw differentiation is the substrate's total derivative
(``DifferentialPolynomial.differentiate``); what this module adds is the DT
bookkeeping that Janet-tree completion relies on:

- **caching**: the partial derivative of a :class:`PolynomialObject` w.r.t. an
  independent variable ``x`` is computed once and stored on the parent under
  the field ``PartialDerivative_<x>`` (reference ``derivation:54-57``).
  Repeated calls return the *same* object, so flags set on a prolongation
  (``NonZeroInitial``, ``ConsideredProlongations``, multiplicative variables
  once inserted into a tree) persist across call sites.

- **Ancestor threading** (``derivation:58``): the derivative inherits the
  parent's ``Ancestor`` (which itself defaults to the parent's leading
  derivation on first access), so a chain of prolongations all point back to
  the original, underived element.  ``Ancestor != LeadingDerivation`` is how
  the engine recognises derived elements (whose initial is the ancestor's
  separant, hence known nonzero) and skips redundant work.

- **Leader propagation** (``derivation:59``): the derivative's leader is set
  directly to the parent's leader prolonged by ``x`` (no substrate scan);
  valid because the separant of a positive-rank element is a nonzero
  polynomial.

- **NonZeroInitial** (``derivation:60``): if the parent is known squarefree,
  the derivative's initial (= the parent's separant) is flagged nonzero.

- **re-consideration on cache hits** (``derivation:61-66``): if the
  derivative is *already* cached, the parent's ``ConsideredProlongations[x]``
  is reset to ``false``.  This handles the integrability-check ordering
  problem documented in the reference (an element inserted with ``x``
  multiplicative while ``d/dx p`` is already in the Q list; if ``x`` later
  turns non-multiplicative the prolongation must be re-examined -- see the
  Gerdt/Blinkov example ``u[1,1,3]-u[4,0,0], u[5,1,0]-u[0,4,0], u[0,6,0],
  u[4,2,0]`` cited there, which is oracle-gated as ``exG`` in
  ``tests/phase2_worker.py``).
"""

from .jetvar import JetVar
from .polyobj import PolynomialObject, create_polynomial_object
from .ranking import get_global_ranking


def partial_derivative(pp, *variables, **kw):
    """`DifferentialThomas/PartialDerivative` (``derivation:27-74``).

    ``pp`` -- a :class:`PolynomialObject` (the cached, metadata-threading
    path) or a substrate polynomial / coercible (plain differentiation).
    ``*variables`` -- independent-variable names, applied left to right
    (the reference's ``x, further_variables``).
    ``ranking`` -- keyword-only; used for the plain-polynomial path when
    ``pp`` is not a PolynomialObject (reference args[3]); defaults to the
    global ranking.

    With no variables, returns ``pp`` unchanged (reference ``nargs=1``).
    """
    ranking = kw.pop("ranking", None)
    if kw:
        raise TypeError("unexpected keyword arguments %s" % sorted(kw))
    if not variables:
        return pp
    p = pp
    if isinstance(p, PolynomialObject):
        rk = p.ranking
    else:
        rk = ranking if ranking is not None else get_global_ranking()
        p = rk.ring(p)                      # the reference's ProcPolynom step
    result = p
    for y in variables:
        y = str(y)
        k = rk.ivar.index(y) + 1            # 1-based (ListTools[Search])
        if isinstance(result, PolynomialObject):
            s = "PartialDerivative_" + y
            if result.f.get(s) is None:
                obj = create_polynomial_object(
                    partial_derivative_internal(result.standard_form(), y),
                    rk)
                # thread the DT metadata (derivation:58-60)
                obj.f["Ancestor"] = list(result.ancestor())
                pl = result.leader()
                assert pl != 1, \
                    "cannot prolong a differential-field element"
                exps = list(pl.exps)
                exps[k - 1] += 1
                obj.f["Leader"] = JetVar(pl.head, exps)
                if result.squarefree():
                    obj.nonzero_initial(True)
                result.f[s] = obj
            else:
                # cache hit: force the parent's x-prolongation to be
                # re-considered (derivation:61-66)
                cp = result.f.get("ConsideredProlongations")
                if cp is None:
                    cp = result.f["ConsideredProlongations"] = {}
                cp[y] = False
            result = result.f[s]
        else:
            result = partial_derivative_internal(result, y)
    return result


def partial_derivative_internal(p, x):
    """`DifferentialThomas/PartialDerivativeInternal` (``derivation:80-111``).

    The reference walks the Maple expression tree applying the sum / product
    / power / jet-variable rules by hand; on the substrate this is exactly
    the total derivative w.r.t. the derivation ``x`` (including the
    ``diff(p, x)`` fallback for the independent variable itself appearing
    polynomially, e.g. ``Vf[0] - V1[0]*x`` in ex3).
    """
    return p.differentiate(str(x))


def multiple_partial_derivative(p, toderive):
    """`DifferentialThomas/MultiplePartialDerivative` (``derivation:116-134``).

    ``toderive`` -- a differentiation multi-index (one entry per independent
    variable).  Applied one step at a time, exhausting each variable in
    ``ivar`` order (the reference's while-loop), so all intermediate
    prolongations are created and cached.
    """
    if not isinstance(p, PolynomialObject):
        raise TypeError("polynom expected as table")
    l = list(toderive)
    result = p
    i = 0
    while i < len(l):
        if l[i] == 0:
            i += 1
        else:
            result = partial_derivative(result, p.ranking.ivar[i])
            l[i] -= 1
    return result
