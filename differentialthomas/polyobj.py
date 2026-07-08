r"""
PolynomialObject -- port of ``polynomobjects`` from the DifferentialThomas
Maple package.

The reference models a differential polynomial as a mutable Maple *table*
carrying the polynomial plus lazily-cached derived fields.  The port keeps
that model literally: a :class:`PolynomialObject` holds a single dict
``self.f`` whose keys are exactly the reference's field names:

mandatory:
  ``Polynom``        the polynomial (a substrate ``DifferentialPolynomial``)
                     in standard form
  ``Inequation``     True for an inequation, False for an equation
  ``NonZeroInitial`` the initial is known nonzero w.r.t. the system
  ``Squarefree``     known squarefree w.r.t. the system
  ``Ranking``        the (shared) :class:`~differentialthomas.ranking.Ranking`

lazily cached (invalidation via :meth:`substitute_polynom`):
  ``Leader``, ``LeadingFunction``, ``LeadingDerivation``, ``Initial``,
  ``Separant``, ``Rank``, ``DiffVarList``, ``ConsideredProlongations``,
  ``Ancestor``, ``Factors``, ``Order`` (order of the leader), ``MaxOrder``,
  ``NumberTerms``, ``MultiplicativeVariables``, ``CompareVector`` (Phase 4),
  ``PreReducedForm`` (Phase 3), ``PartialDerivative_<x>`` (Phase 2).

Invalidation semantics (`DifferentialThomas/SubstitutePolynom`): replacing the
polynomial deletes EVERY field except ``Ranking``, ``Polynom`` and the
caller's keep-set -- including ``Inequation`` / ``NonZeroInitial`` /
``Squarefree``, whose accessors then lazily recompute their defaults.  This
"keep-set" contract is load-bearing for the reduction/split code (Phase 3+),
which substitutes reduced polynomials while asserting which cached facts
survive.

Leaders: the leader is computed by the reference's own scan
(``BiggestDiffVar`` over ``DiffVarList`` with ``Ranking.compare``), NOT by
delegating to BLAD's leader -- the port's comparator is the authority.  In
``__debug__`` mode every leader is cross-asserted against the substrate's
BLAD leader (they agree whenever the BLAD leader's head is a dependent
variable; see the ranking module's degrevlexB note), so any ranking divergence
fails loudly.

The reference encodes "no differential variable" (an element of the
differential field) as leader ``1``; the port keeps that sentinel.
"""

from .jetvar import JetVar
from .ranking import get_global_ranking


# fields never deleted by substitute_polynom
_PROTECTED = frozenset(("Ranking", "Polynom"))


class PolynomialObject(object):
    """See module docstring.  Build via :func:`create_polynomial_object`."""

    __slots__ = ("f",)

    def __init__(self, fields):
        self.f = fields

    # -- helpers ---------------------------------------------------------------

    @property
    def ranking(self):
        return self.f["Ranking"]

    def __repr__(self):
        kind = "<>" if self.f.get("Inequation") else "="
        return "PolynomialObject(%s %s 0)" % (self.f["Polynom"], kind)

    # -- `DifferentialThomas/StandardForm` -------------------------------------

    def standard_form(self):
        return self.f["Polynom"]

    # -- `DifferentialThomas/DiffVarList` --------------------------------------

    def diff_var_list(self):
        """All differential variables (JetVars) appearing in the polynomial.

        The reference returns Maple's ``indets`` selection (set order);
        callers never rely on the order.  We return them sorted highest-rank
        first for determinism.
        """
        if self.f.get("DiffVarList") is None:
            rk = self.ranking
            jets = []
            for name in self.f["Polynom"]._jet_names():
                try:
                    jv = JetVar.from_blad_name(name, rk.ivar)
                except ValueError:
                    continue
                if rk.is_differential_variable(jv):
                    jets.append(jv)
            # sort descending by the ranking (insertion sort via compare,
            # mirroring that only `compare` defines the order)
            jets.sort(key=_rank_sort_key(rk), reverse=True)
            self.f["DiffVarList"] = jets
        return self.f["DiffVarList"]

    # -- `DifferentialThomas/BiggestDiffVar` / `Leader` -------------------------

    def _biggest_diff_var(self):
        l = self.diff_var_list()
        if not l:
            return 1
        result = l[0]
        for v in l[1:]:
            if not self.ranking.compare(result, v):
                result = v
        return result

    def leader(self):
        if self.f.get("Leader") is None:
            ld = self._biggest_diff_var()
            if __debug__:
                self._cross_check_leader(ld)
            self.f["Leader"] = ld
        return self.f["Leader"]

    def _cross_check_leader(self, ld):
        """Assert the comparator-derived leader agrees with BLAD's whenever
        BLAD's leader is a dependent-variable jet (see ranking module)."""
        blad_ld = self.f["Polynom"].leader()
        if blad_ld is None:
            return
        rk = self.ranking
        head = blad_ld.split("[", 1)[0]
        if head not in rk.dvar:
            return                      # BLAD ranked a derivation/param top
        assert ld != 1 and rk.blad_name(ld) == blad_ld, (
            "leader divergence: comparator says %s, BLAD says %s for %s"
            % (ld, blad_ld, self.f["Polynom"]))

    # -- `DifferentialThomas/LeadingFunction` / `LeadingDerivation` -------------

    def leading_function(self):
        if self.f.get("LeadingFunction") is None:
            ld = self.leader()
            self.f["LeadingFunction"] = 1 if ld == 1 else ld.head
        return self.f["LeadingFunction"]

    def leading_derivation(self):
        if self.f.get("LeadingDerivation") is None:
            ld = self.leader()
            if ld != 1:
                self.f["LeadingDerivation"] = list(ld.exps)
            else:
                self.f["LeadingDerivation"] = [0] * len(self.ranking.ivar)
        return self.f["LeadingDerivation"]

    # -- `DifferentialThomas/Rank` ----------------------------------------------

    def rank(self):
        """Degree in the leader (0 for a field element)."""
        if self.f.get("Rank") is None:
            ld = self.leader()
            if ld == 1:
                self.f["Rank"] = 0
            else:
                self.f["Rank"] = self.f["Polynom"].degree_in(
                    self.ranking.blad_name(ld))
        return self.f["Rank"]

    # -- `DifferentialThomas/Initial` -------------------------------------------

    def initial(self):
        """Coefficient of the leader at its highest degree (the whole
        polynomial for a field element)."""
        if self.f.get("Initial") is None:
            ld = self.leader()
            if ld == 1:
                self.f["Initial"] = self.standard_form()
            else:
                rk = self.ranking
                self.f["Initial"] = rk.ring.init(
                    self.f["Polynom"], rk.blad_name(ld))
        return self.f["Initial"]

    # -- `DifferentialThomas/Separant` --------------------------------------------

    def separant(self):
        """Partial derivative w.r.t. the leader (0 for a field element)."""
        if self.f.get("Separant") is None:
            ld = self.leader()
            rk = self.ranking
            if ld == 1:
                self.f["Separant"] = rk.ring.zero()
            else:
                self.f["Separant"] = self.f["Polynom"].separant(
                    rk.blad_name(ld))
        return self.f["Separant"]

    # -- `DifferentialThomas/ProlongationConsidered` -------------------------------

    def prolongation_considered(self, x):
        """Side effect: marks the prolongation w.r.t. ivar ``x`` considered;
        returns whether it had already been considered."""
        if self.f.get("ConsideredProlongations") is None:
            self.f["ConsideredProlongations"] = {}
        t = self.f["ConsideredProlongations"]
        if t.get(x) is not True:
            t[x] = True
            return False
        return True

    # -- `DifferentialThomas/Ancestor` ----------------------------------------------

    def ancestor(self):
        """The derivation list of the ancestor (set by PartialDerivative in
        Phase 2 for genuine prolongations; defaults to the leading
        derivation)."""
        if self.f.get("Ancestor") is None:
            # copy: Maple lists are values; the cached LeadingDerivation list
            # must not be aliased by tree/derivation code (Phase-2 extension)
            self.f["Ancestor"] = list(self.leading_derivation())
        return self.f["Ancestor"]

    def set_ancestor(self, deriv_list):
        self.f["Ancestor"] = list(deriv_list)

    # -- `DifferentialThomas/NonZeroInitial` -------------------------------------------

    def nonzero_initial(self, value=None):
        """Getter (no args) / setter (with value), like the reference."""
        if value is not None:
            self.f["NonZeroInitial"] = value
            return self
        if self.f.get("NonZeroInitial") is None:
            rk = self.ranking
            if (rk.linear
                    or create_polynomial_object(self.initial(), rk).leader() == 1
                    or self.ancestor() != self.leading_derivation()):
                # only derivative elements with nonzero separant
                self.f["NonZeroInitial"] = True
            else:
                self.f["NonZeroInitial"] = False
        return self.f["NonZeroInitial"]

    # -- `DifferentialThomas/Squarefree` ------------------------------------------------

    def squarefree(self, value=None):
        if value is not None:
            self.f["Squarefree"] = value
            return self
        if self.f.get("Squarefree") is None:
            self.f["Squarefree"] = bool(self.ranking.linear or self.rank() == 1)
        return self.f["Squarefree"]

    # -- `DifferentialThomas/Equation` / `Inequation` --------------------------------------

    def equation(self, value=None):
        if value is not None:
            self.f["Inequation"] = not value
            return self
        if self.f.get("Inequation") is None:
            self.f["Inequation"] = False
        return not self.f["Inequation"]

    def inequation(self, value=None):
        if value is not None:
            self.f["Inequation"] = value
            return self
        if self.f.get("Inequation") is None:
            self.f["Inequation"] = False
        return self.f["Inequation"]

    # -- `DifferentialThomas/SubstitutePolynom` ----------------------------------------------

    def substitute_polynom(self, newp, keep=()):
        """Replace the polynomial, deleting every cached field not in
        ``keep`` (the reference's ``values`` set).  ``Ranking`` and
        ``Polynom`` always survive."""
        keep = set(keep) | _PROTECTED
        self.f["Polynom"] = newp
        for k in list(self.f.keys()):
            if k not in keep:
                del self.f[k]
        return self

    # -- `DifferentialThomas/Factors` ---------------------------------------------------------

    def factors(self):
        """``[unit, [[factor, multiplicity], ...]]`` like Maple ``factors``.

        A rank-1 nonzero-initial polynomial is its own factorization (the
        reference's shortcut); factors are substrate elements.
        """
        if self.f.get("Factors") is None:
            if self.rank() == 1 and self.nonzero_initial():
                self.f["Factors"] = [1, [[self.standard_form(), 1]]]
            else:
                F = self.standard_form().factor()
                self.f["Factors"] = [F.unit(), [[g, int(m)] for g, m in F]]
        return self.f["Factors"]

    # -- `DifferentialThomas/NumberTerms` --------------------------------------------------------

    def number_terms(self):
        """Number of monomials of the standard form.

        (The reference proc has a latent bug -- it would return
        ``p['Separant']`` from a never-reachable cache branch; the intended
        and effective behaviour is the term count, ported here.)
        """
        if self.f.get("NumberTerms") is None:
            self.f["NumberTerms"] = self.standard_form().number_of_terms()
        return self.f["NumberTerms"]

    # -- `DifferentialThomas/MaxOrder` / `OrderofLeader` -------------------------------------------

    def max_order(self):
        """Maximal total order of any differential variable (0 if none)."""
        if self.f.get("MaxOrder") is None:
            l = self.diff_var_list()
            self.f["MaxOrder"] = max((v.order for v in l), default=0)
        return self.f["MaxOrder"]

    def order_of_leader(self):
        """Total order of the leader (-1 for a field element)."""
        if self.f.get("Order") is None:
            ld = self.leader()
            self.f["Order"] = -1 if ld == 1 else ld.order
        return self.f["Order"]

    # -- DeepCopy (general.deep_copy protocol) ------------------------------------------------------

    def deep_copy(self):
        """Copy the object.  The Ranking is SHARED (NoDeepCopy); substrate
        polynomials, JetVars and scalars are immutable and shared; nested
        tables (ConsideredProlongations) and lists are copied."""
        from .general import deep_copy as _dc
        g = {}
        for k, v in self.f.items():
            g[k] = _dc(v)
        return PolynomialObject(g)


def _rank_sort_key(rk):
    """A total-order key consistent with rk.compare for sorting jet lists."""
    import functools

    def cmp(a, b):
        if a == b:
            return 0
        return 1 if rk.compare(a, b) else -1
    return functools.cmp_to_key(cmp)


# -- `DifferentialThomas/CreatePolynomialObject` -----------------------------------

def create_polynomial_object(p, ranking=None, inequation=False,
                             standard_form=None):
    """Create (idempotently) a :class:`PolynomialObject`.

    ``p`` -- an existing PolynomialObject (returned unchanged, mirroring the
    reference's ``type(p, table)`` branch), a substrate
    ``DifferentialPolynomial``, or anything the substrate ring coerces
    (string / int / Sage rational).

    ``ranking`` -- the Ranking (defaults to the module-global one).
    ``inequation`` -- the reference's trailing ``"<>"`` argument.
    ``standard_form`` -- optional callable overriding
    ``ranking.standard_form_simplify`` (reference args[3]).
    """
    if isinstance(p, PolynomialObject):
        return p
    if ranking is None:
        ranking = get_global_ranking()
    f = {
        "Ranking": ranking,
        "NonZeroInitial": False,
        "Squarefree": False,
        "Inequation": bool(inequation),
    }
    if standard_form is not None:
        f["Polynom"] = standard_form(p)
    else:
        f["Polynom"] = ranking.standard_form_simplify(ranking.ring(p))
    return PolynomialObject(f)


# -- `DifferentialThomas/IsDifferentialFieldElement` ---------------------------------

def is_differential_field_element(p, ranking=None):
    """True iff no differential variable appears (leader is the sentinel 1)."""
    if isinstance(p, PolynomialObject):
        return p.leader() == 1
    return create_polynomial_object(p, ranking).leader() == 1
