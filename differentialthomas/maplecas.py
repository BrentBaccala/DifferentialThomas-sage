r"""
Oracle-faithful CAS conventions -- the gcd / content normalisations of the
open-maple + sage-server reference stack.

Why this module exists
======================

The reference ``DifferentialThomas/PseudoRemainder`` divides each
pseudo-division step by ``gcd(lr, lv)`` (the gcd of the two leading
coefficients) and ``SimplifyPolynom`` divides by ``content(p, V)``.  Our
parity oracle is the LGPL Maple source executed by **open-maple**, which
delegates ``gcd`` / ``content`` to a Sage subprocess
(``~/open-maple/cas/sage_server.py``).  The *normalisation* of those results
is therefore neither real-Maple's nor BLAD's, but Sage's -- and it leaks into
every intermediate polynomial of the reduction trajectory (a unit change in
``gcd(lr, lv)`` rescales the remainder).  Matching the oracle bit-for-bit
requires reproducing the sage-server conventions exactly (established
empirically 2026-07-08 against Sage 10.7):

* the server builds ``PolynomialRing(QQ, sorted(varnames))`` where
  ``varnames`` are the *sanitized* names of all variables appearing in the
  request's arguments (Go side: ``sanitizer.varList``, sorted; indexed jets
  ``u[1,0]`` sanitize to ``u_DT_1_0``, plain identifiers pass through);
* **multivariate** ring (>= 2 vars): Singular gcd == the integer-primitive
  gcd with *positive leading coefficient* w.r.t. the ring's degrevlex order
  over the sorted names (constants normalise to 1);
* **univariate** ring (<= 1 var, incl. the server's ``t_dummy`` fallback for
  constants): FLINT gcd == the *monic* gcd (rational coefficients allowed,
  e.g. ``gcd(4*u-6, 8*u-12) = u - 3/2``);
* ``content(p, V)`` (server ``_content_wrt``): rational content ``rc``
  (gcd of coefficient numerators / lcm of denominators, positive), then the
  iterated Sage gcd of the coefficient groups of ``p/rc`` w.r.t. the
  variables ``V`` -- **except** when there is a single group, where the raw
  (unnormalised, sign-carrying) group coefficient is used -- finally stripped
  of its own rational content (sign kept).

The substrate's BLAD ``gcd`` returns the gcd *over Z* (integer content
included, e.g. ``gcd(2*u_x, 4*u_x) = 2*u_x``), i.e. a different unit.  This
module computes the **normalisation unit** relating the two so that the
reduction engine can stay in exact BLAD integer arithmetic:

    ``sage_gcd = blad_gcd / unit``,  ``a / sage_gcd = a.exquo(blad_gcd) * unit``

with ``unit`` an integer (the leading coefficient for the monic case, the
signed integer content for the primitive-positive case) -- so the quotients
``lr/g`` and ``lv/g`` the reduction needs are integer polynomials and no
rational coefficients ever materialise.

NOTE this is a deliberate divergence from *real Maple* (whose gcd keeps
integer content, like BLAD): the verified reference decompositions (hydrogen
29 cells etc.) were produced by the open-maple + sage-server stack, so that
stack's conventions are the parity target.
"""

from fractions import Fraction
from math import gcd as _igcd

from .jetvar import JetVar


# ---------------------------------------------------------------------------
# sanitized names / term data
# ---------------------------------------------------------------------------

def sanitized_name(blad_name, rk):
    """The open-maple sage-server sanitized name of a substrate variable.

    Jets of the ranking's dependent variables map to the sanitizer's indexed
    form ``head_DT_e1_e2...`` (Maple ``u[e1,e2]``); anything else (an
    independent variable appearing polynomially, e.g. ``x``) passes through
    (Go ``sanitizeName`` keeps plain identifiers)."""
    head = blad_name.split("[", 1)[0]
    if head in rk.dvar:
        jv = JetVar.from_blad_name(blad_name, rk.ivar)
        return "%s_DT_%s" % (head, "_".join(str(e) for e in jv.exps))
    return blad_name


def term_data(p, rk):
    """``[(Fraction coeff, {sanitized_name: degree})]`` for a substrate
    polynomial (one entry per monomial)."""
    from sage_differential_polynomial import _blad
    out = []
    for coeff, term in _blad.read_terms(p._h()):
        d = {}
        for nm, deg in term:
            d[sanitized_name(nm, rk)] = int(deg)
        out.append((Fraction(str(coeff)), d))
    return out


def var_names(p, rk):
    """Set of sanitized variable names appearing in ``p``."""
    return {n for _c, d in term_data(p, rk) for n in d}


# ---------------------------------------------------------------------------
# rational content / degrevlex leading coefficient
# ---------------------------------------------------------------------------

def _lcm(a, b):
    return a * b // _igcd(a, b)


def integer_content(terms):
    """The sage-server ``_content``: gcd of coefficient numerators over lcm
    of denominators, positive (``Fraction``; 0 for the zero polynomial)."""
    g, l = 0, 1
    for c, _d in terms:
        g = _igcd(g, abs(c.numerator))
        l = _lcm(l, c.denominator)
    return Fraction(g, l)


def drl_lead_coeff(terms):
    """Coefficient of the degrevlex-leading monomial w.r.t. the ring
    ``PolynomialRing(QQ, sorted(varnames))`` (Sage's default term order).

    degrevlex: higher total degree wins; on ties the *smaller* exponent on
    the *last* variable wins (classic reverse-lex tie-break)."""
    names = sorted({n for _c, d in terms for n in d})

    def key(t):
        exps = [t[1].get(n, 0) for n in names]
        return (sum(exps), tuple(-e for e in reversed(exps)))

    return max(terms, key=key)[0]


# ---------------------------------------------------------------------------
# gcd with the sage-server normalisation
# ---------------------------------------------------------------------------

def gcd_normalization_unit(g0_terms, ring_nvars, integer_consts=False):
    """The unit ``u`` with ``oracle_gcd = blad_gcd / u`` (see module
    docstring).

    ``g0_terms`` -- term data of the (nonzero) BLAD gcd;
    ``ring_nvars`` -- number of variables across the two operands:
    0 selects the constants rule, 1 the univariate/monic convention,
    >= 2 the multivariate primitive-positive one;
    ``integer_consts`` -- for the 0-var case, whether BOTH operands are
    integers: open-maple's native fast path (``nativeGCD``) -- and the sage
    server's ``{"int": ...}`` encoding alike -- compute the *integer* gcd
    there (``gcd(4, 8) = 4``), while a rational constant routes to the
    Sage ring where every nonzero constant is a unit (gcd = 1, the same as
    the monic rule)."""
    if not g0_terms:
        return Fraction(1)              # gcd(0,0) = 0; unit irrelevant
    lead = drl_lead_coeff(g0_terms)
    if ring_nvars == 0 and integer_consts:
        return Fraction(1) if lead > 0 else Fraction(-1)   # |integer gcd|
    if ring_nvars <= 1:
        return lead                     # monic normalisation
    cont = integer_content(g0_terms)
    return cont if lead > 0 else -cont  # primitive, positive leading coeff


def oracle_gcd_parts(a, b, rk):
    """``(g0, unit)`` with ``g0`` the substrate/BLAD gcd of ``a`` and ``b``
    and ``oracle_gcd = g0 / unit`` the gcd the oracle stack computes."""
    g0 = a.gcd(b)
    ta, tb = term_data(a, rk), term_data(b, rk)
    nv = len({n for _c, d in ta for n in d} | {n for _c, d in tb for n in d})
    ints = all(c.denominator == 1 for c, _d in ta + tb)
    unit = gcd_normalization_unit(term_data(g0, rk), nv, ints)
    return g0, unit


def oracle_div_gcd(a, g0, unit, rk):
    """``a / sage_gcd = a.exquo(g0) * unit`` -- the exact quotient of ``a``
    by the oracle-normalised gcd, staying in integer arithmetic.

    (``unit`` is integral whenever the operands are integer polynomials --
    the only case the reduction engine produces; asserted.)"""
    q = a.exquo(g0)
    if unit == 1:
        return q
    assert unit.denominator == 1, \
        "non-integer gcd normalisation unit %s" % (unit,)
    return q * rk.ring(int(unit))


# ---------------------------------------------------------------------------
# Maple content(p, V) with the sage-server semantics
# ---------------------------------------------------------------------------

def _div_scalar(p, frac, rk):
    """Exact ``p / frac`` for a rational scalar dividing ``p`` exactly."""
    if frac == 1:
        return p
    R = rk.ring
    q = p * R(int(frac.denominator)) if frac.denominator != 1 else p
    return q.exquo(R(int(frac.numerator)))


def _mul_scalar(p, frac, rk):
    """``p * frac`` (exact; the product is integral by construction here)."""
    if frac == 1:
        return p
    R = rk.ring
    q = p * R(int(frac.numerator))
    if frac.denominator != 1:
        q = q.exquo(R(int(frac.denominator)))
    return q


def maple_content(p, V, rk):
    """``content(p, V)`` exactly as the oracle stack computes it (sage-server
    ``op_content`` / ``_content_wrt``; see module docstring).

    ``p`` -- substrate polynomial; ``V`` -- iterable of BLAD variable names
    (or :class:`JetVar`) w.r.t. which the content is taken."""
    R = rk.ring
    if p.is_zero():
        return R.zero()
    Vb = [rk.blad_name(v) if isinstance(v, JetVar) else str(v) for v in V]
    Vs = [sanitized_name(bn, rk) for bn in Vb]

    terms = term_data(p, rk)
    rc = integer_content(terms)                     # rational content, > 0
    b = _div_scalar(p, rc, rk)

    bterms = term_data(b, rk)
    keys = sorted({tuple(d.get(s, 0) for s in Vs) for _c, d in bterms})
    ring_nvars = len({n for _c, d in bterms for n in d} | set(Vs))

    if len(keys) == 1:
        # single coefficient group: the server keeps it raw (sign included,
        # no Singular normalisation -- ``g`` is assigned, never gcd'd)
        mono = R.one()
        for bn, e in zip(Vb, keys[0]):
            gen = R.gen(bn)
            for _ in range(e):
                mono = mono * gen
        g = b.exquo(mono)
    elif ring_nvars <= 1:
        # univariate request ring: the groups' coefficients are rational
        # constants; the FLINT gcd is monic, i.e. 1
        g = R.one()
    else:
        # iterated Singular gcd of the coefficient groups, normalised
        # integer-primitive with positive degrevlex-leading coefficient
        gg = None
        for kk in keys:
            gp = b
            for bn, e in zip(Vb, kk):
                gp = gp.coefficient_in(bn, e)
            gg = gp if gg is None else gg.gcd(gp)
        gt = term_data(gg, rk)
        unit = integer_content(gt)
        if drl_lead_coeff(gt) < 0:
            unit = -unit
        g = _div_scalar(gg, unit, rk)

    # server: gc = _content(g); g = g / gc   (strip rational content,
    # keeping the sign)
    gc = integer_content(term_data(g, rk))
    if gc not in (0, 1):
        g = _div_scalar(g, gc, rk)
    return _mul_scalar(g, rc, rk)
