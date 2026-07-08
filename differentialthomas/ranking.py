r"""
Rankings -- port of ``ranking`` from the DifferentialThomas Maple package,
adapted over ``sage_differential_polynomial``.

A :class:`Ranking` mirrors the reference RankingTable:

- ``compare(a, b)``       -- the ranking itself: True iff ``a >= b`` on jets;
- ``ranking_string``      -- printable description;
- ``function_to_list``    -- head -> indicator list (see ``DiffVarToList``);
- ``ivar`` / ``dvar``     -- independent / dependent variable names;
- ``is_differential_variable`` -- membership test;
- ``standard_form_simplify``   -- the reference's ``numer@normal@simplify``
  (here: clear coefficient denominators; substrate elements are already
  normalised polynomials);
- ``no_deep_copy = True`` -- rankings are SHARED across system copies
  (reference ``rankingtable['NoDeepCopy'] := true``);
- ``ranking_list(a)``     -- the selection-strategy vector;
- ``ring``                -- the substrate ``DifferentialPolynomialRing``
  whose *internal BLAD ranking agrees with* ``compare``.

Substrate agreement (established empirically against the open-maple oracle,
2026-07-08): the reference's default ``DegRevLex`` comparator -- total order
first, then REVERSE lex on the differentiation exponents (smaller last
exponent wins), then the dependent-variable position (earlier in ``dvar`` =
higher) -- is exactly BLAD's ``degrevlexB`` subranking.  BLAD's default
``grlexA`` does NOT match (it breaks equal-order ties by head before
exponents: ``u[y] > v[x]``, where Maple has ``v[x] > u[y]``).  All substrate
rings built here therefore install ``subranking='degrevlexB'``.

Supported ranking specs (Phase 1):

- ``None`` / ``""`` / ``"Degree"`` / ``"DegRevLex"`` -- the default;
- a *block list* ``[[h1, h2], [h3], ...]`` over the dvar heads (the reference
  converts this to a matrix ranking; we build the same matrix) -- elimination
  between blocks, order-graded within;
- ``("Matrix", M)`` with ``M`` a list of rows -- the raw matrix ranking
  (compare ``M.v`` lexicographically).

Deferred: ``EliminateFunction``, ``EliminateFirstIndependentVariable``,
``EliminateIndependentVariables`` (all reduce to the matrix path when needed).

CAVEAT (multi-derivation block/matrix rankings): within a block the reference
matrix breaks equal-order exponent ties *ascending from the last independent
variable* (a ``+1`` row), which is NOT degrevlex; the substrate ring installs
per-block ``degrevlexB``.  For a single derivation the two coincide (all of
ex1-ex3 and the flat-DegRevLex hydrogen ranking are unaffected); a
multi-derivation *block* ranking may disagree on equal-order ties between the
python comparator and the substrate ring's internal leader.  PolynomialObject
cross-asserts its comparator-derived leader against BLAD's on every
polynomial, so a divergence is caught loudly, not silently.
"""

from fractions import Fraction

from .jetvar import JetVar


# module-global ranking, mirroring `DifferentialThomas/GlobalRanking`
_GLOBAL_RANKING = None


def get_global_ranking():
    if _GLOBAL_RANKING is None:
        raise RuntimeError('no ranking set. Call "compute_ranking"')
    return _GLOBAL_RANKING


def set_global_ranking(r):
    global _GLOBAL_RANKING
    _GLOBAL_RANKING = r


# -- list comparators (ports of CompareLex / CompareReverseLex) ---------------

def compare_lex(v, w):
    """` DifferentialThomas/CompareLex`: 1 iff v>w, 0 iff equal, -1 iff v<w."""
    if len(v) != len(w):
        raise ValueError("lists of different length")
    for a, b in zip(v, w):
        if a > b:
            return 1
        if a < b:
            return -1
    return 0


def compare_reverse_lex(v, w):
    """`DifferentialThomas/CompareReverseLex`: scan from the LAST entry."""
    if len(v) != len(w):
        raise ValueError("lists of different length")
    for a, b in zip(reversed(v), reversed(w)):
        if a > b:
            return 1
        if a < b:
            return -1
    return 0


def compare_componentwise(v, w):
    """`DifferentialThomas/CompareComponentwise`: 1 if v dominates w
    componentwise (some entry strictly), -1 as soon as any v[i]<w[i], else 0."""
    if len(v) != len(w):
        raise ValueError("lists of different length")
    result = 0
    for a, b in zip(v, w):
        if a > b:
            result = 1
        elif a < b:
            return -1
    return result


class Ranking(object):
    """The RankingTable of the reference, as a class.  Build via
    :func:`compute_ranking`."""

    # rankings are shared, never deep-copied (reference NoDeepCopy flag)
    no_deep_copy = True

    def __init__(self, ivar, dvar, mode, matrix=None, ranking_string="",
                 blocks=None, base=None):
        self.ivar = list(ivar)
        self.dvar = list(dvar)
        self._mode = mode                  # "degrevlex" | "matrix"
        self._matrix = matrix              # list of row-lists (matrix mode)
        self.ranking_string = ranking_string
        self._blocks = blocks              # substrate block layout (heads)
        self._compare_cache = {}
        self._ranking_list_cache = {}
        # flags consulted by PolynomialObject (reference sets these during
        # ProcInput; defaults here)
        self.linear = False
        self.often_remove_content = False
        self._ring = None
        self._base = base

    # -- substrate ring -------------------------------------------------------

    @property
    def ring(self):
        """The substrate ``DifferentialPolynomialRing`` agreeing with this
        ranking (constructed lazily; one live ring per process -- see the
        substrate's v1 constraint)."""
        if self._ring is None:
            from sage.all import QQ
            from sage_differential_polynomial import DifferentialPolynomialRing
            base = self._base if self._base is not None else QQ
            blocks = self._blocks if self._blocks else [list(self.dvar)]
            self._ring = DifferentialPolynomialRing(
                base, list(self.dvar), list(self.ivar),
                ranking={"blocks": [list(b) for b in blocks],
                         "subranking": "degrevlexB"})
        return self._ring

    # -- reference RankingTable entries ---------------------------------------

    def function_to_list(self, u):
        """``FunctionToList``: dvar head -> indicator list ``[0,..,1,..,0]``."""
        i = self.dvar.index(u)
        return [0] * i + [1] + [0] * (len(self.dvar) - i - 1)

    def is_differential_variable(self, a):
        """``IsDifferentialVariable``: a :class:`JetVar` over this ranking's
        alphabets (also accepts a BLAD-style name string)."""
        if isinstance(a, str):
            try:
                a = JetVar.from_blad_name(a, self.ivar)
            except ValueError:
                return False
        if not isinstance(a, JetVar):
            return False
        return (a.head in self.dvar and len(a.exps) == len(self.ivar)
                and all(e >= 0 for e in a.exps))

    def diff_var_to_list(self, a):
        """`DifferentialThomas/DiffVarToList`: ``u_i[a_1..a_n] ->
        [a_1,..,a_n, 0$(i-1), 1, 0$(m-i)]``; the field sentinel ``1`` maps to
        all zeros."""
        if a == 1:
            return [0] * (len(self.ivar) + len(self.dvar))
        return list(a.exps) + self.function_to_list(a.head)

    def compare(self, a, b):
        """``Compare``: True iff ``a >= b`` under the ranking.  ``a``/``b``
        are :class:`JetVar` or the field sentinel ``1``.

        Reference convention (CompareDegreeReverseLexicographic): ``1`` loses
        to any jet; ``compare(1, 1)`` is False.
        """
        key = (a, b)
        try:
            return self._compare_cache[key]
        except KeyError:
            pass
        if a == 1:
            result = False
        elif b == 1:
            result = True
        elif self._mode == "degrevlex":
            result = self._compare_degrevlex(a, b)
        else:
            v = self._matrix_apply(self.diff_var_to_list(a))
            w = self._matrix_apply(self.diff_var_to_list(b))
            result = compare_lex(v, w) >= 0
        self._compare_cache[key] = result
        return result

    def _compare_degrevlex(self, a, b):
        """`DifferentialThomas/CompareDegreeReverseLexicographic` (a,b jets)."""
        n = len(self.ivar)
        v = self.diff_var_to_list(a)
        w = self.diff_var_to_list(b)
        sumv = sum(v[:n])
        sumw = sum(w[:n])
        if sumv > sumw:
            return True
        if sumv < sumw:
            return False
        c = compare_reverse_lex(v[:n], w[:n])
        if c == -1:
            return True
        if c == 1:
            return False
        return compare_lex(v[n:], w[n:]) >= 0

    def _matrix_apply(self, v):
        return [sum(r * x for r, x in zip(row, v)) for row in self._matrix]

    def ranking_list(self, a):
        """``RankingList``: the selection-strategy vector of a jet.

        DegRevLex form (reference RankingList2):
        ``[total_order, -e_n, ..., -e_2, indicator...]``;
        matrix form: ``M . DiffVarToList(a)``.
        """
        try:
            return self._ranking_list_cache[a]
        except KeyError:
            pass
        v = self.diff_var_to_list(a)
        n = len(self.ivar)
        if self._mode == "degrevlex":
            out = [sum(v[:n])] + [-e for e in reversed(v[1:n])] + v[n:]
        else:
            out = self._matrix_apply(v)
        self._ranking_list_cache[a] = out
        return out

    def standard_form_simplify(self, p):
        """``StandardFormSimplify`` = ``numer @ normal @ simplify``.

        Substrate elements are already normalised expanded polynomials over
        QQ; the surviving effect of ``numer(normal(.))`` is clearing the
        common denominator of the coefficients (real-Maple semantics; note
        that open-maple's numer/normal is a no-op on such input, but the
        substrate cannot hold non-integer rational coefficients anyway --
        see the Phase-1 report -- so clearing is the only representable and
        the mathematically faithful choice).  It does NOT remove integer
        content -- content removal is ``SimplifyPolynom``'s job, later.
        """
        R = self.ring
        p = R(p)
        from sage_differential_polynomial import _blad
        dens = []
        for coeff, _term in _blad.read_terms(p._h()):
            f = Fraction(str(coeff))
            if f.denominator != 1:
                dens.append(f.denominator)
        if not dens:
            return p
        m = 1
        for d in dens:
            m = m * d // _gcd(m, d)
        return p * R(m)

    # -- jet conversions bound to this ranking --------------------------------

    def jetvar_from_blad(self, name):
        return JetVar.from_blad_name(name, self.ivar)

    def blad_name(self, jv):
        return jv.to_blad_name(self.ivar)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# -- matrix construction for block rankings (reference lines 220-241) ---------

def _block_matrix(ivar, blocks):
    """The reference's matrix for a block ranking ``[[..],[..],..]``:

    - rows 1..nblocks: block-membership indicators (elimination between
      blocks, earlier block higher);
    - row nblocks+1: total differentiation order;
    - rows nblocks+2..nblocks+nivar: equal-order tie-break, ascending from the
      LAST independent variable (the reference's ``+1`` convention);
    - remaining rows: within-block dvar position (earlier = higher).
    """
    ni = len(ivar)
    nd = sum(len(b) for b in blocks)
    nb = len(blocks)
    n = ni + nd
    A = [[0] * n for _ in range(n)]
    # ivar rows
    for i in range(1, ni + 1):                      # 1-based i
        A[nb + 1 - 1][i - 1] = 1                    # row nb+1: total order
        if i != 1:
            A[nb + ni - i + 2 - 1][i - 1] = 1       # tie-break rows
    # dvar rows
    k = 0
    for i, blk in enumerate(blocks, start=1):       # 1-based block index
        for j in range(len(blk)):                   # 0-based j as in reference
            col = ni + i + k + j - 1                # dvar column (0-based)
            A[i - 1][col] = 1                       # block indicator row
            if j != len(blk) - 1:
                A[nb + ni + k + j + 1 - 1][col] = 1  # within-block position
        k += len(blk) - 1
    return A


def _validate_global_matrix(M):
    """Reference check: every column's first nonzero entry must be positive
    (a global ranking, not mixed/local)."""
    n = len(M[0])
    for i in range(n):
        col = [row[i] for row in M]
        if compare_lex([0] * len(col), col) >= 0:
            raise ValueError(
                "the matrix describing the ranking has to describe a global "
                "ranking rather than a mixed or local ranking")


def compute_ranking(ivar, dvar, spec=None, set_global=True, base=None):
    """`DifferentialThomas/ComputeRanking` -- build a :class:`Ranking`.

    ``ivar``  -- independent-variable names (list of str).
    ``dvar``  -- dependent-variable names (flat list of str), highest first,
                 or a *block list* (list of lists) which implies the block
                 ranking over its flattening.
    ``spec``  -- ``None``/``""``/``"Degree"``/``"DegRevLex"``; or a block list
                 over the dvar heads; or ``("Matrix", rows)``.
    ``set_global`` -- install as the module-global ranking (the reference
                 always sets `DifferentialThomas/GlobalRanking` unless
                 "return" is passed; pass False for the "return" behaviour).

    Returns the Ranking.
    """
    ivar = [str(x) for x in ivar]
    # dvar given as blocks implies the block ranking (reference line 69-71)
    if dvar and isinstance(dvar[0], (list, tuple)):
        if spec is None:
            spec = [list(b) for b in dvar]
        dvar = [str(h) for b in dvar for h in b]
    else:
        dvar = [str(h) for h in dvar]
    if len(set(dvar)) != len(dvar):
        raise ValueError("duplicate dependent variables")

    blocks = None
    if spec is None or spec in ("", "Degree", "DegRevLex"):
        r = Ranking(ivar, dvar, "degrevlex",
                    ranking_string="DegRevLex", base=base)
    elif isinstance(spec, (list, tuple)) and not (
            len(spec) == 2 and spec[0] == "Matrix"):
        # block ranking: normalise entries to lists, check coverage
        blocks = [list(b) if isinstance(b, (list, tuple)) else [b]
                  for b in spec]
        flat = [h for b in blocks for h in b]
        if flat != dvar:
            raise ValueError(
                "the order of dependent variables in the dvar and block "
                "parameters does not match")
        M = _block_matrix(ivar, blocks)
        _validate_global_matrix(M)
        r = Ranking(ivar, dvar, "matrix", matrix=M,
                    ranking_string="Matrix(blocks=%s)" % (blocks,),
                    blocks=blocks, base=base)
    elif isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "Matrix":
        M = [list(row) for row in spec[1]]
        if any(len(row) != len(ivar) + len(dvar) for row in M):
            raise ValueError("matrix must have one column per (total) variable")
        _validate_global_matrix(M)
        r = Ranking(ivar, dvar, "matrix", matrix=M,
                    ranking_string="Matrixordering", base=base)
    else:
        raise ValueError("ranking not recognised or not yet supported")

    if set_global:
        set_global_ranking(r)
    return r
