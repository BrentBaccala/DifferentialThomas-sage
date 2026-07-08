r"""
Jet (differential) variables -- port of ``differentialvariables`` (and the
input half of ``conversion``) from the DifferentialThomas Maple package.

Representation decision
=======================

The Maple reference represents a differential variable as an *indexed name*
``u[e1, ..., en]`` whose indices are the differentiation exponents with respect
to the ranking's independent variables ``[x1, ..., xn]`` (in order).  E.g. with
``ivar = [x, y]``, ``u[1, 2]`` is `\partial_x \partial_y^2 u`.

The substrate (``sage_differential_polynomial`` / BLAD) names the same jet by
*derivation names*: BLAD ``u[x,y,y]``, Sage ``u_x_y_y``.

The port's canonical representation is :class:`JetVar` -- an immutable
``(head, exps)`` pair mirroring the reference's exponent-list form, because

- every reference comparator (``DiffVarToList``, ``CompareDegreeReverse-
  Lexicographic``, the matrix rankings) consumes the exponent list directly;
- ``LCMList`` / Janet-tree arithmetic (Phase 2) is exponent arithmetic;
- the open-maple oracle serialises jets in exactly this form (``u[1, 2]``),
  so parity diffing needs no reverse translation.

Conversions to/from the substrate's BLAD names are provided here; the oracle
normaliser (:mod:`differentialthomas.oracle`) reduces both worlds to the
canonical string form ``u[1,2]`` (Maple lprint1D with spaces stripped).

The reference encodes "element of the differential field" (no differential
variable present) by the literal ``1``; the port keeps that convention:
functions that may return a jet variable or "field element" return either a
:class:`JetVar` or the Python integer ``1``.
"""


class JetVar(object):
    """An immutable differential variable ``head[e1,...,en]``.

    ``head`` -- the differential indeterminate's name (a str).
    ``exps`` -- tuple of non-negative ints, one per independent variable of
    the ranking, in ranking ``ivar`` order.
    """

    __slots__ = ("head", "exps")

    def __init__(self, head, exps):
        object.__setattr__(self, "head", str(head))
        object.__setattr__(self, "exps", tuple(int(e) for e in exps))

    def __setattr__(self, *a):
        raise AttributeError("JetVar is immutable")

    def __eq__(self, other):
        return (isinstance(other, JetVar)
                and self.head == other.head and self.exps == other.exps)

    def __hash__(self):
        return hash((self.head, self.exps))

    def __repr__(self):
        # canonical (spaceless) Maple jet form
        return "%s[%s]" % (self.head, ",".join(str(e) for e in self.exps))

    # -- reference accessors -------------------------------------------------

    @property
    def order(self):
        """Total differentiation order (`DifferentialVariableOrder`)."""
        return sum(self.exps)

    # -- conversions ----------------------------------------------------------

    def to_blad_name(self, derivations):
        """The substrate/BLAD jet name: ``u[x,x,y]`` (order-zero: ``u``).

        ``derivations`` -- the ranking's independent-variable names, in the
        same order as ``self.exps``.
        """
        if len(derivations) != len(self.exps):
            raise ValueError("derivation list does not match exponent arity")
        parts = []
        for d, e in zip(derivations, self.exps):
            parts.extend([d] * e)
        if not parts:
            return self.head
        return "%s[%s]" % (self.head, ",".join(parts))

    @staticmethod
    def from_blad_name(name, derivations):
        """Parse a BLAD jet name ``u[x,x,y]`` (or bare head ``u``) into a
        :class:`JetVar` with exponents aligned to ``derivations`` order.
        """
        name = str(name)
        if "[" not in name:
            return JetVar(name, (0,) * len(derivations))
        head, rest = name.split("[", 1)
        ders = rest.rstrip("]").split(",")
        counts = {}
        for d in ders:
            d = d.strip()
            counts[d] = counts.get(d, 0) + 1
        unknown = set(counts) - set(derivations)
        if unknown:
            raise ValueError("unknown derivations %s in %r" % (unknown, name))
        return JetVar(head, tuple(counts.get(d, 0) for d in derivations))

    @staticmethod
    def from_maple(s):
        """Parse the Maple/lprint1D jet form ``u[1, 2]`` (or ``u[1,2]``)."""
        s = str(s).strip()
        if "[" not in s:
            raise ValueError("not an indexed Maple jet: %r" % (s,))
        head, rest = s.split("[", 1)
        exps = [int(t) for t in rest.rstrip("]").split(",")]
        return JetVar(head.strip(), exps)


# -- proc-level ports of `differentialvariables` ------------------------------

def differential_variable_function(u):
    """`DifferentialThomas/DifferentialVariableFunction`: ``u[a,b,..] -> u``;
    the field-element sentinel ``1`` maps to ``1``."""
    if u == 1:
        return 1
    return u.head


def differential_variable_derivation(u):
    """`DifferentialThomas/DifferentialVariableDerivation`:
    ``u[a,b,..] -> [a,b,..]``."""
    return list(u.exps)


def differential_variable_order(u):
    """`DifferentialThomas/DifferentialVariableOrder`: total order."""
    return sum(u.exps)
