r"""
Port-side reduction trace -- the counterpart of open-maple's
``OPENMAPLE_REDUCTION_TRACE`` (``~/open-maple/src/trace_reduction.go``).

The oracle emits one stderr line per traced proc call **at call exit** (so a
nested ``PseudoRemainder`` line precedes its enclosing
``DifferentialPseudoReduction`` line)::

    [red] #<n> <Proc>(<desc>, ...) -> <desc> | after=<desc>,...

where ``<desc>`` is ``t<terms>`` for a PolynomialObject table (term count of
its ``Polynom``), ``p<terms>`` for a bare value, with a truncated printed
form in ``{...}`` for small operands (<= 8 terms).

The port mirrors this: when tracing is enabled (:func:`start_trace`), each
traced proc appends a :class:`TraceEntry` at exit.  Descriptors are
``(kind, terms, small)`` tuples where ``small`` is the substrate polynomial
itself for small polynomial operands (semantic diffing), the literal string
for flag/name operands, or ``None``.

Term-count convention (matching ``redTermCount``): number of monomials of
the expanded polynomial, with constants -- including 0 -- counting as 1.
"""


TRACE = None            # None = disabled; a list while capturing


class TraceEntry(object):
    __slots__ = ("proc", "args", "result")

    def __init__(self, proc, args, result):
        self.proc = proc
        self.args = list(args)
        self.result = result

    def fingerprint(self):
        """(proc, ((kind, terms), ...), (kind, terms)) -- the comparison key."""
        return (self.proc,
                tuple((k, n) for (k, n, _s) in self.args),
                (self.result[0], self.result[1]))

    def __repr__(self):
        def d(t):
            k, n, s = t
            return "%s%d" % (k, n) if s is None else "%s%d{%s}" % (k, n, s)
        return "[red] %s(%s) -> %s" % (
            self.proc, ", ".join(d(a) for a in self.args), d(self.result))


def start_trace():
    """Begin capturing; returns the live list."""
    global TRACE
    TRACE = []
    return TRACE


def stop_trace():
    """Stop capturing; returns the captured entries."""
    global TRACE
    t = TRACE
    TRACE = None
    return t


_SMALL_TERMS = 8        # open-maple redTraceSmallTerms


def desc_poly(p):
    """Descriptor of a bare substrate polynomial (kind ``p``)."""
    n = p.number_of_terms()
    if n == 0:
        n = 1                       # redTermCount: the constant 0 counts as 1
    return ("p", n, p if n <= _SMALL_TERMS else None)


def desc_obj(obj):
    """Descriptor of a PolynomialObject (kind ``t``)."""
    p = obj.standard_form()
    n = p.number_of_terms()
    if n == 0:
        n = 1
    return ("t", n, p if n <= _SMALL_TERMS else None)


def desc_literal(s):
    """Descriptor of a flag / name operand (bool, string, unassigned name)."""
    return ("p", 1, str(s))


def desc_seq_result(obj):
    """Descriptor of a Maple sequence result ``(object, f, ...)`` --
    ``redTermCount`` of a Seq counts the FIRST item, and the kind stays
    ``p`` (a Seq is not a Table); the printed form is table junk, skipped."""
    p = obj.standard_form()
    n = p.number_of_terms()
    if n == 0:
        n = 1
    return ("p", n, None)


def record(proc, args, result):
    if TRACE is not None:
        TRACE.append(TraceEntry(proc, args, result))
