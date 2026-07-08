r"""
Oracle harness: line-level validation against the reference DifferentialThomas
Maple source, executed unmodified by open-maple.

Mechanism
=========

``~/open-maple/src/openmaple <file.mpl>`` interprets a Maple script; ``with
(DifferentialThomas)`` loads the LGPL reference source from ``$DT_SRC``
(default ``~/DifferentialThomas/src``).  Internal procs are directly callable
by their qualified names (`` `DifferentialThomas/CreatePolynomialObject` ``,
`` `DifferentialThomas/Leader` ``, ...), so ANY internal step of the reference
is scriptable.  Results are serialised with ``convert(expr, string)`` --
open-maple's lprint1D (Maple 1-D surface: spaceless operators, indexed names
as ``u[1, 2]``) -- prefixed by a marker via printf:

    printf("OMQ|<i>|%s\n", convert((<expr>), string)):

:class:`DTOracle` builds such a script (ranking setup + polynomial objects +
queries), runs it in one openmaple process, and returns the answer strings in
query order.

The CAS backend is selected by ``OPENMAPLE_CAS=sage`` (a Sage subprocess
serving the polynomial ops open-maple does not implement natively);
``OPENMAPLE_ROOT`` locates ``cas/sage_server.py``.

Reduction traces
================

``OPENMAPLE_REDUCTION_TRACE=<Proc1,Proc2|1>`` makes open-maple emit one stderr
line ``[red] #<n> <Proc>(<operands>) -> <result> | after=...`` per call of the
named reduction procs (term-count fingerprints; small operands carry printed
forms).  :func:`capture_reduction_trace` wraps a scripted run and returns the
trace lines -- the backbone of the Phase-3 reduction-trajectory diff.

Jet-name normalisation
======================

Reference jets are exponent-indexed (``u[1, 2]``); substrate jets are
derivation-named (BLAD ``u[x,y,y]`` / Sage ``u_x_y_y``).  Both worlds
normalise through :class:`~differentialthomas.jetvar.JetVar` and, for whole
polynomials, through Sage's symbolic ring with jets renamed to neutral
symbols ``jet_<head>_<e1>_<e2>`` (:func:`maple_poly_to_sr`,
:func:`dp_to_sr`); equality is then a symbolic-difference zero test
(:func:`poly_strings_equal`).
"""

import os
import re
import subprocess

from .jetvar import JetVar

OPENMAPLE_BIN = os.path.expanduser("~/open-maple/src/openmaple")
OPENMAPLE_ROOT = os.path.expanduser("~/open-maple")
DT_SRC_DEFAULT = os.path.expanduser("~/DifferentialThomas/src")


def _base_env(extra=None):
    env = dict(os.environ)
    env.setdefault("OPENMAPLE_CAS", "sage")
    env.setdefault("OPENMAPLE_ROOT", OPENMAPLE_ROOT)
    env.setdefault("DT_SRC", DT_SRC_DEFAULT)
    if extra:
        env.update(extra)
    return env


def run_openmaple(script_text, env_extra=None, timeout=600):
    """Run a Maple script through openmaple; return (stdout, stderr).

    Raises RuntimeError on a nonzero exit."""
    import tempfile
    with tempfile.NamedTemporaryFile(
            "w", suffix=".mpl", prefix="dtoracle_", delete=False) as fh:
        fh.write(script_text)
        path = fh.name
    try:
        proc = subprocess.run(
            [OPENMAPLE_BIN, path], capture_output=True, text=True,
            timeout=timeout, env=_base_env(env_extra))
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        raise RuntimeError(
            "openmaple failed (rc=%d):\nstdout:\n%s\nstderr:\n%s"
            % (proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:]))
    return proc.stdout, proc.stderr


_OMQ_RE = re.compile(r"^OMQ\|(\d+)\|(.*)$", re.M)


class DTOracle(object):
    """Scripted queries against the reference package.

    ``ivar`` / ``dvar`` -- ranking alphabets (lists of names).
    ``setup``  -- extra Maple statements executed after the ranking is set
                  (each a full statement WITHOUT the trailing ``:``), e.g.
                  ``p1 := `DifferentialThomas/CreatePolynomialObject`(u[1]^2-4*u[0], R)``.
                  The global ranking is bound to ``R``.
    """

    def __init__(self, ivar, dvar, setup=()):
        self.ivar = list(ivar)
        self.dvar = list(dvar)
        self.setup = list(setup)

    def _script(self, queries):
        lines = ["with(DifferentialThomas):"]
        lines.append("`DifferentialThomas/ComputeRanking`([%s],[%s]):"
                     % (",".join(self.ivar), ",".join(self.dvar)))
        lines.append("R := `DifferentialThomas/GlobalRanking`:")
        for s in self.setup:
            lines.append(s.rstrip(":;") + ":")
        for i, q in enumerate(queries):
            lines.append('printf("OMQ|%d|%%s\\n", convert((%s), string)):'
                         % (i, q))
        lines.append("quit:")
        return "\n".join(lines) + "\n"

    def query(self, queries, env_extra=None, timeout=600):
        """Run the queries (Maple expression strings) in ONE openmaple
        process; return the lprint1D answer strings in order."""
        queries = list(queries)
        stdout, _stderr = run_openmaple(self._script(queries), env_extra,
                                        timeout)
        found = {int(m.group(1)): m.group(2)
                 for m in _OMQ_RE.finditer(stdout)}
        missing = [i for i in range(len(queries)) if i not in found]
        if missing:
            raise RuntimeError(
                "oracle answers missing for queries %s\nstdout:\n%s"
                % (missing, stdout[-4000:]))
        return [found[i] for i in range(len(queries))]

    def exec_proc(self, name, *args):
        """Maple source text for calling internal proc ``name`` on ``args``
        (each arg a Maple source fragment)."""
        return "`DifferentialThomas/%s`(%s)" % (name, ", ".join(args))


def capture_reduction_trace(script_text, procs="1", env_extra=None,
                            timeout=3600):
    """Run a script with OPENMAPLE_REDUCTION_TRACE=<procs>; return
    ``(stdout, trace_lines)`` where trace_lines are the stderr ``[red]``
    records."""
    env = dict(env_extra or {})
    env["OPENMAPLE_REDUCTION_TRACE"] = procs
    stdout, stderr = run_openmaple(script_text, env, timeout)
    trace = [ln for ln in stderr.splitlines() if ln.startswith("[red]")]
    return stdout, trace


# ---------------------------------------------------------------------------
# Normalisers: reference lprint1D <-> substrate polynomials
# ---------------------------------------------------------------------------

_MAPLE_JET_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[([0-9, ]+)\]")


def normalize_maple_name(s):
    """Canonical spaceless jet form: ``u[1, 2]`` -> ``u[1,2]``."""
    return re.sub(r"\s+", "", str(s))


def maple_leader_to_jetvar(s):
    """Parse an oracle Leader answer: ``u[1, 2]`` -> JetVar; ``1`` -> 1."""
    s = str(s).strip()
    if s == "1":
        return 1
    return JetVar.from_maple(s)


def _jet_symbol_name(head, exps):
    return "jet_%s_%s" % (head, "_".join(str(e) for e in exps))


def maple_poly_to_sr(s, dvar, nivar):
    """Parse a reference lprint1D polynomial into a Sage SR expression with
    jets replaced by neutral ``jet_<head>_<exps>`` symbols.

    ``dvar`` -- the dependent-variable heads (jet detection); other indexed
    names would be an error in Phase-1 outputs.
    """
    from sage.all import SR
    dvar = set(dvar)

    def repl(m):
        head = m.group(1)
        exps = [int(t) for t in m.group(2).split(",")]
        if head not in dvar:
            raise ValueError("unexpected indexed name %r" % (m.group(0),))
        if len(exps) != nivar:
            raise ValueError("jet arity mismatch in %r" % (m.group(0),))
        return _jet_symbol_name(head, exps)

    py = _MAPLE_JET_RE.sub(repl, str(s))
    return SR(py)


def dp_to_sr(p, ranking):
    """Convert a substrate DifferentialPolynomial to the same SR normal form
    (jets as ``jet_<head>_<exps>`` symbols, derivations/others as symbols)."""
    from sage.all import SR, QQ
    from sage_differential_polynomial import _blad
    dvar = set(ranking.dvar)
    expr = SR(0)
    for coeff, term in _blad.read_terms(p._h()):
        mon = SR(QQ(str(coeff)))
        for nm, deg in term:
            head = nm.split("[", 1)[0]
            if head in dvar:
                jv = JetVar.from_blad_name(nm, ranking.ivar)
                sym = SR.var(_jet_symbol_name(jv.head, jv.exps))
            else:
                sym = SR.var(nm)        # a derivation appearing polynomially
            mon = mon * sym ** int(deg)
        expr = expr + mon
    return expr


def poly_strings_equal(maple_str, dp, ranking):
    """True iff the reference lprint1D polynomial and the substrate polynomial
    are equal (symbolic difference is zero)."""
    a = maple_poly_to_sr(maple_str, ranking.dvar, len(ranking.ivar))
    b = dp_to_sr(dp, ranking)
    return bool((a - b).expand().is_zero())
