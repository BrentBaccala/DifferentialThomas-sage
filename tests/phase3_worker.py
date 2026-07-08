r"""
Phase-3 parity worker: the reduction engine vs the open-maple oracle, one
example per process (single live substrate ring per process).

Usage:  sage -python tests/phase3_worker.py --example ex1|ex2|ex3

Protocol
========

1. **Quasi-engine tree build** (mirrors how the engine uses reduction): a
   FIFO worklist starts with the example's input polynomials; each element is
   head+initial reduced (``ReduceWRTJanetTrees``), and -- when the result is
   nonzero, has a differential-variable leader, and has no Janet divisor --
   flagged ``NonZeroInitial`` and inserted; elements returned by the
   insertion (removed leaves, required prolongations) join the worklist.
   The port runs first, recording every decision; the oracle script then
   performs the same operations unconditionally, and the per-step checks
   (reduced form, leader, divisor verdict, insertion returns, leaf/multvar
   snapshots) confirm the decisions agree.

2. **Batteries** against the finished trees, all mirrored 1:1 in the oracle
   script: head+initial reduction (default and ``nonlineartail`` modes,
   equations and inequations), direct ``DifferentialPseudoReduction`` (incl.
   the multiplicative-variable break, the rank break, and the
   negative-derivation break), direct ``PseudoRemainder`` (plain / with
   multiplier ``f`` / with cofactor write-back, whose exact invariant
   ``f*u == q*v + r`` is also asserted port-side), tail reduction in all
   four modes (default / ``denominator`` / ``final`` / ``linearcombination``,
   the last with the port-side certificate check), and ``SimplifyPolynom``
   (content removal, forced and unforced).

3. **Trajectory diff**: the single oracle run is captured with
   ``OPENMAPLE_REDUCTION_TRACE=PseudoRemainder,DifferentialPseudoReduction,
   SimplifyPolynom`` while the port captures its own trace
   (:mod:`differentialthomas.rtrace`).  The two sequences are compared
   entry-by-entry: proc name, per-operand kind+term-count, result
   kind+term-count, and -- where both sides carry a printable small operand
   -- semantic polynomial equality.

Output: ``CHECK <label> OK|FAIL`` lines and ``SUMMARY passed=N failed=M``,
plus a ``TRACE-EXCERPT`` block for the report.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from differentialthomas import (                     # noqa: E402
    JetVar, compute_ranking, create_polynomial_object,
    create_janet_trees_object, janet_divisor_in_trees, janet_trees_leafs,
    insert_into_janet_trees,
    pseudo_remainder, differential_pseudo_reduction,
    reduce_wrt_janet_trees, reduce_nonlinear_tail_wrt_janet_trees,
    verify_linear_combination)
from differentialthomas import rtrace                # noqa: E402
from differentialthomas.janet import INFINITY        # noqa: E402
from differentialthomas.oracle import (              # noqa: E402
    run_openmaple, poly_strings_equal, normalize_maple_name, _MAPLE_JET_RE)

DT = "`DifferentialThomas/%s`"


# ---------------------------------------------------------------------------
# example definitions
# ---------------------------------------------------------------------------

EXAMPLES = {
    "ex1": dict(
        ivar=["x"], dvar=["u"],
        polys=[
            "u[1]^2-4*u[0]",
            "2*u[1]*u[2]-4*u[1]",
            "u[0]",
            "5",
            "3*u[1]-6*u[0]",
        ],
        head=[
            ("u[2]^2+u[1]-u[0]", False),
            ("3*u[3]-2*u[2]+u[0]^2", False),
            ("u[1]^3-8*u[0]*u[1]+16*u[0]^2", False),
            ("7", False),
            ("u[0]+1", False),
            ("u[1]-2*u[0]", True),
        ],
        tail=[
            ("u[2]^2+u[1]-u[0]", "default"),
            ("u[2]^2+u[1]-u[0]", "denominator"),
            ("u[1]*u[2]+u[0]^2*u[1]+5", "denominator"),
            ("u[2]+u[1]^2", "final"),
            ("u[2]^2+u[0]*u[1]", "linearcombination"),
        ],
        dpr=[
            ("2*u[1]*u[2]", "u[1]^2-4*u[0]", [0], ()),
            ("u[1]-1", "u[1]^2-4*u[0]", None, ()),
            ("u[0]^2", "u[1]^2-4*u[0]", None, ()),
            ("u[2]^2-u[1]", "u[1]^2-4*u[0]", None, ()),
            ("u[2]^2-u[1]", "u[1]^2-4*u[0]", None, ("nonlineartail",)),
        ],
        pr=[
            ("u[1]^3+u[0]", "2*u[1]^2-u[0]", "plain"),
            ("u[1]^3+u[0]", "2*u[1]^2-u[0]", "f"),
            ("u[1]^3+u[0]", "2*u[1]^2-u[0]", "fq"),
            ("4*u[1]^2-6*u[0]", "8*u[1]-12", "fq"),
            ("u[0]^3", "u[1]-u[0]", "plain"),
        ],
        simp=[
            ("6*u[1]^2-4*u[1]", False, True),
            ("-2*u[0]*u[1]", False, False),
            ("3*u[1]^2-4*u[1]", True, False),
        ]),
    "ex2": dict(
        ivar=["x"], dvar=["u", "a"],
        polys=[
            "a[0]*u[1]-u[0]",
            "a[1]",
            "a[0]",
            "a[1]*u[1]+a[0]*u[2]-u[1]",
            "u[0]^2-a[0]",
        ],
        head=[
            ("a[0]*u[1]+u[0]", False),
            ("a[0]*u[0]+3", False),           # strip-leading-term branch
            ("u[1]*a[1]-u[0]*a[0]", False),
            ("u[2]+a[2]", False),
            ("(u[0]^2-a[0])*u[1]+a[0]", False),
            ("(2*a[0]*u[0])*u[1]+a[0]", False),
            ("2*a[0]^2-a[0]", True),
        ],
        tail=[
            ("u[1]*u[0]+a[1]*u[0]+a[0]", "default"),
            ("u[1]*u[0]+a[1]*u[0]+a[0]", "denominator"),
            ("u[0]^3+a[0]*u[0]+1", "denominator"),
            ("u[1]+a[1]^2", "final"),
            ("u[1]^2+a[1]^2", "linearcombination"),
        ],
        dpr=[
            ("u[1]*u[0]", "u[0]^2-a[0]", [0], ()),
            ("u[0]-a[0]", "u[0]^2-a[0]", None, ()),
            ("u[1]^2", "u[0]^2-a[0]", None, ()),
            ("u[1]^2", "u[0]^2-a[0]", None, ("nonlineartail",)),
        ],
        pr=[
            ("(2*a[0]*u[0])*u[1]^2+u[0]", "(4*a[0]*u[0]^2)*u[1]-u[0]", "fq"),
            ("2*a[0]*u[1]^2+u[0]", "2*a[0]*u[1]-a[0]", "f"),
            ("u[1]^2-a[1]", "u[1]-a[1]", "fq"),
        ],
        simp=[
            ("4*a[0]*u[1]^2-2*a[0]*u[1]", False, True),
            ("-6*a[0]^2*u[0]", False, False),
            ("2*u[0]^2-2*a[0]", True, False),
        ]),
    "ex3": dict(
        ivar=["x"],
        dvar=["DDPs", "DPs", "Ps", "Vf",
              "a0", "a1", "b0", "b1", "c0", "c1", "V1"],
        polys=[
            "Ps[1]-DPs[0]*Vf[1]",
            "DPs[1]-DDPs[0]*Vf[1]",
            "(a0[0]+a1[0]*Vf[0])*DDPs[0]+(b0[0]+b1[0]*Vf[0])*DPs[0]"
            "+(c0[0]+c1[0]*Vf[0])*Ps[0]",
            "Vf[0]-V1[0]*x",
            "a0[1]",
            "V1[1]",
        ],
        head=[
            ("Ps[2]-DPs[1]*Vf[1]", False),
            ("DPs[1]*Ps[1]-Vf[1]^2", False),
            ("DDPs[1]", False),
            ("(a0[0]+a1[0]*Vf[0])*DDPs[0]", False),
            ("x*V1[1]+V1[0]", False),
            ("Vf[1]^2-V1[0]^2", False),
            ("c1[0]*Vf[1]+c1[1]*Vf[0]", True),
        ],
        tail=[
            ("DPs[1]*Vf[1]+Ps[1]*Vf[1]", "default"),
            ("DPs[1]*Vf[1]+Ps[1]*Vf[1]", "denominator"),
            ("Ps[2]+Vf[1]^2", "final"),
            ("Ps[1]^2+Vf[1]*Ps[0]+x*V1[0]", "linearcombination"),
        ],
        dpr=[
            ("Ps[2]*DPs[0]", "Ps[1]-DPs[0]*Vf[1]", [0], ()),
            ("Ps[1]^2+1", "Ps[1]-DPs[0]*Vf[1]", None, ()),
            ("Ps[3]-x*Ps[1]", "Ps[1]-DPs[0]*Vf[1]", None, ()),
            ("Ps[3]-x*Ps[1]", "Ps[1]-DPs[0]*Vf[1]", None, ("nonlineartail",)),
        ],
        pr=[
            ("(2*x^2+2*x)*Vf[1]^2+Ps[0]", "(4*x^3+4*x^2)*Vf[1]-x", "fq"),
            ("Vf[1]^3-Vf[0]", "2*Vf[1]-V1[0]", "f"),
            ("Ps[1]^2-x", "Ps[1]+x*Vf[0]", "fq"),
        ],
        simp=[
            ("(2*x^2+2*x)*V1[1]", False, True),
            ("3*Ps[1]^2-6*Ps[1]*Vf[0]", False, False),
            ("x*Vf[1]-x*Vf[0]", True, False),
        ]),
}


# ---------------------------------------------------------------------------
# helpers (shared shape with phase2_worker)
# ---------------------------------------------------------------------------

def maple_to_blad(s, ivar):
    def repl(m):
        return JetVar.from_maple(m.group(0)).to_blad_name(ivar)
    return _MAPLE_JET_RE.sub(repl, s)


def split_maple_list(s):
    s = s.strip()
    assert s.startswith("[") and s.endswith("]"), s
    body = s[1:-1]
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def mv_str(mv):
    return "[%s]" % ",".join(
        "infinity" if m == INFINITY else str(int(m)) for m in mv)


def verdict_str(divisor):
    return "[]" if divisor is None else "[%s]" % (divisor.leader(),)


# ---------------------------------------------------------------------------
# oracle reduction-trace parsing
# ---------------------------------------------------------------------------

_RED_RE = re.compile(r"^\[red\] #\d+ (\w+)\((.*)\) -> (.*?)(?: \| after=.*)?$")
_DESC_RE = re.compile(r"^([tp])(\d+|\?|ref)(?:\{(.*)\})?$", re.S)


def _split_top(s):
    """Split a trace operand list at top-level ', ' (outside {}, [], ())."""
    parts, depth, cur, i = [], 0, "", 0
    while i < len(s):
        ch = s[i]
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if depth == 0 and s.startswith(", ", i):
            parts.append(cur)
            cur = ""
            i += 2
            continue
        cur += ch
        i += 1
    if cur:
        parts.append(cur)
    return parts


def parse_red_line(line):
    """(proc, [(kind, count, smallform), ...], (kind, count, smallform))."""
    m = _RED_RE.match(line)
    if not m:
        return None
    proc, argstr, resstr = m.group(1), m.group(2), m.group(3)

    def parse_desc(d):
        dm = _DESC_RE.match(d.strip())
        if not dm:
            return ("?", -1, d.strip())
        kind, cnt, small = dm.group(1), dm.group(2), dm.group(3)
        n = int(cnt) if cnt.isdigit() else -1
        return (kind, n, small)

    args = [parse_desc(a) for a in _split_top(argstr)]
    return (proc, args, parse_desc(resstr))


def _comparable_form(s):
    """An oracle small-form string usable for semantic polynomial compare."""
    if s is None or s.endswith("...") or "table(" in s:
        return None
    t = s.strip()
    if t in ("true", "false", "q") or t.startswith('"'):
        return None
    return t


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True, choices=sorted(EXAMPLES))
    args = ap.parse_args()
    ex = EXAMPLES[args.example]
    ivar, dvar = ex["ivar"], ex["dvar"]

    rk = compute_ranking(ivar, dvar)
    rk.often_remove_content = True      # mirrored in the oracle setup
    R = rk.ring

    passed = failed = 0

    def check(label, ok, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print("CHECK %s OK" % (label,))
        else:
            failed += 1
            print("CHECK %s FAIL %s" % (label, detail))

    def mk(s, ineq=False):
        o = create_polynomial_object(maple_to_blad(s, ivar), rk)
        if ineq:
            o.inequation(True)
        return o

    # =======================================================================
    # PORT pass (trace capturing)
    # =======================================================================
    trace = rtrace.start_trace()
    T = create_janet_trees_object(rk)

    # -- 1. quasi-engine tree build ----------------------------------------
    steps = []
    worklist = [("poly", i) for i in range(len(ex["polys"]))]
    while worklist:
        src = worklist.pop(0)
        if src[0] == "poly":
            obj = mk(ex["polys"][src[1]])
        else:
            obj = steps[src[1]]["ret"][src[2]]
        red = reduce_wrt_janet_trees(T, obj)
        div = janet_divisor_in_trees(T, red)
        inserted = (not red.standard_form().is_zero()
                    and red.leader() != 1 and div is None)
        rec = dict(src=src, red_std=red.standard_form(),
                   red_ldr=str(red.leader()), verdict=verdict_str(div),
                   inserted=inserted, ret=None, ret_pairs=None, leaves=None)
        if inserted:
            red.nonzero_initial(True)
            ret = insert_into_janet_trees(T, red)
            rec["ret"] = ret
            rec["ret_pairs"] = [(str(q.leader()), q.standard_form())
                                for q in ret]
            rec["leaves"] = [(str(q.leader()),
                              mv_str(q.f["MultiplicativeVariables"]))
                             for q in janet_trees_leafs(T)]
            si = len(steps)
            worklist.extend(("ret", si, k) for k in range(len(ret)))
        steps.append(rec)

    # -- 2. head+initial reduction battery ---------------------------------
    head_res = []
    for s, ineq in ex["head"]:
        o1 = mk(s, ineq)
        r1 = reduce_wrt_janet_trees(T, o1)
        o2 = mk(s, ineq)
        r2, f2 = reduce_wrt_janet_trees(T, o2, "nonlineartail")
        head_res.append(dict(std=r1.standard_form(), ldr=str(r1.leader()),
                             ineq=r1.inequation(),
                             std2=r2.standard_form(), f2=f2))

    # -- 3. DifferentialPseudoReduction battery ----------------------------
    dpr_res = []
    for ps, qs, mv, extra in ex["dpr"]:
        qo = mk(qs)
        if mv is not None:
            qo.f["MultiplicativeVariables"] = list(mv)
        po = mk(ps)
        if "nonlineartail" in extra:
            res, f = differential_pseudo_reduction(po, qo, *extra)
            dpr_res.append(dict(std=res.standard_form(), f=f))
        else:
            res = differential_pseudo_reduction(po, qo, *extra)
            dpr_res.append(dict(std=res.standard_form(), f=None))

    # -- 4. PseudoRemainder battery -----------------------------------------
    pr_res = []
    for us, vs, mode in ex["pr"]:
        uo, vo = mk(us), mk(vs)
        if mode == "plain":
            r = pseudo_remainder(uo, vo, False)
            pr_res.append(dict(std=r.standard_form(), f=None, q=None))
        elif mode == "f":
            r, f = pseudo_remainder(uo, vo, True)
            pr_res.append(dict(std=r.standard_form(), f=f, q=None))
        else:                                   # fq: with cofactor
            cof = []
            r, f = pseudo_remainder(uo, vo, True, cofactor=cof)
            pr_res.append(dict(std=r.standard_form(), f=f, q=cof[0]))
            # exact pseudo-division invariant, port-side
            lhs = f * uo.standard_form() - cof[0] * vo.standard_form() \
                - r.standard_form()
            check(("pr-invariant", us, vs), lhs.is_zero(),
                  "f*u - q*v - r != 0")

    # -- 5. tail-reduction battery -------------------------------------------
    tail_res = []
    for s, mode in ex["tail"]:
        o = mk(s)
        p0 = o.standard_form()
        if mode == "final":
            o.f["MultiplicativeVariables"] = [0] * len(ivar)
            res = reduce_nonlinear_tail_wrt_janet_trees(T, o, "final")
            tail_res.append(dict(std=res.standard_form(), f=None,
                                 mv=res.f.get("MultiplicativeVariables")))
        elif mode == "denominator":
            res, f = reduce_nonlinear_tail_wrt_janet_trees(T, o, "denominator")
            tail_res.append(dict(std=res.standard_form(), f=f, mv=None))
        elif mode == "linearcombination":
            res, f, lc = reduce_nonlinear_tail_wrt_janet_trees(
                T, o, "linearcombination")
            tail_res.append(dict(std=res.standard_form(), f=f, mv=None))
            check(("lc-certificate", s),
                  verify_linear_combination(p0, res.standard_form(), f, lc, R),
                  "linearcombination identity failed")
        else:
            res = reduce_nonlinear_tail_wrt_janet_trees(T, o)
            tail_res.append(dict(std=res.standard_form(), f=None, mv=None))

    # -- 6. SimplifyPolynom battery -------------------------------------------
    simp_res = []
    for s, force, prime in ex["simp"]:
        o = mk(s)
        if prime:
            o.initial()
        o.simplify_polynom(force)
        simp_res.append(dict(std=o.standard_form(),
                             init=o.f.get("Initial") if prime else None))

    port_trace = rtrace.stop_trace()

    # =======================================================================
    # ORACLE script (mirrors the recorded operations exactly; queries are
    # printf statements INTERLEAVED with the setup so each one observes the
    # same state the port did -- e.g. divisor verdicts / leaf snapshots at
    # the step they belong to, not after the whole build)
    # =======================================================================
    script_lines = [
        "with(DifferentialThomas):",
        "`DifferentialThomas/ComputeRanking`([%s],[%s]):"
        % (",".join(ivar), ",".join(dvar)),
        "R := `DifferentialThomas/GlobalRanking`:",
    ]
    labels = []

    def setup_append(stmt):
        script_lines.append(stmt.rstrip(":;") + ":")
    setup = type("S", (), {"append": staticmethod(setup_append)})()
    setup.append("R['OftenRemoveContent'] := true")
    # ProcInput installs these on the ranking (main:66/238); a bare
    # ComputeRanking leaves ReductionSystem unassigned, which would turn the
    # initial-strip test of ReduceWRTJanetTrees into an inert (never-zero)
    # comparison -- install the engine default explicitly
    setup.append("R['ReductionSystem'] := (a -> a)")

    def q(label, text):
        labels.append(label)
        script_lines.append('printf("OMQ|%d|%%s\\n", convert((%s), string)):'
                            % (len(labels) - 1, text))

    ldr = DT % "Leader"
    sf = DT % "StandardForm"
    pair_map = "map(qq->[%s(qq), %s(qq)], %%s)" % (ldr, sf)
    leaf_map = ("map(qq->[%s(qq), qq['MultiplicativeVariables']], %s(T))"
                % (ldr, DT % "JanetTreesLeafs"))

    setup.append("T := %s(R)" % (DT % "CreateJanetTreesObject",))
    for i, rec in enumerate(steps):
        src = rec["src"]
        if src[0] == "poly":
            setup.append("w%d := %s(%s, R)"
                         % (i, DT % "CreatePolynomialObject",
                            ex["polys"][src[1]]))
        else:
            setup.append("w%d := r%d[%d]" % (i, src[1], src[2] + 1))
        setup.append("red%d := %s(T, w%d)"
                     % (i, DT % "ReduceWRTJanetTrees", i))
        q(("st", i, "std"), "%s(red%d)" % (sf, i))
        q(("st", i, "ldr"), "%s(red%d)" % (ldr, i))
        q(("st", i, "dv"), "map(qq->%s(qq), [%s(T, red%d)])"
          % (ldr, DT % "JanetDivisorInTrees", i))
        if rec["inserted"]:
            setup.append("%s(red%d, true)" % (DT % "NonZeroInitial", i))
            setup.append("r%d := %s(T, red%d)"
                         % (i, DT % "InsertIntoJanetTrees", i))
            q(("st", i, "ret"), pair_map % ("r%d" % i))
            q(("st", i, "leaves"), leaf_map)

    for j, (s, ineq) in enumerate(ex["head"]):
        setup.append("hb%d := %s(%s, R)"
                     % (j, DT % "CreatePolynomialObject", s))
        if ineq:
            setup.append("%s(hb%d, true)" % (DT % "Inequation", j))
        setup.append("hr%d := %s(T, hb%d)"
                     % (j, DT % "ReduceWRTJanetTrees", j))
        q(("hd", j, "std"), "%s(hr%d)" % (sf, j))
        q(("hd", j, "ldr"), "%s(hr%d)" % (ldr, j))
        q(("hd", j, "ineq"), "%s(hr%d)" % (DT % "Inequation", j))
        setup.append("hc%d := %s(%s, R)"
                     % (j, DT % "CreatePolynomialObject", s))
        if ineq:
            setup.append("%s(hc%d, true)" % (DT % "Inequation", j))
        setup.append('hs%d := [%s(T, hc%d, "nonlineartail")]'
                     % (j, DT % "ReduceWRTJanetTrees", j))
        q(("hd", j, "std2"), "%s(hs%d[1])" % (sf, j))
        q(("hd", j, "f2"), "hs%d[2]" % j)

    for j, (ps, qs, mv, extra) in enumerate(ex["dpr"]):
        setup.append("dq%d := %s(%s, R)"
                     % (j, DT % "CreatePolynomialObject", qs))
        if mv is not None:
            setup.append("dq%d['MultiplicativeVariables'] := [%s]"
                         % (j, ",".join(str(m) for m in mv)))
        setup.append("dp%d := %s(%s, R)"
                     % (j, DT % "CreatePolynomialObject", ps))
        extra_src = "".join(', "%s"' % e for e in extra)
        setup.append("dr%d := [%s(dp%d, dq%d%s)]"
                     % (j, DT % "DifferentialPseudoReduction", j, j,
                        extra_src))
        q(("dpr", j, "std"), "%s(dr%d[1])" % (sf, j))
        if "nonlineartail" in extra:
            q(("dpr", j, "f"), "dr%d[2]" % j)

    for j, (us, vs, mode) in enumerate(ex["pr"]):
        setup.append("pu%d := %s(%s, R)"
                     % (j, DT % "CreatePolynomialObject", us))
        setup.append("pv%d := %s(%s, R)"
                     % (j, DT % "CreatePolynomialObject", vs))
        if mode == "plain":
            setup.append("pr%d := [%s(pu%d, pv%d, false)]"
                         % (j, DT % "PseudoRemainder", j, j))
        elif mode == "f":
            setup.append("pr%d := [%s(pu%d, pv%d, true)]"
                         % (j, DT % "PseudoRemainder", j, j))
        else:
            # NOTE the cofactor is passed but NOT read back: open-maple does
            # not implement Maple's assign-through-parameter write-back
            # (verified 2026-07-08: the caller's name stays unassigned), so
            # the oracle cannot serialise it.  Since v != 0 in an integral
            # domain, q = (f*u - r)/v is UNIQUE -- parity of f and r (checked
            # below) implies parity of q, and the port asserts the exact
            # invariant f*u == q*v + r on its own cofactor.
            setup.append("qv%d := 'qv%d'" % (j, j))
            setup.append("pr%d := [%s(pu%d, pv%d, true, qv%d)]"
                         % (j, DT % "PseudoRemainder", j, j, j))
        q(("pr", j, "std"), "%s(pr%d[1])" % (sf, j))
        if mode in ("f", "fq"):
            q(("pr", j, "f"), "pr%d[2]" % j)

    for j, (s, mode) in enumerate(ex["tail"]):
        setup.append("tb%d := %s(%s, R)"
                     % (j, DT % "CreatePolynomialObject", s))
        if mode == "final":
            setup.append("tb%d['MultiplicativeVariables'] := [%s]"
                         % (j, ",".join("0" for _ in ivar)))
            setup.append('tr%d := [%s(T, tb%d, "final")]'
                         % (j, DT % "ReduceNonLinearTailWRTJanetTrees", j))
        elif mode == "default":
            setup.append("tr%d := [%s(T, tb%d)]"
                         % (j, DT % "ReduceNonLinearTailWRTJanetTrees", j))
        else:
            setup.append('tr%d := [%s(T, tb%d, "%s")]'
                         % (j, DT % "ReduceNonLinearTailWRTJanetTrees", j,
                            mode))
        q(("tl", j, "std"), "%s(tr%d[1])" % (sf, j))
        if mode in ("denominator", "linearcombination"):
            q(("tl", j, "f"), "tr%d[2]" % j)
        if mode == "final":
            q(("tl", j, "mv"), "eval(tr%d[1]['MultiplicativeVariables'])" % j)

    for j, (s, force, prime) in enumerate(ex["simp"]):
        setup.append("sp%d := %s(%s, R)"
                     % (j, DT % "CreatePolynomialObject", s))
        if prime:
            setup.append("%s(sp%d)" % (DT % "Initial", j))
        if force:
            setup.append('%s(sp%d, "force")' % (DT % "SimplifyPolynom", j))
        else:
            setup.append("%s(sp%d)" % (DT % "SimplifyPolynom", j))
        q(("sp", j, "std"), "%s(sp%d)" % (sf, j))
        if prime:
            q(("sp", j, "init"), "eval(sp%d['Initial'])" % j)

    script_lines.append("quit:")
    script = "\n".join(script_lines) + "\n"
    stdout, stderr = run_openmaple(
        script,
        env_extra={"OPENMAPLE_REDUCTION_TRACE":
                   "PseudoRemainder,DifferentialPseudoReduction,"
                   "SimplifyPolynom"},
        timeout=3000)
    from differentialthomas.oracle import _OMQ_RE
    found = {int(m.group(1)): m.group(2) for m in _OMQ_RE.finditer(stdout)}
    missing = [i for i in range(len(labels)) if i not in found]
    if missing:
        raise RuntimeError("oracle answers missing for %s\n%s"
                           % (missing, stdout[-4000:]))
    answers = {labels[i]: found[i] for i in range(len(labels))}
    oracle_trace = [parse_red_line(ln) for ln in stderr.splitlines()
                    if ln.startswith("[red]")]
    oracle_trace = [t for t in oracle_trace if t is not None]

    # =======================================================================
    # comparisons
    # =======================================================================

    def cmp_poly(label, ans, dp):
        check(label, poly_strings_equal(ans, dp, rk),
              "oracle=%r port=%r" % (ans, str(dp)))

    def cmp_str(label, ans, s):
        check(label, normalize_maple_name(ans) == s,
              "oracle=%r port=%r" % (ans, s))

    def cmp_bool(label, ans, b):
        check(label, (ans.strip() == "true") == bool(b),
              "oracle=%r port=%r" % (ans, b))

    # quasi-engine steps
    for i, rec in enumerate(steps):
        cmp_poly(("st", i, "std"), answers[("st", i, "std")], rec["red_std"])
        cmp_str(("st", i, "ldr"), answers[("st", i, "ldr")], rec["red_ldr"])
        cmp_str(("st", i, "dv"), answers[("st", i, "dv")], rec["verdict"])
        if rec["inserted"]:
            elems = split_maple_list(answers[("st", i, "ret")])
            if len(elems) != len(rec["ret_pairs"]):
                check(("st", i, "ret"), False,
                      "length: oracle=%r" % answers[("st", i, "ret")])
            else:
                for t, (e, (pl, pp)) in enumerate(zip(elems,
                                                      rec["ret_pairs"])):
                    lstr, pstr = split_maple_list(e)
                    cmp_str(("st", i, "ret", t, "ldr"), lstr, pl)
                    cmp_poly(("st", i, "ret", t, "poly"), pstr, pp)
            lelems = split_maple_list(answers[("st", i, "leaves")])
            if len(lelems) != len(rec["leaves"]):
                check(("st", i, "leaves"), False,
                      "length: oracle=%r" % answers[("st", i, "leaves")])
            else:
                for t, (e, (pl, pm)) in enumerate(zip(lelems,
                                                      rec["leaves"])):
                    lstr, mstr = split_maple_list(e)
                    cmp_str(("st", i, "leaf", t, "ldr"), lstr, pl)
                    cmp_str(("st", i, "leaf", t, "mv"), mstr, pm)

    # head battery
    for j, hr in enumerate(head_res):
        cmp_poly(("hd", j, "std"), answers[("hd", j, "std")], hr["std"])
        cmp_str(("hd", j, "ldr"), answers[("hd", j, "ldr")], hr["ldr"])
        cmp_bool(("hd", j, "ineq"), answers[("hd", j, "ineq")], hr["ineq"])
        cmp_poly(("hd", j, "std2"), answers[("hd", j, "std2")], hr["std2"])
        cmp_poly(("hd", j, "f2"), answers[("hd", j, "f2")], hr["f2"])

    # DPR battery
    for j, dr in enumerate(dpr_res):
        cmp_poly(("dpr", j, "std"), answers[("dpr", j, "std")], dr["std"])
        if dr["f"] is not None:
            cmp_poly(("dpr", j, "f"), answers[("dpr", j, "f")], dr["f"])

    # PseudoRemainder battery
    for j, prr in enumerate(pr_res):
        cmp_poly(("pr", j, "std"), answers[("pr", j, "std")], prr["std"])
        if prr["f"] is not None:
            cmp_poly(("pr", j, "f"), answers[("pr", j, "f")], prr["f"])
        # the cofactor q is not oracle-serialisable (no write-back in
        # open-maple) but is uniquely determined by f and r -- see above

    # tail battery
    for j, ((s, mode), tr) in enumerate(zip(ex["tail"], tail_res)):
        cmp_poly(("tl", j, "std"), answers[("tl", j, "std")], tr["std"])
        if tr["f"] is not None:
            cmp_poly(("tl", j, "f"), answers[("tl", j, "f")], tr["f"])
        if mode == "final":
            cmp_str(("tl", j, "mv"), answers[("tl", j, "mv")],
                    mv_str(tr["mv"]))

    # SimplifyPolynom battery
    for j, sr in enumerate(simp_res):
        cmp_poly(("sp", j, "std"), answers[("sp", j, "std")], sr["std"])
        if sr["init"] is not None:
            cmp_poly(("sp", j, "init"), answers[("sp", j, "init")],
                     sr["init"])

    # -- trajectory ---------------------------------------------------------
    check(("trace", "count"), len(oracle_trace) == len(port_trace),
          "oracle=%d port=%d" % (len(oracle_trace), len(port_trace)))
    for k, (ot, pt) in enumerate(zip(oracle_trace, port_trace)):
        oproc, oargs, ores = ot
        ok = (oproc == pt.proc and len(oargs) == len(pt.args)
              and all(a[:2] == b[:2] for a, b in zip(oargs, pt.args))
              and ores[:2] == pt.result[:2])
        detail = ""
        if ok:
            # semantic compare of small printed operand forms where possible
            for a, b in zip(oargs, pt.args):
                form = _comparable_form(a[2])
                if form is not None and b[2] is not None \
                        and not isinstance(b[2], str):
                    if not poly_strings_equal(form, b[2], rk):
                        ok = False
                        detail = "small-form mismatch: %r vs %r" % (form,
                                                                    str(b[2]))
                        break
        else:
            detail = "oracle=%s port=%r" % (ot, pt)
        check(("trace", k, oproc), ok, detail)

    # excerpt for the report
    print("TRACE-EXCERPT (first 8 of %d, oracle | port):" % len(port_trace))
    for k in range(min(8, min(len(oracle_trace), len(port_trace)))):
        oproc, oargs, ores = oracle_trace[k]

        def fmt(d):
            return "%s%d" % (d[0], d[1]) if d[2] is None \
                else "%s%d{%s}" % (d[0], d[1], d[2])
        print("  #%d %s(%s) -> %s" % (k, oproc,
                                      ", ".join(fmt(a) for a in oargs),
                                      fmt(ores)))
        print("     %r" % (port_trace[k],))

    print("SUMMARY passed=%d failed=%d" % (passed, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
