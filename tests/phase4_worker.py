r"""
Phase-4 parity worker: the split / factor / sort / strategy / passivity layer
vs the open-maple oracle, one example per process (one live substrate ring per
process).

Usage:  sage -python tests/phase4_worker.py --example ex1|ex2|ex3

Gate groups (each mirrored 1:1 in a single interleaved oracle script):

1. **SplitByInitial** -- child systems + mutated original + reduced element.
2. **SplitBySquarefree** -- discriminant/subresultant split children.
3. **DivideByInequation** -- subresultant-vanishing split children.
4. **Factorize** -- factor-branch children (equation and inequation).
5. **Reduction(DS, q)** -- the reduced ``q`` and any spawned children.
6. **InsertIntoQList / Strategy** -- resulting Q order (by leader/inequation)
   and the selected 1-based index.
7. **Criteria** -- the involutive skip/no-skip verdict, over a built tree.

Systems for groups 1-5 use an EMPTY Janet-trees object plus an explicit
``Inequations`` list: reduction w.r.t. empty trees is the identity, so the split
*logic* (child spawning, subresultant zero-pattern, cofactor, implied-inequation
suppression) is exercised in isolation and mirrors the reference exactly.
Group 7 builds a one-leaf tree (mirrored insert) so ``Criteria`` can inspect
tree leafs.

Equations / inequations of the spawned systems are compared **up to a nonzero
scalar (sign + content)** -- the mathematically correct equivalence for a Thomas
system's constraint sets, and the equivalence under which the substrate's
Ducos subresultants (sign-up-to-parity) match the oracle's SPRS.  Systems are
compared as an unordered multiset (the decomposition is a *set* of systems).

Output: ``CHECK <label> OK|FAIL`` lines and ``SUMMARY passed=N failed=M``.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import differentialthomas as dt                        # noqa: E402
from differentialthomas.jetvar import JetVar           # noqa: E402
from differentialthomas.maplecas import (              # noqa: E402
    term_data, integer_content, drl_lead_coeff, _div_scalar)
from differentialthomas.oracle import (                # noqa: E402
    run_openmaple, _MAPLE_JET_RE, _OMQ_RE)

DT = "`DifferentialThomas/%s`"


# ---------------------------------------------------------------------------
# example definitions
# ---------------------------------------------------------------------------

EXAMPLES = {
    "ex1": dict(
        ivar=["x"], dvar=["u"],
        # split-logic scenarios: (op, kwargs) each rebuilds a fresh empty-tree
        # system with the given inequations, then runs the op on q.
        split_by_initial=[
            dict(q="u[1]*u[2]-u[0]", ineqs=[]),
            dict(q="u[1]*u[2]-u[0]", ineqs=["u[1]"]),   # initial implied
        ],
        split_by_squarefree=[
            dict(p="u[1]^2-2*u[0]*u[1]+u[0]^2", ineqs=[]),   # (u[1]-u[0])^2
        ],
        divide_by_inequation=[
            dict(p="u[1]^2-u[0]^2", q="u[1]-u[0]", ineqs=[]),
        ],
        factorize=[
            dict(q="u[1]*u[0]-u[1]", ineq=False),        # equation
            dict(q="u[1]*u[0]-u[1]", ineq=True),         # inequation
        ],
        reduction=[
            dict(q="u[1]*u[0]-u[1]", ineq=False, ineqs=[]),
        ],
        qlist=[
            ["u[2]-u[0]", "u[1]-u[0]", ("u[2]+u[0]", "ineq"), "u[0]-1"],
        ],
        # criteria: a one-leaf tree (leaf poly), q = prolongation, divisor=leaf
        criteria=[
            dict(leaf="u[1]^2-u[0]", derive=[1]),        # crit2 -> true
            dict(leaf="u[1]^2-u[0]", derive=[0]),        # q == divisor
        ]),
    "ex2": dict(
        ivar=["x"], dvar=["u", "a"],
        split_by_initial=[
            dict(q="a[0]*u[1]-u[0]", ineqs=[]),
            dict(q="a[0]*u[1]-u[0]", ineqs=["a[0]"]),
        ],
        split_by_squarefree=[
            dict(q_is_p="u[1]^2-2*a[0]*u[1]+a[0]^2", ineqs=[]),
        ],
        divide_by_inequation=[
            dict(p="u[1]^2-a[0]^2", q="u[1]-a[0]", ineqs=[]),
        ],
        factorize=[
            dict(q="u[1]*a[0]-u[1]", ineq=False),
            dict(q="u[1]*a[0]-u[1]", ineq=True),
        ],
        reduction=[
            dict(q="u[1]*a[0]-u[1]", ineq=False, ineqs=[]),
        ],
        qlist=[
            ["u[1]-a[0]", "a[1]-a[0]", ("u[1]+a[0]", "ineq"), "a[0]-1"],
        ],
        criteria=[
            dict(leaf="u[1]^2-a[0]", derive=[1]),
            dict(leaf="u[1]^2-a[0]", derive=[0]),
        ]),
    "ex3": dict(
        ivar=["x"],
        dvar=["DDPs", "DPs", "Ps", "Vf",
              "a0", "a1", "b0", "b1", "c0", "c1", "V1"],
        split_by_initial=[
            dict(q="Vf[0]*Ps[1]-DPs[0]", ineqs=[]),
            dict(q="Vf[0]*Ps[1]-DPs[0]", ineqs=["Vf[0]"]),
        ],
        split_by_squarefree=[
            dict(q_is_p="Ps[1]^2-2*Vf[0]*Ps[1]+Vf[0]^2", ineqs=[]),
        ],
        divide_by_inequation=[
            dict(p="Ps[1]^2-Vf[0]^2", q="Ps[1]-Vf[0]", ineqs=[]),
        ],
        factorize=[
            dict(q="Ps[1]*Vf[0]-Ps[1]", ineq=False),
            dict(q="Ps[1]*Vf[0]-Ps[1]", ineq=True),
        ],
        reduction=[
            dict(q="Ps[1]*Vf[0]-Ps[1]", ineq=False, ineqs=[]),
        ],
        qlist=[
            ["Ps[1]-Vf[0]", "DPs[1]-Vf[0]", ("Ps[1]+Vf[0]", "ineq"),
             "Vf[0]-1"],
        ],
        criteria=[
            dict(leaf="Ps[1]^2-Vf[0]", derive=[1]),
            dict(leaf="Ps[1]^2-Vf[0]", derive=[0]),
        ]),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def maple_to_blad(s, ivar):
    return _MAPLE_JET_RE.sub(
        lambda m: JetVar.from_maple(m.group(0)).to_blad_name(ivar), s)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True, choices=sorted(EXAMPLES))
    args = ap.parse_args()
    ex = EXAMPLES[args.example]
    ivar, dvar = ex["ivar"], ex["dvar"]

    rk = dt.compute_ranking(ivar, dvar)
    rk.often_remove_content = True
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
        o = dt.create_polynomial_object(maple_to_blad(s, ivar), rk)
        if ineq:
            o.inequation(True)
        return o

    def canon(dp):
        """Canonical representative up to a nonzero scalar (content + sign)."""
        if dp.is_zero():
            return "0"
        terms = term_data(dp, rk)
        c = integer_content(terms)
        b = _div_scalar(dp, c, rk)
        bt = term_data(b, rk)
        if drl_lead_coeff(bt) < 0:
            b = -b
        return str(b)

    def canon_str(s):
        """Canon of an oracle poly string."""
        return canon(R(maple_to_blad(s, ivar)))

    def sys_key_port(system):
        eqs = tuple(sorted(canon(p)
                           for p in dt.differential_system_equations(system)))
        ineqs = tuple(sorted(canon(p)
                             for p in dt.differential_system_inequations(system)))
        return (eqs, ineqs)

    def sys_key_oracle(eq_str, ineq_str):
        eqs = tuple(sorted(canon_str(p) for p in split_maple_list(eq_str)))
        ineqs = tuple(sorted(canon_str(p) for p in split_maple_list(ineq_str)))
        return (eqs, ineqs)

    # =======================================================================
    # PORT pass
    # =======================================================================
    port = {}   # label -> comparable value

    def fresh_system(ineqs):
        T = dt.create_janet_trees_object(rk)
        ds = dt.create_differential_system([], rk, T)
        ds.Inequations = [mk(s, ineq=True) for s in ineqs]
        return ds

    # 1. SplitByInitial
    for j, sc in enumerate(ex["split_by_initial"]):
        ds = fresh_system(sc["ineqs"])
        q = mk(sc["q"])
        kids = dt.split_by_initial(ds, q)
        systems = [sys_key_port(ds)] + [sys_key_port(k) for k in kids]
        port[("si", j)] = (sorted(systems), canon(q.standard_form()))

    # 2. SplitBySquarefree -- PRS-engine parity (the reference operator itself
    #    cannot run in open-maple: CoFactorPRS needs a pseudo-quotient
    #    write-back open-maple does not implement, so p['Polynom'] becomes an
    #    unassigned name and the operator stack-overflows).  Gate the
    #    subresultant chain (SubResultantPRS) + PRSGCD at the split index (both
    #    run), and check the cofactor's exact-quotient invariant port-side.
    for j, sc in enumerate(ex["split_by_squarefree"]):
        src = sc.get("p") or sc.get("q_is_p")
        p = mk(src)
        p.nonzero_initial(True)
        sep = dt.create_polynomial_object(p.separant(), rk)
        sep.simplify_polynom(force=True)
        res = dt.initialize_resultant(None, p, sep)
        deg = res.deg
        subres = [canon(R(dt.sub_resultant(res, i))) for i in range(deg + 1)]
        i0 = next((i for i in range(deg + 1)
                   if not R(dt.sub_resultant(res, i)).is_zero()), deg)
        gcd0 = canon(dt.prs_gcd(res, i0).standard_form()) if i0 < deg else "1"
        cof = R(dt.co_factor(None, res, i0))
        # exact-quotient invariant: p == cofactor * gcd (up to nonzero scalar)
        gpoly = res.chain[i0] if i0 < deg else R.one()
        inv_ok = canon(p.standard_form()) == canon(cof * gpoly)
        port[("sf", j)] = dict(deg=deg, subres=subres, i0=i0, gcd0=gcd0,
                               inv=inv_ok)

    # 3. DivideByInequation -- same PRS-engine parity gate + cofactor invariant.
    for j, sc in enumerate(ex["divide_by_inequation"]):
        p = mk(sc["p"])
        q = mk(sc["q"], ineq=True)
        p.nonzero_initial(True)
        res = dt.initialize_resultant(None, p, q)
        deg = res.deg
        lim = min(p.rank() - 1, q.rank())
        subres = [canon(R(dt.sub_resultant(res, i))) for i in range(deg + 1)]
        i0 = next((i for i in range(lim + 1)
                   if not R(dt.sub_resultant(res, i)).is_zero()), lim)
        gpoly = res.chain[i0] if i0 < deg else R.one()
        cof = R(dt.co_factor(None, res, i0))
        inv_ok = canon(p.standard_form()) == canon(cof * gpoly)
        port[("dbi", j)] = dict(deg=deg, subres=subres, i0=i0, inv=inv_ok)

    # 4. Factorize
    for j, sc in enumerate(ex["factorize"]):
        ds = fresh_system([])
        q = mk(sc["q"], ineq=sc["ineq"])
        kids = dt.factorize(ds, q)
        systems = [sys_key_port(ds)] + [sys_key_port(k) for k in kids]
        port[("fac", j)] = (sorted(systems), canon(q.standard_form()))

    # 5. Reduction
    for j, sc in enumerate(ex["reduction"]):
        ds = fresh_system(sc["ineqs"])
        q = mk(sc["q"], ineq=sc["ineq"])
        kids, qred = dt.reduction(ds, q)
        systems = [sys_key_port(ds)] + [sys_key_port(k) for k in kids]
        port[("red", j)] = (sorted(systems), canon(qred.standard_form()))

    # 6. InsertIntoQList / Strategy
    for j, ql in enumerate(ex["qlist"]):
        objs = [mk(e[0], ineq=True) if isinstance(e, tuple) else mk(e)
                for e in ql]
        ordered = dt.insert_into_qlist(objs, [])
        ds = fresh_system([])
        ds.Q = list(ordered)
        idx = dt.strategy(ds)
        port[("ql", j)] = (
            [(str(o.leader()), o.inequation(), canon(o.standard_form()))
             for o in ordered], idx)

    # 7. Criteria
    for j, sc in enumerate(ex["criteria"]):
        T = dt.create_janet_trees_object(rk)
        leaf = mk(sc["leaf"])
        leaf.nonzero_initial(True)
        dt.insert_into_janet_trees(T, leaf)
        divisor = dt.janet_trees_leafs(T)[0]
        q = dt.multiple_partial_derivative(divisor, sc["derive"])
        port[("crit", j)] = bool(dt.criteria(q, T, divisor))

    # =======================================================================
    # ORACLE script (interleaved, mirroring each recorded operation)
    # =======================================================================
    lines = [
        "with(DifferentialThomas):",
        "`DifferentialThomas/ComputeRanking`([%s],[%s]):"
        % (",".join(ivar), ",".join(dvar)),
        "R := `DifferentialThomas/GlobalRanking`:",
        # ProcInput installs these on the ranking table; a bare ComputeRanking
        # leaves them unassigned (which would silently disable Factorize and
        # crash InsertIntoQList), so install the engine defaults explicitly.
        "R['OftenRemoveContent'] := true:",
        "R['ReductionSystem'] := (a -> a):",
        "R['Factor'] := true:",
        "R['FactorStrong'] := false:",
        "R['FactorInequations'] := true:",
        "R['InequationsNotCoprime'] := false:",
        "R['InequationsNotSquarefree'] := false:",
        "R['TailReduction'] := true:",
        "R['ReductionOld'] := false:",
        "R['ReductionFactor'] := false:",
        "R['TailReductionIntermediate'] := false:",
        "R['MaxSizeMultiplicator'] := infinity:",
        "R['CompareStrategy'] := "
        "`DifferentialThomas/ComparePolynomialsByEquationThenRanking`:",
        "R['FillS'] := `DifferentialThomas/FillSBySmallestLeader`:",
        "R['SelectionStrategy'] := `DifferentialThomas/StrategySmallestElement`:",
    ]
    labels = []

    def q_(label, text):
        labels.append(label)
        lines.append('printf("OMQ|%d|%%s\\n", convert((%s), string)):'
                     % (len(labels) - 1, text))

    CPO = DT % "CreatePolynomialObject"
    INEQ = DT % "Inequation"
    NZI = DT % "NonZeroInitial"
    SQF = DT % "Squarefree"
    SF = DT % "StandardForm"
    EQS = DT % "DifferentialSystemEquations"
    INEQS = DT % "DifferentialSystemInequations"
    CJT = DT % "CreateJanetTreesObject"
    CDS = DT % "CreateDifferentialSystem"

    ctr = [0]

    def new(prefix):
        ctr[0] += 1
        return "%s%d" % (prefix, ctr[0])

    def build_sys(ineqs):
        t = new("T")
        d = new("DS")
        lines.append("%s := %s(R):" % (t, CJT))
        lines.append("%s := %s([], R, %s):" % (d, CDS, t))
        if ineqs:
            objs = []
            for s in ineqs:
                nm = new("iq")
                lines.append("%s := %s(%s, R): %s(%s, true):"
                             % (nm, CPO, s, INEQ, nm))
                objs.append(nm)
            lines.append("%s['Inequations'] := [%s]:" % (d, ",".join(objs)))
        return d

    def query_split(label, d, resname):
        q_((label, "dseq"), "%s(%s)" % (EQS, d))
        q_((label, "dsineq"), "%s(%s)" % (INEQS, d))
        q_((label, "ceq"), "map(c->%s(c), %s)" % (EQS, resname))
        q_((label, "cineq"), "map(c->%s(c), %s)" % (INEQS, resname))

    # 1. SplitByInitial
    for j, sc in enumerate(ex["split_by_initial"]):
        d = build_sys(sc["ineqs"])
        qn = new("q")
        lines.append("%s := %s(%s, R):" % (qn, CPO, sc["q"]))
        rn = new("res")
        lines.append("%s := [%s(%s, %s)]:"
                     % (rn, DT % "SplitByInitial", d, qn))
        query_split(("si", j), d, rn)
        q_(("si", j, "qsf"), "%s(%s)" % (SF, qn))

    IRPRS = DT % "InitializeResultantPRS"
    SRPRS = DT % "SubResultantPRS"
    PRSGCD = DT % "PRSGCD"

    # 2. SplitBySquarefree -- gate the reference PRS engine (SubResultantPRS
    #    chain + PRSGCD at the split index).  The operator itself cannot run in
    #    open-maple (CoFactor write-back missing), so p / the child set are not
    #    directly oracle-comparable; the subresultant zero-pattern fixes the
    #    split index and the child's s, which IS the reference-checkable core.
    for j, sc in enumerate(ex["split_by_squarefree"]):
        src = sc.get("p") or sc.get("q_is_p")
        pn = new("p")
        sn = new("sep")
        rn = new("res")
        lines.append("%s := %s(%s, R): %s(%s, true):" % (pn, CPO, src, NZI, pn))
        lines.append("%s := %s(diff(%s(%s), %s(%s)), R):"
                     % (sn, CPO, SF, pn, DT % "Leader", pn))
        lines.append("%s(%s, \"force\"):" % (DT % "SimplifyPolynom", sn))
        lines.append("%s := %s(0, %s, %s):" % (rn, IRPRS, pn, sn))
        deg = port[("sf", j)]["deg"]
        q_(("sf", j, "deg"), "%s['Deg']" % rn)
        for i in range(deg + 1):
            q_(("sf", j, "sr", i), "%s(%s, %d)" % (SRPRS, rn, i))
        i0 = port[("sf", j)]["i0"]
        if i0 < deg:
            q_(("sf", j, "gcd"), "%s(%s(%s, %d))" % (SF, PRSGCD, rn, i0))

    # 3. DivideByInequation -- same PRS-engine gate.
    for j, sc in enumerate(ex["divide_by_inequation"]):
        pn, qn = new("p"), new("q")
        rn = new("res")
        lines.append("%s := %s(%s, R): %s(%s, true):"
                     % (pn, CPO, sc["p"], NZI, pn))
        lines.append("%s := %s(%s, R): %s(%s, true):"
                     % (qn, CPO, sc["q"], NZI, qn))
        lines.append("%s := %s(0, %s, %s):" % (rn, IRPRS, pn, qn))
        deg = port[("dbi", j)]["deg"]
        q_(("dbi", j, "deg"), "%s['Deg']" % rn)
        for i in range(deg + 1):
            q_(("dbi", j, "sr", i), "%s(%s, %d)" % (SRPRS, rn, i))

    # 4. Factorize
    for j, sc in enumerate(ex["factorize"]):
        d = build_sys([])
        qn = new("q")
        lines.append("%s := %s(%s, R):" % (qn, CPO, sc["q"]))
        if sc["ineq"]:
            lines.append("%s(%s, true):" % (INEQ, qn))
        rn = new("res")
        lines.append("%s := [%s(%s, %s)]:"
                     % (rn, DT % "Factorize", d, qn))
        query_split(("fac", j), d, rn)
        q_(("fac", j, "qsf"), "%s(%s)" % (SF, qn))

    # 5. Reduction
    for j, sc in enumerate(ex["reduction"]):
        d = build_sys(sc["ineqs"])
        qn = new("q")
        lines.append("%s := %s(%s, R):" % (qn, CPO, sc["q"]))
        if sc["ineq"]:
            lines.append("%s(%s, true):" % (INEQ, qn))
        rn = new("res")
        lines.append("%s := [%s(%s, %s)]:"
                     % (rn, DT % "Reduction", d, qn))
        query_split(("red", j), d, rn)
        q_(("red", j, "qsf"), "%s(%s)" % (SF, qn))

    # 6. InsertIntoQList / Strategy
    for j, ql in enumerate(ex["qlist"]):
        objs = []
        for e in ql:
            nm = new("e")
            if isinstance(e, tuple):
                lines.append("%s := %s(%s, R): %s(%s, true):"
                             % (nm, CPO, e[0], INEQ, nm))
            else:
                lines.append("%s := %s(%s, R):" % (nm, CPO, e))
            objs.append(nm)
        d = build_sys([])
        lines.append("%s['Q'] := %s([%s], []):"
                     % (d, DT % "InsertIntoQList", ",".join(objs)))
        q_(("ql", j, "order"),
           "map(o->[convert(%s(o),string), %s(o), %s(o)], %s['Q'])"
           % (DT % "Leader", INEQ, SF, d))
        q_(("ql", j, "idx"), "%s(%s)" % (DT % "Strategy", d))

    # 7. Criteria
    for j, sc in enumerate(ex["criteria"]):
        t = new("T")
        lf = new("lf")
        lines.append("%s := %s(R):" % (t, CJT))
        lines.append("%s := %s(%s, R): %s(%s, true):"
                     % (lf, CPO, sc["leaf"], NZI, lf))
        lines.append("%s(%s, %s):" % (DT % "InsertIntoJanetTrees", t, lf))
        dv = new("dv")
        lines.append("%s := %s(%s)[1]:"
                     % (dv, DT % "JanetTreesLeafs", t))
        qn = new("cq")
        lines.append("%s := %s(%s, [%s]):"
                     % (qn, DT % "MultiplePartialDerivative", dv,
                        ",".join(str(x) for x in sc["derive"])))
        q_(("crit", j), "%s(%s, %s, %s)"
           % (DT % "Criteria", qn, t, dv))

    lines.append("quit:")
    script = "\n".join(lines) + "\n"
    stdout, stderr = run_openmaple(script, timeout=3000)
    found = {int(m.group(1)): m.group(2) for m in _OMQ_RE.finditer(stdout)}
    missing = [i for i in range(len(labels)) if i not in found]
    if missing:
        raise RuntimeError("oracle answers missing for %s\n%s"
                           % (missing, stdout[-4000:]))
    ans = {labels[i]: found[i] for i in range(len(labels))}

    # =======================================================================
    # comparisons
    # =======================================================================

    def oracle_systems(label):
        eqDS = ans[(label, "dseq")]
        ineqDS = ans[(label, "dsineq")]
        ceq = split_maple_list(ans[(label, "ceq")])
        cineq = split_maple_list(ans[(label, "cineq")])
        systems = [sys_key_oracle(eqDS, ineqDS)]
        for e, i in zip(ceq, cineq):
            systems.append(sys_key_oracle(e, i))
        return sorted(systems)

    def cmp_split(group, count):
        for j in range(count):
            osys = oracle_systems((group, j))
            psys, pq = port[(group, j)]
            oq = canon_str(ans[(group, j, "qsf")])
            check((group, j, "systems"), osys == psys,
                  "\n  oracle=%s\n  port  =%s" % (osys, psys))
            check((group, j, "q"), oq == pq,
                  "oracle=%s port=%s" % (oq, pq))

    def cmp_prs(group, scenarios, has_gcd):
        for j in range(len(scenarios)):
            pd = port[(group, j)]
            check((group, j, "deg"),
                  int(ans[(group, j, "deg")]) == pd["deg"],
                  "oracle=%s port=%s" % (ans[(group, j, "deg")], pd["deg"]))
            for i in range(pd["deg"] + 1):
                osr = canon_str(ans[(group, j, "sr", i)])
                check((group, j, "subres", i), osr == pd["subres"][i],
                      "oracle=%s port=%s" % (osr, pd["subres"][i]))
            if has_gcd and pd["i0"] < pd["deg"]:
                og = canon_str(ans[(group, j, "gcd")])
                check((group, j, "gcd"), og == pd["gcd0"],
                      "oracle=%s port=%s" % (og, pd["gcd0"]))
            # port-side cofactor exact-quotient invariant (not oracle-gatable:
            # open-maple's CoFactorPRS write-back is absent)
            check((group, j, "cofactor-invariant"), pd["inv"],
                  "p != cofactor * gcd (up to scalar)")

    cmp_split("si", len(ex["split_by_initial"]))
    cmp_prs("sf", ex["split_by_squarefree"], has_gcd=True)
    cmp_prs("dbi", ex["divide_by_inequation"], has_gcd=False)
    cmp_split("fac", len(ex["factorize"]))
    cmp_split("red", len(ex["reduction"]))

    # Q order + strategy
    for j, ql in enumerate(ex["qlist"]):
        elems = split_maple_list(ans[("ql", j, "order")])
        oracle_order = []
        for e in elems:
            ldr, ineq, sf = split_maple_list(e)
            ldr = re.sub(r"\s+", "", ldr).strip('"')
            oracle_order.append((ldr, ineq.strip() == "true", canon_str(sf)))
        port_order, port_idx = port[("ql", j)]
        # normalise the port leader string ("u[1]") to spaceless
        pk = [(re.sub(r"\s+", "", l), b, c) for (l, b, c) in port_order]
        check(("ql", j, "order"), oracle_order == pk,
              "\n  oracle=%s\n  port  =%s" % (oracle_order, pk))
        check(("ql", j, "idx"),
              int(ans[("ql", j, "idx")]) == port_idx,
              "oracle=%s port=%s" % (ans[("ql", j, "idx")], port_idx))

    # Criteria
    for j in range(len(ex["criteria"])):
        overdict = ans[("crit", j)].strip() == "true"
        check(("crit", j), overdict == port[("crit", j)],
              "oracle=%s port=%s" % (ans[("crit", j)], port[("crit", j)]))

    print("SUMMARY passed=%d failed=%d" % (passed, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
