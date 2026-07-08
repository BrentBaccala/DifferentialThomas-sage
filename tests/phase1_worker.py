r"""
Phase-1 parity worker: runs ONE example's accessor + ranking parity checks
against the open-maple oracle, in its own process (the substrate supports a
single live DifferentialPolynomialRing per process).

Usage:  sage -python tests/phase1_worker.py --example ex1|ex2|ex3

Output: one ``CHECK <label> OK|FAIL ...`` line per comparison and a final
``SUMMARY passed=<N> failed=<M>``.  Exit code 0 iff failed == 0.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from differentialthomas import (            # noqa: E402
    JetVar, compute_ranking, create_polynomial_object,
    is_differential_field_element)
from differentialthomas.oracle import (     # noqa: E402
    DTOracle, maple_leader_to_jetvar, poly_strings_equal, _MAPLE_JET_RE)


# Example corpus: the ex1-ex3 systems from ~/thomas-experiments, written in
# the reference's exponent-jet notation (as they appear inside the package
# after Diff2JetList), plus derived polynomials (prolongations, field
# elements, fractional coefficients) that exercise the accessor edge cases.
EXAMPLES = {
    "ex1": dict(
        ivar=["x"], dvar=["u"],
        polys=[
            "u[1]^2-4*u[0]",            # the singular ODE
            "2*u[1]*u[2]-4*u[1]",       # its x-prolongation
            "u[0]",                     # the singular cell's equation
            "5",                        # a field element
            "3*u[1]-6*u[0]",            # integer content survives StandardForm
        ]),
    "ex2": dict(
        ivar=["x"], dvar=["u", "a"],
        polys=[
            "a[0]*u[1]-u[0]",           # parametric ODE, a ranked lowest
            "a[1]",                     # constancy of a
            "a[0]",
            "a[1]*u[1]+a[0]*u[2]-u[1]",  # prolongation of the ODE
            "u[0]^2-a[0]",
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
            "Vf[0]-V1[0]*x",            # a derivation appearing polynomially
            "a0[1]",
            "V1[1]",
        ]),
}


def maple_to_blad(s, ivar):
    """Rewrite exponent-jets to BLAD derivation-jets:
    ``u[1]`` -> ``u[x]``, ``u[0]`` -> ``u`` (given ivar=['x'])."""
    def repl(m):
        return JetVar.from_maple(m.group(0)).to_blad_name(ivar)
    return _MAPLE_JET_RE.sub(repl, s)


def parse_int_list(s):
    s = s.strip()
    assert s.startswith("[") and s.endswith("]"), s
    body = s[1:-1].strip()
    if not body:
        return []
    return [int(t) for t in body.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True, choices=sorted(EXAMPLES))
    args = ap.parse_args()
    ex = EXAMPLES[args.example]

    rk = compute_ranking(ex["ivar"], ex["dvar"])
    objs = [create_polynomial_object(maple_to_blad(p, rk.ivar), rk)
            for p in ex["polys"]]

    # jet universe: all differential variables of the polys + one prolongation
    # step in every ivar direction (ranking parity must hold on jets the
    # engine will create, not only on jets present in the input)
    jets = []
    seen = set()
    for o in objs:
        for jv in o.diff_var_list():
            for cand in [jv] + [
                    JetVar(jv.head, tuple(
                        e + (1 if k == i else 0)
                        for k, e in enumerate(jv.exps)))
                    for i in range(len(rk.ivar))]:
                if cand not in seen:
                    seen.add(cand)
                    jets.append(cand)

    # -- build the single oracle batch ---------------------------------------
    setup = ["p%d := `DifferentialThomas/CreatePolynomialObject`(%s, R)"
             % (i, p) for i, p in enumerate(ex["polys"])]
    oracle = DTOracle(ex["ivar"], ex["dvar"], setup=setup)

    labels = []
    queries = []

    def q(label, text):
        labels.append(label)
        queries.append(text)

    for i in range(len(objs)):
        pi = "p%d" % i
        q(("sf", i), oracle.exec_proc("StandardForm", pi))
        q(("leader", i), oracle.exec_proc("Leader", pi))
        q(("rank", i), oracle.exec_proc("Rank", pi))
        q(("initial", i), oracle.exec_proc("Initial", pi))
        q(("separant", i), oracle.exec_proc("Separant", pi))
        q(("isfield", i), oracle.exec_proc("IsDifferentialFieldElement", pi))
        q(("maxorder", i), oracle.exec_proc("MaxOrder", pi))
        q(("ordld", i), oracle.exec_proc("OrderofLeader", pi))
        q(("nterms", i), oracle.exec_proc("NumberTerms", pi))
        q(("leadderiv", i), oracle.exec_proc("LeadingDerivation", pi))
        q(("leadfun", i), oracle.exec_proc("LeadingFunction", pi))
    for a in jets:
        q(("rlist", str(a)), "R['RankingList'](%s)" % a)
    for a in jets:
        for b in jets:
            q(("cmp", str(a), str(b)), "R['Compare'](%s, %s)" % (a, b))

    answers = oracle.query(queries)

    # -- compare --------------------------------------------------------------
    passed = failed = 0
    jet_by_repr = {str(j): j for j in jets}

    def check(label, ok, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print("CHECK %s OK" % (label,))
        else:
            failed += 1
            print("CHECK %s FAIL %s" % (label, detail))

    for label, ans in zip(labels, answers):
        kind = label[0]
        if kind in ("sf", "leader", "rank", "initial", "separant", "isfield",
                    "maxorder", "ordld", "nterms", "leadderiv", "leadfun"):
            o = objs[label[1]]
        if kind == "sf":
            check(label, poly_strings_equal(ans, o.standard_form(), rk),
                  "oracle=%r port=%r" % (ans, str(o.standard_form())))
        elif kind == "leader":
            want = maple_leader_to_jetvar(ans)
            check(label, want == o.leader(),
                  "oracle=%r port=%r" % (ans, str(o.leader())))
        elif kind == "rank":
            check(label, int(ans) == o.rank(),
                  "oracle=%s port=%s" % (ans, o.rank()))
        elif kind == "initial":
            check(label, poly_strings_equal(ans, o.initial(), rk),
                  "oracle=%r port=%r" % (ans, str(o.initial())))
        elif kind == "separant":
            check(label, poly_strings_equal(ans, o.separant(), rk),
                  "oracle=%r port=%r" % (ans, str(o.separant())))
        elif kind == "isfield":
            check(label, (ans == "true") == is_differential_field_element(o),
                  "oracle=%s port=%s" % (ans, is_differential_field_element(o)))
        elif kind == "maxorder":
            check(label, int(ans) == o.max_order(),
                  "oracle=%s port=%s" % (ans, o.max_order()))
        elif kind == "ordld":
            check(label, int(ans) == o.order_of_leader(),
                  "oracle=%s port=%s" % (ans, o.order_of_leader()))
        elif kind == "nterms":
            check(label, int(ans) == o.number_terms(),
                  "oracle=%s port=%s" % (ans, o.number_terms()))
        elif kind == "leadderiv":
            check(label, parse_int_list(ans) == o.leading_derivation(),
                  "oracle=%s port=%s" % (ans, o.leading_derivation()))
        elif kind == "leadfun":
            want = 1 if ans.strip() == "1" else ans.strip()
            check(label, want == o.leading_function(),
                  "oracle=%s port=%s" % (ans, o.leading_function()))
        elif kind == "rlist":
            jv = jet_by_repr[label[1]]
            check(label, parse_int_list(ans) == list(rk.ranking_list(jv)),
                  "oracle=%s port=%s" % (ans, rk.ranking_list(jv)))
        elif kind == "cmp":
            a = jet_by_repr[label[1]]
            b = jet_by_repr[label[2]]
            check(label, (ans == "true") == rk.compare(a, b),
                  "oracle=%s port=%s" % (ans, rk.compare(a, b)))
        else:
            raise AssertionError(kind)

    print("SUMMARY passed=%d failed=%d" % (passed, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
