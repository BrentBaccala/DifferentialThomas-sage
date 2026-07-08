r"""
Phase-2 parity worker: Janet trees + derivation vs the open-maple oracle,
one example per process (single live substrate ring per process).

Usage:  sage -python tests/phase2_worker.py --example ex1|ex2|ex3|exG

Protocol (mirrors how the engine uses the trees -- only Janet-irreducible
elements are ever inserted):

For each input polynomial, in order:
  1. probe ``JanetDivisorInTrees`` -- the found/not-found verdict (and which
     leaf, by leader) is compared against the oracle;
  2. if no divisor was found (and the element is not a differential-field
     element), ``InsertIntoJanetTrees`` it; the returned set (removed leaves
     + required prolongations, as (Leader, StandardForm) pairs, in order) and
     the post-insertion tree state (``JanetTreesLeafs`` leaders +
     ``MultiplicativeVariables`` vectors, in order) are compared.
Then: optional follow-up insertions of *returned prolongation objects*
(exercising insertion of cached PartialDerivative objects), a probe battery
of ``JanetDivisorInTrees`` verdicts over jets derived from the final leaves,
a direct ``RemoveElementsInSubtree`` check, a manual-multiplicative-variable
``CompleteElementInJanetTree`` check (including the considered-prolongation
idempotence), and the ``PartialDerivative`` / ``MultiplePartialDerivative``
metadata checks (Leader / StandardForm / Ancestor / NonZeroInitial /
ConsideredProlongations-reset-on-cache-hit).

ex1-ex3 have a single independent variable (chain trees, no multiplicative-
variable flips); ``exG`` is the Gerdt/Blinkov system cited in the reference
source (``derivation:64``) -- ``u[1,1,3]-u[4,0,0], u[5,1,0]-u[0,4,0],
u[0,6,0], u[4,2,0]`` over ``[x,y,z]`` with DegRevLex -- which exercises tree
branching, ``RemoveMultiplicativeVariableInSubtree`` flips (with their
emitted prolongations) and non-multiplicative completion.

Output: ``CHECK <label> OK|FAIL ...`` lines and ``SUMMARY passed=N failed=M``.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from differentialthomas import (                     # noqa: E402
    INFINITY, JetVar, compute_ranking, create_polynomial_object,
    create_janet_trees_object, janet_divisor_in_trees, janet_trees_leafs,
    insert_into_janet_trees, remove_elements_in_subtree,
    complete_element_in_janet_tree, partial_derivative,
    multiple_partial_derivative)
from differentialthomas.oracle import (              # noqa: E402
    DTOracle, poly_strings_equal, normalize_maple_name, _MAPLE_JET_RE)

DT = "`DifferentialThomas/%s`"


EXAMPLES = {
    "ex1": dict(
        ivar=["x"], dvar=["u"],
        polys=[
            "u[1]^2-4*u[0]",             # inserted (leader u[1])
            "2*u[1]*u[2]-4*u[1]",        # divisor u[1] found -> skipped
            "u[0]",                      # order-zero: replaces the tree
            "5",                         # field element: NULL verdict
            "3*u[1]-6*u[0]",             # divisor u[0] (leaf root) -> skipped
        ],
        followups=[],
        remove=("u", [1]),               # root is a leaf: removes nothing
        deriv_target=0, mpd_index=[2],
        complete_target=0, complete_mv=[0]),
    "ex2": dict(
        ivar=["x"], dvar=["u", "a"],
        polys=[
            "a[0]*u[1]-u[0]",            # inserted (u-tree)
            "a[1]",                      # inserted (a-tree)
            "a[0]",                      # order-zero: removes a[1]
            "a[1]*u[1]+a[0]*u[2]-u[1]",  # divisor u[1] -> skipped
            "u[0]^2-a[0]",               # order-zero: removes u[1]
        ],
        followups=[],
        remove=("u", [1]),
        deriv_target=0, mpd_index=[2],
        complete_target=0, complete_mv=[0]),
    "ex3": dict(
        ivar=["x"],
        dvar=["DDPs", "DPs", "Ps", "Vf",
              "a0", "a1", "b0", "b1", "c0", "c1", "V1"],
        polys=[
            "Ps[1]-DPs[0]*Vf[1]",
            "DPs[1]-DDPs[0]*Vf[1]",
            "(a0[0]+a1[0]*Vf[0])*DDPs[0]+(b0[0]+b1[0]*Vf[0])*DPs[0]"
            "+(c0[0]+c1[0]*Vf[0])*Ps[0]",             # order-zero leader
            "Vf[0]-V1[0]*x",                          # order-zero leader
            "a0[1]",
            "V1[1]",
        ],
        followups=[],
        remove=("DPs", [1]),
        deriv_target=3,                  # d/dx(Vf[0]-V1[0]*x): x in a coeff
        mpd_index=[2],
        complete_target=0, complete_mv=[0]),
    # Gerdt/Blinkov (reference derivation:64): branching + multvar flips
    "exG": dict(
        ivar=["x", "y", "z"], dvar=["u"],
        polys=[
            "u[1,1,3]-u[4,0,0]",
            "u[5,1,0]-u[0,4,0]",         # flips x on u[1,1,3] -> prolongation
            "u[0,6,0]",                  # gets x non-multiplicative
            "u[4,2,0]",                  # gets x non-multiplicative
        ],
        followups=[(1, 0)],              # insert the u[2,1,3] prolongation
        remove=("u", [1, 0, 0]),         # cuts the whole x>=1 subtree
        deriv_target=0, mpd_index=[1, 0, 1],
        complete_target=1, complete_mv=[0, "infinity", 0]),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def maple_to_blad(s, ivar):
    """Exponent-jets -> BLAD derivation-jets (``u[1,2]`` -> ``u[x,y,y]``)."""
    def repl(m):
        return JetVar.from_maple(m.group(0)).to_blad_name(ivar)
    return _MAPLE_JET_RE.sub(repl, s)


def split_maple_list(s):
    """Split a (possibly nested) Maple list string at top-level commas."""
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
    """Port MultiplicativeVariables list -> canonical Maple string."""
    return "[%s]" % ",".join(
        "infinity" if m == INFINITY else str(int(m)) for m in mv)


def verdict_str(divisor):
    return "[]" if divisor is None else "[%s]" % (divisor.leader(),)


def cmv(entry):
    return INFINITY if entry == "infinity" else int(entry)


def probe_jets_from_leaves(leaves, rk, limit=40):
    """Deterministic probe battery derived from the final leaves."""
    n = len(rk.ivar)
    out, seen = [], set()

    def add(jv):
        if all(e >= 0 for e in jv.exps) and jv not in seen:
            seen.add(jv)
            out.append(jv)

    for leaf in leaves:
        v = leaf.leader()
        add(v)
        for i in range(n):
            add(JetVar(v.head, tuple(
                e + (1 if k == i else 0) for k, e in enumerate(v.exps))))
            add(JetVar(v.head, tuple(
                e - (1 if k == i else 0) for k, e in enumerate(v.exps))))
        add(JetVar(v.head, (v.exps[0] + 2,) + v.exps[1:]))
    for head in rk.dvar:
        add(JetVar(head, (0,) * n))
    return out[:limit]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True, choices=sorted(EXAMPLES))
    args = ap.parse_args()
    ex = EXAMPLES[args.example]
    ivar, dvar = ex["ivar"], ex["dvar"]
    x0 = ivar[0]

    rk = compute_ranking(ivar, dvar)
    objs = [create_polynomial_object(maple_to_blad(p, ivar), rk)
            for p in ex["polys"]]

    passed = failed = 0

    def check(label, ok, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print("CHECK %s OK" % (label,))
        else:
            failed += 1
            print("CHECK %s FAIL %s" % (label, detail))

    # -- port side: run the whole scenario, recording expectations -----------
    trees = create_janet_trees_object(rk)
    steps = []           # (verdict, inserted, result-objs or None, leaves)
    step_results = []
    for o in objs:
        d = janet_divisor_in_trees(trees, o)
        inserted = d is None and o.leader() != 1
        if inserted:
            r = insert_into_janet_trees(trees, o)
            leaves = [(str(q.leader()), mv_str(q.f["MultiplicativeVariables"]),
                       q) for q in janet_trees_leafs(trees)]
        else:
            r, leaves = None, None
        steps.append((verdict_str(d), inserted, r, leaves))
        step_results.append(r)

    followups = []
    for (si, ei) in ex["followups"]:
        o = step_results[si][ei]
        d = janet_divisor_in_trees(trees, o)
        assert d is None, "followup element unexpectedly has a divisor"
        r = insert_into_janet_trees(trees, o)
        leaves = [(str(q.leader()), mv_str(q.f["MultiplicativeVariables"]), q)
                  for q in janet_trees_leafs(trees)]
        followups.append((verdict_str(d), r, leaves))

    probes = probe_jets_from_leaves(janet_trees_leafs(trees), rk)
    probe_verdicts = []
    for jv in probes:
        po = create_polynomial_object(maple_to_blad(str(jv), ivar), rk)
        probe_verdicts.append(verdict_str(janet_divisor_in_trees(trees, po)))
    fe = create_polynomial_object(rk.ring(7), rk)      # field-element probe
    fe_verdict = verdict_str(janet_divisor_in_trees(trees, fe))

    # RemoveElementsInSubtree (direct check; dead code in the reference)
    rem_head, rem_togo = ex["remove"]
    rem = remove_elements_in_subtree(trees[rem_head], list(rem_togo))
    rem_pairs = [(str(q.leader()), q.standard_form()) for q in rem]
    rem_leaves = [(str(q.leader()), mv_str(q.f["MultiplicativeVariables"]))
                  for q in janet_trees_leafs(trees)]
    # a PolynomialObject leaf root has no delete flag (oracle: unassigned key)
    rem_delete = getattr(trees[rem_head], "delete", False)

    # CompleteElementInJanetTree with manual multiplicative variables
    qc = create_polynomial_object(
        maple_to_blad(ex["polys"][ex["complete_target"]], ivar), rk)
    qc.f["MultiplicativeVariables"] = [cmv(m) for m in ex["complete_mv"]]
    c1 = complete_element_in_janet_tree(qc)
    c2 = complete_element_in_janet_tree(qc)
    assert c2 == [], "second completion must be empty (all considered)"

    # PartialDerivative / MultiplePartialDerivative metadata
    qd = create_polynomial_object(
        maple_to_blad(ex["polys"][ex["deriv_target"]], ivar), rk)
    dp = partial_derivative(qd, x0)
    dp2 = partial_derivative(qd, x0)
    check(("pd-cache-identity",), dp2 is dp)
    cpflag = qd.f["ConsideredProlongations"][x0]
    mp = multiple_partial_derivative(qd, ex["mpd_index"])

    # -- oracle script (same operation order) ---------------------------------
    setup = ["p%d := %s(%s, R)" % (i, DT % "CreatePolynomialObject", p)
             for i, p in enumerate(ex["polys"])]
    setup.append("T := %s(R)" % (DT % "CreateJanetTreesObject",))
    ldr = "%s" % (DT % "Leader",)
    sf = "%s" % (DT % "StandardForm",)
    pair_map = "map(q->[%s(q), %s(q)], %%s)" % (ldr, sf)
    leaf_map = ("map(q->[%s(q), q['MultiplicativeVariables']], %s(T))"
                % (ldr, DT % "JanetTreesLeafs"))
    for i, (_v, inserted, _r, _l) in enumerate(steps):
        setup.append("dv%d := map(q->%s(q), [%s(T, p%d)])"
                     % (i, ldr, DT % "JanetDivisorInTrees", i))
        if inserted:
            setup.append("r%d := %s(T, p%d)"
                         % (i, DT % "InsertIntoJanetTrees", i))
            setup.append("ins%d := %s" % (i, pair_map % ("r%d" % i)))
            setup.append("snap%d := %s" % (i, leaf_map))
    for j, (si, ei) in enumerate(ex["followups"]):
        setup.append("fp%d := r%d[%d]" % (j, si, ei + 1))
        setup.append("dvf%d := map(q->%s(q), [%s(T, fp%d)])"
                     % (j, ldr, DT % "JanetDivisorInTrees", j))
        setup.append("rf%d := %s(T, fp%d)"
                     % (j, DT % "InsertIntoJanetTrees", j))
        setup.append("insf%d := %s" % (j, pair_map % ("rf%d" % j)))
        setup.append("snapf%d := %s" % (j, leaf_map))
    for k, jv in enumerate(probes):
        setup.append("dpr%d := map(q->%s(q), [%s(T, "
                     "%s(%s, R))])"
                     % (k, ldr, DT % "JanetDivisorInTrees",
                        DT % "CreatePolynomialObject", jv))
    setup.append("dfe := map(q->%s(q), [%s(T, %s(7, R))])"
                 % (ldr, DT % "JanetDivisorInTrees",
                    DT % "CreatePolynomialObject"))
    setup.append("rem := %s(T[%s], %s)"
                 % (DT % "RemoveElementsInSubtree", rem_head,
                    "[%s]" % ",".join(str(t) for t in rem_togo)))
    setup.append("rempairs := %s" % (pair_map % "rem"))
    setup.append("remleaves := %s" % leaf_map)
    setup.append("remdel := evalb(assigned(T[%s]['Delete']) and "
                 "T[%s]['Delete']=true)" % (rem_head, rem_head))
    setup.append("qc := %s(%s, R)" % (DT % "CreatePolynomialObject",
                                      ex["polys"][ex["complete_target"]]))
    setup.append("qc['MultiplicativeVariables'] := [%s]"
                 % ",".join(str(m) for m in ex["complete_mv"]))
    setup.append("c1 := %s" % (pair_map
                               % ("%s(qc)" % (DT % "CompleteElementInJanetTree",))))
    setup.append("c2 := %s" % (pair_map
                               % ("%s(qc)" % (DT % "CompleteElementInJanetTree",))))
    setup.append("qd := %s(%s, R)" % (DT % "CreatePolynomialObject",
                                      ex["polys"][ex["deriv_target"]]))
    setup.append("dp := %s(qd, %s)" % (DT % "PartialDerivative", x0))
    setup.append("pdinfo := [%s(dp), %s(dp), %s(dp), %s(dp)]"
                 % (ldr, sf, DT % "Ancestor", DT % "NonZeroInitial"))
    setup.append("dp2 := %s(qd, %s)" % (DT % "PartialDerivative", x0))
    setup.append("cpflag := eval(qd['ConsideredProlongations'][%s])" % x0)
    setup.append("mp := %s(qd, [%s])"
                 % (DT % "MultiplePartialDerivative",
                    ",".join(str(t) for t in ex["mpd_index"])))
    setup.append("mpinfo := [%s(mp), %s(mp), %s(mp)]"
                 % (ldr, sf, DT % "Ancestor"))

    labels, queries = [], []

    def q(label, text):
        labels.append(label)
        queries.append(text)

    for i, (_v, inserted, _r, _l) in enumerate(steps):
        q(("dv", i), "dv%d" % i)
        if inserted:
            q(("ins", i), "ins%d" % i)
            q(("snap", i), "snap%d" % i)
    for j in range(len(ex["followups"])):
        q(("dvf", j), "dvf%d" % j)
        q(("insf", j), "insf%d" % j)
        q(("snapf", j), "snapf%d" % j)
    for k in range(len(probes)):
        q(("dpr", k), "dpr%d" % k)
    q(("dfe",), "dfe")
    q(("rempairs",), "rempairs")
    q(("remleaves",), "remleaves")
    q(("remdel",), "remdel")
    q(("c1",), "c1")
    q(("c2",), "c2")
    q(("pdinfo",), "pdinfo")
    q(("cpflag",), "cpflag")
    q(("mpinfo",), "mpinfo")

    oracle = DTOracle(ivar, dvar, setup=setup)
    answers = dict(zip(labels, oracle.query(queries)))

    # -- comparisons -----------------------------------------------------------

    def cmp_pairs(label, ans, port_objs):
        """Compare an oracle list of [Leader, StandardForm] pairs with a list
        of port PolynomialObjects, ordered."""
        elems = split_maple_list(ans)
        if len(elems) != len(port_objs):
            check(label, False, "length: oracle=%r port=%s"
                  % (ans, [str(o.leader()) for o in port_objs]))
            return
        for t, (e, o) in enumerate(zip(elems, port_objs)):
            lstr, pstr = split_maple_list(e)
            check(label + ("ldr", t),
                  normalize_maple_name(lstr) == str(o.leader()),
                  "oracle=%r port=%r" % (lstr, str(o.leader())))
            check(label + ("poly", t),
                  poly_strings_equal(pstr, o.standard_form(), rk),
                  "oracle=%r port=%r" % (pstr, str(o.standard_form())))

    def cmp_leaves(label, ans, port_leaves):
        elems = split_maple_list(ans)
        if len(elems) != len(port_leaves):
            check(label, False, "length: oracle=%r port=%s"
                  % (ans, [pl[0] for pl in port_leaves]))
            return
        for t, (e, pl) in enumerate(zip(elems, port_leaves)):
            lstr, mstr = split_maple_list(e)
            check(label + ("ldr", t), normalize_maple_name(lstr) == pl[0],
                  "oracle=%r port=%r" % (lstr, pl[0]))
            check(label + ("mv", t), normalize_maple_name(mstr) == pl[1],
                  "oracle=%r port=%r" % (mstr, pl[1]))

    for i, (verdict, inserted, r, leaves) in enumerate(steps):
        ans = answers[("dv", i)]
        check(("dv", i), normalize_maple_name(ans) == verdict,
              "oracle=%r port=%r" % (ans, verdict))
        if inserted:
            cmp_pairs(("ins", i), answers[("ins", i)], r)
            cmp_leaves(("snap", i), answers[("snap", i)], leaves)
    for j, (verdict, r, leaves) in enumerate(followups):
        ans = answers[("dvf", j)]
        check(("dvf", j), normalize_maple_name(ans) == verdict,
              "oracle=%r port=%r" % (ans, verdict))
        cmp_pairs(("insf", j), answers[("insf", j)], r)
        cmp_leaves(("snapf", j), answers[("snapf", j)], leaves)
    for k, jv in enumerate(probes):
        ans = answers[("dpr", k)]
        check(("dpr", str(jv)),
              normalize_maple_name(ans) == probe_verdicts[k],
              "oracle=%r port=%r" % (ans, probe_verdicts[k]))
    check(("dfe",), normalize_maple_name(answers[("dfe",)]) == fe_verdict,
          "oracle=%r port=%r" % (answers[("dfe",)], fe_verdict))

    cmp_pairs(("rempairs",), answers[("rempairs",)], rem)
    cmp_leaves(("remleaves",), answers[("remleaves",)], rem_leaves)
    check(("remdel",), (answers[("remdel",)] == "true") == bool(rem_delete),
          "oracle=%s port=%s" % (answers[("remdel",)], rem_delete))

    cmp_pairs(("c1",), answers[("c1",)], c1)
    cmp_pairs(("c2",), answers[("c2",)], c2)

    pdans = split_maple_list(answers[("pdinfo",)])
    check(("pd", "leader"),
          normalize_maple_name(pdans[0]) == str(dp.leader()),
          "oracle=%r port=%r" % (pdans[0], str(dp.leader())))
    check(("pd", "poly"), poly_strings_equal(pdans[1], dp.standard_form(), rk),
          "oracle=%r port=%r" % (pdans[1], str(dp.standard_form())))
    check(("pd", "ancestor"),
          normalize_maple_name(pdans[2])
          == "[%s]" % ",".join(str(t) for t in dp.ancestor()),
          "oracle=%r port=%r" % (pdans[2], dp.ancestor()))
    check(("pd", "nzi"), (pdans[3] == "true") == bool(dp.nonzero_initial()),
          "oracle=%s port=%s" % (pdans[3], dp.nonzero_initial()))
    check(("cpflag",),
          (answers[("cpflag",)] == "true") == bool(cpflag),
          "oracle=%s port=%s" % (answers[("cpflag",)], cpflag))
    mpans = split_maple_list(answers[("mpinfo",)])
    check(("mpd", "leader"),
          normalize_maple_name(mpans[0]) == str(mp.leader()),
          "oracle=%r port=%r" % (mpans[0], str(mp.leader())))
    check(("mpd", "poly"), poly_strings_equal(mpans[1], mp.standard_form(), rk),
          "oracle=%r port=%r" % (mpans[1], str(mp.standard_form())))
    check(("mpd", "ancestor"),
          normalize_maple_name(mpans[2])
          == "[%s]" % ",".join(str(t) for t in mp.ancestor()),
          "oracle=%r port=%r" % (mpans[2], mp.ancestor()))

    print("SUMMARY passed=%d failed=%d" % (passed, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
