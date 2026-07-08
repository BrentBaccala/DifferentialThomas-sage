r"""
Phase-3 gate tests: the reduction engine vs the open-maple oracle.

Run:  ~/miniforge3/envs/sage/bin/sage -python -m unittest tests.test_phase3_reduction -v

Each parity example runs in its OWN subprocess (tests/phase3_worker.py): the
substrate supports one live DifferentialPolynomialRing shape per process.
The worker gates results (head/initial reduction, tail reduction in all four
modes, direct DifferentialPseudoReduction / PseudoRemainder incl. the
multiplier and the cofactor invariant, SimplifyPolynom content removal) AND
the full reduction trajectory (``OPENMAPLE_REDUCTION_TRACE`` fingerprints,
entry by entry).

The ``maxsizemultiplier`` size cap cannot be oracle-gated (open-maple has no
``length`` builtin, so the reference's cap condition is inert and never
fires); it gets a port-side unit test here instead, in a subprocess of its
own since it needs a live ring.
"""

import os
import re
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_SUMMARY_RE = re.compile(r"^SUMMARY passed=(\d+) failed=(\d+)$", re.M)


def run_worker(example):
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "tests", "phase3_worker.py"),
         "--example", example],
        capture_output=True, text=True, cwd=REPO, timeout=3600)
    m = _SUMMARY_RE.search(proc.stdout)
    return proc, m


class ReductionParityVsOracle(unittest.TestCase):
    """Result + trajectory parity against the open-maple reference."""

    def _run(self, example):
        proc, m = run_worker(example)
        fails = "\n".join(ln for ln in proc.stdout.splitlines()
                          if " FAIL" in ln)
        self.assertIsNotNone(
            m, "no SUMMARY from worker (rc=%d)\nstdout:\n%s\nstderr:\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
        passed, failed = int(m.group(1)), int(m.group(2))
        self.assertEqual(failed, 0, "parity failures:\n%s" % fails)
        self.assertGreater(passed, 0)
        print("[%s] parity %d/%d" % (example, passed, passed + failed))

    def test_ex1(self):
        self._run("ex1")

    def test_ex2(self):
        self._run("ex2")

    def test_ex3(self):
        self._run("ex3")


_CAP_SCRIPT = r"""
import sys
sys.path.insert(0, %r)
from differentialthomas import (
    compute_ranking, create_polynomial_object, create_janet_trees_object,
    insert_into_janet_trees, reduce_nonlinear_tail_wrt_janet_trees)

rk = compute_ranking(["x"], ["u"])
leaf = create_polynomial_object("u[x]^2-4*u", rk)
leaf.nonzero_initial(True)
T = create_janet_trees_object(rk)
insert_into_janet_trees(T, leaf)

# cap 0: the first size check must abandon the reduction and return the
# INPUT object untouched (reference: `return eval(p), 1`)
p = create_polynomial_object("u[x,x]^2+u[x]-u", rk)
before = str(p.standard_form())
res, f = reduce_nonlinear_tail_wrt_janet_trees(T, p, "denominator", 0)
assert res is p, "cap must return the input object"
assert str(res.standard_form()) == before, "cap must leave p unchanged"
assert str(f) == "1", "cap must return f = 1"

# no cap (default infinity): the same reduction proceeds
p2 = create_polynomial_object("u[x,x]^2+u[x]-u", rk)
res2, f2 = reduce_nonlinear_tail_wrt_janet_trees(T, p2, "denominator")
assert str(res2.standard_form()) != before, "uncapped reduction must reduce"

# generous cap: behaves like no cap
p3 = create_polynomial_object("u[x,x]^2+u[x]-u", rk)
res3, f3 = reduce_nonlinear_tail_wrt_janet_trees(T, p3, "denominator", 1000)
assert str(res3.standard_form()) == str(res2.standard_form())
assert str(f3) == str(f2)
print("CAP-OK")
"""


class UnitNoOracle(unittest.TestCase):
    """Port-side unit checks (no oracle)."""

    def test_maxsizemultiplier_cap(self):
        """The size cap (not oracle-gatable -- see module docstring)."""
        proc = subprocess.run(
            [sys.executable, "-c", _CAP_SCRIPT % REPO],
            capture_output=True, text=True, cwd=REPO, timeout=600)
        self.assertIn("CAP-OK", proc.stdout,
                      "rc=%d\nstdout:\n%s\nstderr:\n%s"
                      % (proc.returncode, proc.stdout[-2000:],
                         proc.stderr[-2000:]))

    def test_maple_copy_is_shallow_with_list_value_semantics(self):
        from differentialthomas.polyobj import PolynomialObject
        inner = {"x": True}
        mv = [0, 1]
        p = PolynomialObject({"Ranking": "RK", "Polynom": "P",
                              "MultiplicativeVariables": mv,
                              "ConsideredProlongations": inner})
        c = p.copy()
        self.assertIsNot(c, p)
        self.assertIsNot(c.f["MultiplicativeVariables"], mv)   # list = value
        self.assertEqual(c.f["MultiplicativeVariables"], mv)
        self.assertIs(c.f["ConsideredProlongations"], inner)   # table shared
        self.assertIs(c.f["Ranking"], p.f["Ranking"])

    def test_tail_arg_parsing(self):
        from differentialthomas.reduction import _parse_tail_args
        self.assertEqual(_parse_tail_args(()), (False, False))
        self.assertEqual(_parse_tail_args(("tail",)), (True, False))
        self.assertEqual(_parse_tail_args(("nonlineartail",)), (False, True))
        self.assertEqual(_parse_tail_args((("tail", False),
                                           "nonlineartail")), (False, True))
        with self.assertRaises(ValueError):
            _parse_tail_args(("bogus",))

    def test_gcd_normalization_units(self):
        """The oracle-stack gcd conventions (empirically established; see
        maplecas docstring) at the unit level."""
        from fractions import Fraction
        from differentialthomas.maplecas import gcd_normalization_unit
        F = Fraction
        # multivariate: primitive, positive DRL-leading coefficient.
        # terms of g0 = -12*a + 8*u over sorted names [a, u]: lead is a.
        terms = [(F(-12), {"a_DT_0": 1}), (F(8), {"u_DT_0": 1})]
        self.assertEqual(gcd_normalization_unit(terms, 2), F(-4))
        # univariate: monic (unit = leading coefficient)
        terms = [(F(4), {"u_DT_1": 1}), (F(-6), {})]
        self.assertEqual(gcd_normalization_unit(terms, 1), F(4))
        # both integer constants: |integer gcd| (unit = sign only)
        self.assertEqual(gcd_normalization_unit([(F(2), {})], 0, True), F(1))
        self.assertEqual(gcd_normalization_unit([(F(-2), {})], 0, True),
                         F(-1))
        # rational constants: monic -> 1
        self.assertEqual(gcd_normalization_unit([(F(1, 6), {})], 0, False),
                         F(1, 6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
