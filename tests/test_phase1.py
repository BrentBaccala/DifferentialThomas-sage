r"""
Phase-1 gate tests: object model + ranking + oracle harness.

Run:  ~/miniforge3/envs/sage/bin/sage -python -m unittest discover -s tests -v
(from the repo root; or ``sage -python tests/test_phase1.py``)

Each parity example runs in its OWN subprocess (tests/phase1_worker.py):
the substrate supports one live DifferentialPolynomialRing shape per process,
and ex1/ex2/ex3 need different rings.
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
        [sys.executable, os.path.join(REPO, "tests", "phase1_worker.py"),
         "--example", example],
        capture_output=True, text=True, cwd=REPO, timeout=3600)
    m = _SUMMARY_RE.search(proc.stdout)
    return proc, m


class ParityVsOracle(unittest.TestCase):
    """Accessor + ranking parity against the open-maple reference."""

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


class OracleHarness(unittest.TestCase):
    """Gate items (i) and (ii): trace capture and internal-proc Exec."""

    def test_exec_internal_proc(self):
        from differentialthomas.oracle import DTOracle
        o = DTOracle(["x"], ["u"],
                     setup=["p := `DifferentialThomas/CreatePolynomialObject`"
                            "(u[1]^2-4*u[0], R)"])
        leader, rank, ini = o.query([
            o.exec_proc("Leader", "p"),
            o.exec_proc("Rank", "p"),
            o.exec_proc("Initial", "p"),
        ])
        self.assertEqual(leader.replace(" ", ""), "u[1]")
        self.assertEqual(rank, "2")
        self.assertEqual(ini, "1")

    def test_reduction_trace_capture(self):
        from differentialthomas.oracle import capture_reduction_trace
        script = (
            "with(DifferentialThomas):\n"
            "Ranking([x], [u]):\n"
            "res := DifferentialThomasDecomposition("
            "[diff(u(x),x)^2 - 4*u(x)], []):\n"
            'printf("cells=%d\\n", nops(res)):\n'
            "quit:\n")
        stdout, trace = capture_reduction_trace(script, procs="1")
        self.assertIn("cells=2", stdout)
        self.assertTrue(trace, "no [red] trace lines captured")
        self.assertTrue(all(ln.startswith("[red]") for ln in trace))


class UnitNoOracle(unittest.TestCase):
    """Pure-python unit checks (no oracle, no ring construction)."""

    def test_jetvar_roundtrip(self):
        from differentialthomas import JetVar
        j = JetVar("u", (1, 2))
        self.assertEqual(str(j), "u[1,2]")
        self.assertEqual(j.to_blad_name(["x", "y"]), "u[x,y,y]")
        self.assertEqual(JetVar.from_blad_name("u[x,y,y]", ["x", "y"]), j)
        self.assertEqual(JetVar.from_maple("u[1, 2]"), j)
        self.assertEqual(JetVar.from_blad_name("u", ["x", "y"]),
                         JetVar("u", (0, 0)))
        self.assertEqual(j.order, 3)

    def test_lcm_list_and_deep_copy_shares_ranking(self):
        from differentialthomas import lcm_list, deep_copy
        self.assertEqual(lcm_list([1, 0, 2], [0, 3, 2]), [1, 3, 2])

        class FakeRanking(object):
            no_deep_copy = True
        r = FakeRanking()
        payload = [r, {"a": [r, 1]}, "s"]
        c = deep_copy(payload)
        self.assertIsNot(c, payload)
        self.assertIs(c[0], r)                  # shared: NoDeepCopy
        self.assertIs(c[1]["a"][0], r)          # shared even when nested
        self.assertIsNot(c[1], payload[1])      # dict copied

    def test_degrevlex_compare_pure(self):
        # comparator only; no substrate ring is built
        from differentialthomas import compute_ranking, JetVar
        rk = compute_ranking(["x", "y"], ["u", "v"], set_global=False)
        u10, u01 = JetVar("u", (1, 0)), JetVar("u", (0, 1))
        v10 = JetVar("v", (1, 0))
        self.assertTrue(rk.compare(u10, u01))    # x-derivative higher
        self.assertFalse(rk.compare(u01, u10))
        self.assertTrue(rk.compare(v10, u01))    # exps beat function
        self.assertTrue(rk.compare(JetVar("u", (0, 0)), JetVar("v", (0, 0))))
        self.assertFalse(rk.compare(1, u10))     # field sentinel loses
        self.assertTrue(rk.compare(u10, 1))
        self.assertFalse(rk.compare(1, 1))

    def test_substitute_polynom_keepset(self):
        # exercise invalidation semantics without a real polynomial: stub
        # ranking + preloaded fields
        from differentialthomas.polyobj import PolynomialObject
        po = PolynomialObject({
            "Ranking": "RK", "Polynom": "P0", "Leader": "L",
            "Initial": "I", "Inequation": True, "NonZeroInitial": True,
        })
        po.substitute_polynom("P1", keep=("Inequation",))
        self.assertEqual(po.f["Polynom"], "P1")
        self.assertEqual(po.f["Ranking"], "RK")
        self.assertEqual(po.f["Inequation"], True)   # kept
        self.assertNotIn("Leader", po.f)             # invalidated
        self.assertNotIn("Initial", po.f)
        self.assertNotIn("NonZeroInitial", po.f)     # NOT protected


if __name__ == "__main__":
    unittest.main(verbosity=2)
