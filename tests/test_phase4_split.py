r"""
Phase-4 gate tests: the split / factor / sort / strategy / passivity layer vs
the open-maple oracle.

Run:  ~/miniforge3/envs/sage/bin/sage -python -m unittest tests.test_phase4_split -v

Each parity example runs in its OWN subprocess (tests/phase4_worker.py): the
substrate supports one live DifferentialPolynomialRing shape per process.  The
worker gates, per example (see its docstring for the full protocol):

- **SplitByInitial** -- child systems + mutated original + reduced element;
- **SplitBySquarefree / DivideByInequation** -- the reference PRS engine
  (``SubResultantPRS`` chain + ``PRSGCD`` at the split index, up to sign) plus a
  port-side cofactor exact-quotient invariant.  The reference operators
  themselves cannot be executed by open-maple: ``CoFactorPRS`` needs a Maple
  pseudo-quotient write-back open-maple does not implement, so ``p['Polynom']``
  becomes an unassigned name and the operator stack-overflows.  The
  subresultant zero-pattern (which fixes the split index and the child's
  vanishing polynomial) IS the reference-checkable core and is gated exactly;
- **Factorize** -- factor-branch children (equation and inequation);
- **Reduction(DS, q)** -- reduced ``q`` and spawned children;
- **InsertIntoQList / Strategy** -- Q order and selected index;
- **Criteria** -- the involutive skip/no-skip verdict.

Constraint sets of spawned systems are compared up to a nonzero scalar
(content + sign) -- the correct equivalence for a Thomas system's constraints,
and the equivalence under which the substrate's Ducos subresultants
(sign-up-to-parity) match the oracle's SPRS.  Systems are compared as an
unordered multiset.
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
        [sys.executable, os.path.join(REPO, "tests", "phase4_worker.py"),
         "--example", example],
        capture_output=True, text=True, cwd=REPO, timeout=3600)
    m = _SUMMARY_RE.search(proc.stdout)
    return proc, m


class SplitParityVsOracle(unittest.TestCase):
    """Split / factor / sort / strategy / passivity parity vs open-maple."""

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
