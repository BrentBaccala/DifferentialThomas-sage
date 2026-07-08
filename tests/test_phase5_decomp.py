r"""
Phase-5 gate tests: the full differential Thomas decomposition (ProcInput +
DoNextStep + the work-queue main loop) vs the open-maple oracle -- cell for
cell.

Run:  ~/miniforge3/envs/sage/bin/sage -python -m unittest tests.test_phase5_decomp -v

Each example runs in its OWN subprocess (tests/phase5_worker.py): the substrate
supports one live DifferentialPolynomialRing shape per process.  The worker
gates, per example:

- **cell-for-cell decomposition parity** -- the port's
  ``differential_thomas_decomposition`` and the reference's
  ``DifferentialThomasDecomposition`` on identical input produce the same
  unordered multiset of cells; each cell's (equations, inequations) is compared
  as a multiset of polynomials up to a nonzero scalar (content + sign), modulo
  the jet-name normaliser and the resultant sign convention;
- **lockstep DoNextStep parity** (ex1, ex2) -- stepping the port and the
  reference main loops from an identical initial SystemList, the whole
  SystemList agrees (as an unordered multiset of systems) after every step.

Cell counts (port vs reference), per example, are reported on the ``INFO`` line.
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
        [sys.executable, os.path.join(REPO, "tests", "phase5_worker.py"),
         "--example", example],
        capture_output=True, text=True, cwd=REPO, timeout=3600)
    m = _SUMMARY_RE.search(proc.stdout)
    return proc, m


class DecompositionParityVsOracle(unittest.TestCase):
    """Full-decomposition cell-for-cell parity vs open-maple."""

    def _run(self, example):
        proc, m = run_worker(example)
        fails = "\n".join(ln for ln in proc.stdout.splitlines()
                          if " FAIL" in ln)
        info = "\n".join(ln for ln in proc.stdout.splitlines()
                         if ln.startswith("INFO"))
        self.assertIsNotNone(
            m, "no SUMMARY from worker (rc=%d)\nstdout:\n%s\nstderr:\n%s"
            % (proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:]))
        passed, failed = int(m.group(1)), int(m.group(2))
        self.assertEqual(failed, 0, "parity failures:\n%s\n%s" % (info, fails))
        self.assertGreater(passed, 0)
        print("[%s] %s -- parity %d/%d" % (example, info, passed,
                                           passed + failed))

    def test_ex1(self):
        self._run("ex1")

    def test_ex2(self):
        self._run("ex2")

    def test_ex3(self):
        self._run("ex3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
