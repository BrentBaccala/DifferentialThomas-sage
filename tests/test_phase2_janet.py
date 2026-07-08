r"""
Phase-2 gate tests: Janet trees + derivation vs the open-maple oracle.

Run:  ~/miniforge3/envs/sage/bin/sage -python -m unittest discover -s tests -v

Each parity example runs in its OWN subprocess (tests/phase2_worker.py): the
substrate supports one live DifferentialPolynomialRing shape per process, and
the examples need different rings.  ex1-ex3 are the systems from
~/thomas-experiments; exG is the Gerdt/Blinkov 3-independent-variable system
cited in the reference source (derivation:64), which exercises tree branching
and multiplicative-variable flips (impossible with a single derivation).
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
        [sys.executable, os.path.join(REPO, "tests", "phase2_worker.py"),
         "--example", example],
        capture_output=True, text=True, cwd=REPO, timeout=3600)
    m = _SUMMARY_RE.search(proc.stdout)
    return proc, m


class JanetParityVsOracle(unittest.TestCase):
    """Insertion / divisor-search / multiplicative-variable / completion /
    derivation parity against the open-maple reference."""

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

    def test_exG(self):
        self._run("exG")


class UnitNoOracle(unittest.TestCase):
    """Pure-python unit checks (no oracle, no substrate ring)."""

    @staticmethod
    def _stub_leaf(name, nvars=1, mv=None):
        # a leaf: PolynomialObject with preloaded fields, no ring needed
        from differentialthomas.polyobj import PolynomialObject
        f = {"Ranking": "RK", "Polynom": name}
        if mv is not None:
            f["MultiplicativeVariables"] = list(mv)
        return PolynomialObject(f)

    def test_current_var(self):
        from differentialthomas import current_var
        self.assertEqual(current_var([0, -1, -1]), 1)
        self.assertEqual(current_var([2, 0, -1]), 2)
        self.assertEqual(current_var([2, 0, 3]), 3)
        self.assertEqual(current_var([-1, -1]), 1)   # reference -infinity case

    def test_tree_leafs_order_degree_before_variable(self):
        from differentialthomas import JanetNode, janet_tree_leafs
        root = JanetNode([0, -1])
        a = self._stub_leaf("a")
        b = self._stub_leaf("b")
        root.degree = JanetNode([1, -1])
        root.degree.degree = a          # deeper derivative
        root.variable = b               # next variable
        self.assertEqual(janet_tree_leafs(root), [a, b])
        self.assertEqual(janet_tree_leafs(a), [a])   # a leaf is its own list

    def test_remove_elements_in_subtree_and_delete_flag(self):
        from differentialthomas import JanetNode, remove_elements_in_subtree
        # root -Degree-> n1 -Degree-> leaf a ; togo=[1] cuts the whole
        # Degree subtree.  The reference does NOT flag the emptied root
        # deletable: its NULL-assignment unassigns the entry, defeating its
        # own `= NULL` test (oracle-confirmed quirk, see janet.py).
        root = JanetNode([0])
        n1 = JanetNode([1])
        a = self._stub_leaf("a")
        root.degree = n1
        n1.degree = a
        removed = remove_elements_in_subtree(root, [1])
        self.assertEqual(removed, [a])
        self.assertIsNone(root.degree)
        self.assertFalse(root.delete)
        # a genuinely empty node reached by the walk IS flagged
        empty = JanetNode([0])
        self.assertEqual(remove_elements_in_subtree(empty, [1]), [])
        self.assertTrue(empty.delete)
        # calling on a leaf removes nothing (reference tree:433)
        self.assertEqual(remove_elements_in_subtree(a, [1]), [])

    def test_deep_copy_shares_ranking_copies_leaves(self):
        from differentialthomas import JanetNode, deep_copy

        class FakeRanking(object):
            no_deep_copy = True
        rk = FakeRanking()
        leaf = self._stub_leaf("p", mv=[0])
        leaf.f["Ranking"] = rk
        root = JanetNode([0])
        root.degree = leaf
        c = deep_copy(root)
        self.assertIsNot(c, root)
        self.assertIsNot(c.degree, leaf)                       # leaf copied
        self.assertIs(c.degree.f["Ranking"], rk)               # ranking shared
        self.assertIsNot(c.degree.f["MultiplicativeVariables"],
                         leaf.f["MultiplicativeVariables"])    # mv list copied
        self.assertEqual(c.position, root.position)
        self.assertIsNot(c.position, root.position)


if __name__ == "__main__":
    unittest.main(verbosity=2)
