r"""
Phase-5 parity worker: the full differential Thomas decomposition (ProcInput +
DoNextStep + the work-queue main loop) vs the open-maple oracle, one example
per process (one live substrate ring per process).

Usage:  sage -python tests/phase5_worker.py --example ex1|ex2|ex3

Gate groups
===========

1. **Cell-for-cell decomposition parity.**  Run the port's
   ``differential_thomas_decomposition`` and the reference's
   ``DifferentialThomasDecomposition`` on the identical input + ranking.  Each
   cell's (equations, inequations) is reduced to a canonical multiset of
   polynomials **up to a nonzero scalar (content + sign)** -- the correct
   equivalence for a Thomas system's constraints and the equivalence under
   which the substrate's Ducos subresultants (sign-up-to-parity) match the
   oracle's SPRS.  The two decompositions are compared as an **unordered
   multiset of cells**.

2. **Lockstep DoNextStep parity** (ex1, ex2).  Drive the port and the reference
   main loops in lockstep from an identical initial ``SystemList`` and, after
   each iteration, compare the whole ``SystemList`` (each system's
   equations+inequations, as an unordered multiset of systems, each up to
   scalar).  This localises any divergence to the first differing step.

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
    run_openmaple, _MAPLE_JET_RE)

DT = "`DifferentialThomas/%s`"


# ---------------------------------------------------------------------------
# example definitions (jet / exponent form, as the reference consumes)
# ---------------------------------------------------------------------------

EXAMPLES = {
    "ex1": dict(
        ivar=["x"], dvar=["u"],
        eqs=["u[1]^2-4*u[0]"], ineqs=[],
        lockstep=True),
    "ex2": dict(
        ivar=["x"], dvar=["u", "a"],
        eqs=["a[0]*u[1]-u[0]", "a[1]"], ineqs=[],
        lockstep=True),
    "ex3": dict(
        ivar=["x"],
        dvar=["DDPs", "DPs", "Ps", "Vf",
              "a0", "a1", "b0", "b1", "c0", "c1", "V1"],
        eqs=[
            "Ps[1]-DPs[0]*Vf[1]",
            "DPs[1]-DDPs[0]*Vf[1]",
            "(a0[0]+a1[0]*Vf[0])*DDPs[0]+(b0[0]+b1[0]*Vf[0])*DPs[0]"
            "+(c0[0]+c1[0]*Vf[0])*Ps[0]",
            "Vf[0]-V1[0]*x",
            "a0[1]", "a1[1]", "b0[1]", "b1[1]", "c0[1]", "c1[1]", "V1[1]"],
        ineqs=[],
        lockstep=False),
}

# how many lockstep steps to gate (bounded so the oracle script stays cheap)
MAX_LOCKSTEP_STEPS = 40


def maple_to_blad(s, ivar):
    return _MAPLE_JET_RE.sub(
        lambda m: JetVar.from_maple(m.group(0)).to_blad_name(ivar), s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True, choices=sorted(EXAMPLES))
    args = ap.parse_args()
    ex = EXAMPLES[args.example]
    ivar, dvar = ex["ivar"], ex["dvar"]

    rk = dt.compute_ranking(ivar, dvar)
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

    # -- up-to-scalar canonical representative (content + sign) --------------
    def canon(dp):
        dp = R(dp)
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
        return canon(R(maple_to_blad(s, ivar)))

    def cell_key_port(system):
        eqs = tuple(sorted(canon(p)
                           for p in dt.differential_system_equations(system)))
        ineqs = tuple(sorted(
            canon(p) for p in dt.differential_system_inequations(system)))
        return (eqs, ineqs)

    # =======================================================================
    # 1. full decomposition -- PORT
    # =======================================================================
    eq_objs = [R(maple_to_blad(s, ivar)) for s in ex["eqs"]]
    ineq_objs = [R(maple_to_blad(s, ivar)) for s in ex["ineqs"]]
    port_cells = dt.differential_thomas_decomposition(eq_objs, ineq_objs, rk)
    port_keys = sorted(cell_key_port(c) for c in port_cells)

    # =======================================================================
    # 1. full decomposition -- ORACLE
    # =======================================================================
    ref_keys = sorted(run_reference_decomposition(ivar, dvar, ex["eqs"],
                                                   ex["ineqs"], canon_str))

    check("ncells", len(port_keys) == len(ref_keys),
          "port=%d ref=%d" % (len(port_keys), len(ref_keys)))
    check("cells-match", port_keys == ref_keys,
          _cell_diff(port_keys, ref_keys))

    print("INFO %s cells: port=%d ref=%d"
          % (args.example, len(port_keys), len(ref_keys)))

    # =======================================================================
    # 2. lockstep DoNextStep parity
    # =======================================================================
    if ex["lockstep"]:
        port_steps = run_port_lockstep(rk, eq_objs, ineq_objs, cell_key_port)
        ref_steps = run_reference_lockstep(ivar, dvar, ex["eqs"], ex["ineqs"],
                                           canon_str)
        nls = min(len(port_steps), len(ref_steps))
        check("lockstep-nsteps", len(port_steps) == len(ref_steps),
              "port=%d ref=%d" % (len(port_steps), len(ref_steps)))
        for i in range(nls):
            ok = port_steps[i] == ref_steps[i]
            check("lockstep-step%d" % (i + 1), ok,
                  "" if ok else _cell_diff(port_steps[i], ref_steps[i]))

    print("SUMMARY passed=%d failed=%d" % (passed, failed))
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# oracle drivers
# ---------------------------------------------------------------------------

_RANK = ("`DifferentialThomas/ComputeRanking`([%s],[%s]):"
         "\nR := `DifferentialThomas/GlobalRanking`:")

_DUMP_ONE = r"""
  eqs := `DifferentialThomas/DifferentialSystemEquations`(%(sys)s):
  for e in eqs do printf("%(tag)sEQ|%%d|%%s\n", %(idx)s, convert(e, string)): od:
  ineqs := `DifferentialThomas/DifferentialSystemInequations`(%(sys)s):
  for e in ineqs do printf("%(tag)sINEQ|%%d|%%s\n", %(idx)s, convert(e, string)): od:
"""


def run_reference_decomposition(ivar, dvar, eqs, ineqs, canon_str):
    """Run the reference DifferentialThomasDecomposition; return the list of
    cell keys (each ``(eq_tuple, ineq_tuple)`` of up-to-scalar canon strings)."""
    lines = ["with(DifferentialThomas):",
             _RANK % (",".join(ivar), ",".join(dvar))]
    lines.append("res := `DifferentialThomas/DifferentialThomasDecomposition`"
                 "([%s],[%s]):" % (",".join(eqs), ",".join(ineqs)))
    lines.append('printf("NCELLS|%d\\n", nops(res)):')
    lines.append("for i from 1 to nops(res) do")
    lines.append(_DUMP_ONE % dict(sys="res[i]", tag="", idx="i"))
    lines.append('  printf("CELLEND|%d\\n", i):')
    lines.append("od:")
    lines.append("quit:")
    stdout, _ = run_openmaple("\n".join(lines) + "\n")
    return _parse_cells(stdout, canon_str)


def run_reference_lockstep(ivar, dvar, eqs, ineqs, canon_str):
    """Drive the reference main loop (ProcInput + DoNextStep) manually,
    dumping the whole SystemList after each iteration.  Returns a list of
    per-step states; each state is a sorted multiset of cell keys."""
    lines = ["with(DifferentialThomas):",
             _RANK % (",".join(ivar), ",".join(dvar))]
    # ProcInput installs the option table + StandardFormSimplify etc.
    lines.append("pi := [`DifferentialThomas/ProcInput`([%s],[%s])]:"
                 % (",".join(eqs), ",".join(ineqs)))
    lines.append("SL := pi[1]:")
    lines.append("step := 0:")
    lines.append("while SL <> [] and step < %d do" % MAX_LOCKSTEP_STEPS)
    lines.append("  cur := SL[1]:")
    lines.append("  if cur['Inconsistent']=true then")
    lines.append("    SL := [op(2..nops(SL), SL)]:")
    lines.append("  elif cur['Finished']=true then")
    lines.append("    `DifferentialThomas/DifferentialSystemTailReduction`"
                 "(cur):")
    lines.append("    if cur['Finished']=true then "
                 "SL := [op(2..nops(SL), SL)]: fi:")
    lines.append("  else")
    lines.append("    new := [`DifferentialThomas/DoNextStep`(cur, [])]:")
    lines.append("    SL := [op(new), op(2..nops(SL), SL)]:")
    lines.append("  fi:")
    lines.append("  step := step + 1:")
    lines.append('  printf("STEP|%d|%d\\n", step, nops(SL)):')
    lines.append("  for si from 1 to nops(SL) do")
    lines.append(_DUMP_ONE % dict(sys="SL[si]", tag="S", idx="si"))
    lines.append('    printf("SYSEND|%d\\n", si):')
    lines.append("  od:")
    lines.append('  printf("STEPEND|%d\\n", step):')
    lines.append("od:")
    lines.append("quit:")
    stdout, _ = run_openmaple("\n".join(lines) + "\n")
    return _parse_lockstep(stdout, canon_str)


# ---------------------------------------------------------------------------
# port lockstep driver
# ---------------------------------------------------------------------------

def run_port_lockstep(rk, eq_objs, ineq_objs, cell_key_port):
    """Drive the port main loop, capturing the SystemList after each step."""
    steps = []

    def on_step(system_list):
        steps.append(sorted(cell_key_port(s) for s in system_list))
        if len(steps) >= MAX_LOCKSTEP_STEPS:
            raise _StopLockstep()

    system_list = dt.proc_input(eq_objs, ineq_objs, rk)
    try:
        dt.differential_thomas_decomposition(
            None, None, rk, system_list=system_list, on_step=on_step)
    except _StopLockstep:
        pass
    return steps


class _StopLockstep(Exception):
    pass


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------

def _parse_cells(stdout, canon_str):
    cells = {}
    for ln in stdout.splitlines():
        if ln.startswith("EQ|"):
            _, idx, poly = ln.split("|", 2)
            cells.setdefault(int(idx), ([], []))[0].append(canon_str(poly))
        elif ln.startswith("INEQ|"):
            _, idx, poly = ln.split("|", 2)
            cells.setdefault(int(idx), ([], []))[1].append(canon_str(poly))
    keys = []
    for idx in sorted(cells):
        eqs, ineqs = cells[idx]
        keys.append((tuple(sorted(eqs)), tuple(sorted(ineqs))))
    return keys


def _parse_lockstep(stdout, canon_str):
    steps = []
    cur = {}
    for ln in stdout.splitlines():
        if ln.startswith("STEP|"):
            cur = {}
        elif ln.startswith("SEQ|"):
            _, idx, poly = ln.split("|", 2)
            cur.setdefault(int(idx), ([], []))[0].append(canon_str(poly))
        elif ln.startswith("SINEQ|"):
            _, idx, poly = ln.split("|", 2)
            cur.setdefault(int(idx), ([], []))[1].append(canon_str(poly))
        elif ln.startswith("STEPEND|"):
            state = []
            for si in sorted(cur):
                eqs, ineqs = cur[si]
                state.append((tuple(sorted(eqs)), tuple(sorted(ineqs))))
            steps.append(sorted(state))
    return steps


def _cell_diff(a, b):
    """A short human-readable diff of two cell-key multisets."""
    sa, sb = set(a), set(b)
    only_port = [c for c in a if c not in sb]
    only_ref = [c for c in b if c not in sa]
    parts = []
    if only_port:
        parts.append("PORT-ONLY: %s" % (only_port[:3],))
    if only_ref:
        parts.append("REF-ONLY: %s" % (only_ref[:3],))
    return " | ".join(parts)[:2000]


if __name__ == "__main__":
    sys.exit(main())
