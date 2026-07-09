r"""
Phase-6 worker: drive the hydrogen ansatz (param_field=False -- parameters
ranked) through the port's differential Thomas decomposition and report the
cell count + branch/RSS profile.  Optionally cross-check the cells against the
open-maple reference decomposition.

Usage:
  sage -python tests/phase6_worker.py                 # port only, profile
  sage -python tests/phase6_worker.py --verify        # + oracle cell parity

The hydrogen input (jets high, 10 params low, all ranked) is
``~/thomas-experiments/ex4_hydrogen.mpl``.  ivar = [x, y, z].
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import differentialthomas as dt                        # noqa: E402
from differentialthomas.jetvar import JetVar           # noqa: E402
from differentialthomas.maplecas import (              # noqa: E402
    term_data, integer_content, drl_lead_coeff, _div_scalar)
from differentialthomas.oracle import (                # noqa: E402
    run_openmaple, _MAPLE_JET_RE)

IVAR = ["x", "y", "z"]
JETS = ["DDPs", "DPs", "Ps", "Vf", "rho"]
PARS = ["V1", "V2", "V3", "V4", "a0", "a1", "b0", "b1", "c0", "c1"]
DVAR = JETS + PARS

# hydrogen ansatz, jet form (ivar order [x,y,z]); order-0 jets written [0,0,0]
ANSATZ = [
    "Ps[1,0,0]-DPs[0,0,0]*Vf[1,0,0]",
    "Ps[0,1,0]-DPs[0,0,0]*Vf[0,1,0]",
    "Ps[0,0,1]-DPs[0,0,0]*Vf[0,0,1]",
    "DPs[1,0,0]-DDPs[0,0,0]*Vf[1,0,0]",
    "DPs[0,1,0]-DDPs[0,0,0]*Vf[0,1,0]",
    "DPs[0,0,1]-DDPs[0,0,0]*Vf[0,0,1]",
    "(a0[0,0,0]+a1[0,0,0]*Vf[0,0,0])*DDPs[0,0,0]"
    "+(b0[0,0,0]+b1[0,0,0]*Vf[0,0,0])*DPs[0,0,0]"
    "+(c0[0,0,0]+c1[0,0,0]*Vf[0,0,0])*Ps[0,0,0]",
    "Vf[0,0,0]-(V1[0,0,0]*x+V2[0,0,0]*y+V3[0,0,0]*z+V4[0,0,0]*rho[0,0,0])",
    "rho[0,0,0]^2-x^2-y^2-z^2",
]

# constancy of the 10 parameters: every first derivative wrt each ivar is zero
_EXP = {"x": "1,0,0", "y": "0,1,0", "z": "0,0,1"}
PCONST = ["%s[%s]" % (p, _EXP[v]) for p in PARS for v in IVAR]

# the exact list the reference's ex4 consumes: ansatz ++ pconst
EQS_MAPLE = ANSATZ + PCONST
INEQS_MAPLE = []


def maple_to_blad(s, ivar=IVAR):
    return _MAPLE_JET_RE.sub(
        lambda m: JetVar.from_maple(m.group(0)).to_blad_name(ivar), s)


def build_ranking():
    return dt.compute_ranking(IVAR, DVAR)


def canon_factory(rk):
    R = rk.ring

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
        return canon(R(maple_to_blad(s)))

    return canon, canon_str


def cell_key_port(system, canon):
    eqs = tuple(sorted(canon(p)
                       for p in dt.differential_system_equations(system)))
    ineqs = tuple(sorted(
        canon(p) for p in dt.differential_system_inequations(system)))
    return (eqs, ineqs)


def run_port(rk):
    R = rk.ring
    eq_objs = [R(maple_to_blad(s)) for s in EQS_MAPLE]
    ineq_objs = [R(maple_to_blad(s)) for s in INEQS_MAPLE]
    t0 = time.time()
    cells = dt.differential_thomas_decomposition(eq_objs, ineq_objs, rk)
    wall = time.time() - t0
    return cells, wall


# ---------------------------------------------------------------------------
# oracle reference decomposition (slow ~5 min)
# ---------------------------------------------------------------------------

_RANK = ("`DifferentialThomas/ComputeRanking`([%s],[%s]):"
         "\nR := `DifferentialThomas/GlobalRanking`:")

_DUMP_ONE = r"""
  eqs := `DifferentialThomas/DifferentialSystemEquations`(%(sys)s):
  for e in eqs do printf("EQ|%%d|%%s\n", %(idx)s, convert(e, string)): od:
  ineqs := `DifferentialThomas/DifferentialSystemInequations`(%(sys)s):
  for e in ineqs do printf("INEQ|%%d|%%s\n", %(idx)s, convert(e, string)): od:
"""


def run_reference(canon_str, timeout=1800):
    lines = ["with(DifferentialThomas):",
             _RANK % (",".join(IVAR), ",".join(DVAR))]
    lines.append(
        "res := `DifferentialThomas/DifferentialThomasDecomposition`"
        "([%s],[%s]):" % (",".join(EQS_MAPLE), ",".join(INEQS_MAPLE)))
    lines.append('printf("NCELLS|%d\\n", nops(res)):')
    lines.append("for i from 1 to nops(res) do")
    lines.append(_DUMP_ONE % dict(sys="res[i]", idx="i"))
    lines.append('  printf("CELLEND|%d\\n", i):')
    lines.append("od:")
    lines.append("quit:")
    stdout, _ = run_openmaple("\n".join(lines) + "\n", timeout=timeout)
    return _parse_cells(stdout, canon_str)


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
    return sorted(keys)


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="cross-check cells against the open-maple oracle")
    ap.add_argument("--dump-port", metavar="FILE",
                    help="run the port, write its cell keys as JSON to FILE")
    ap.add_argument("--dump-ref", metavar="FILE",
                    help="run the oracle reference, write cell keys to FILE")
    ap.add_argument("--compare", nargs=2, metavar=("PORT", "REF"),
                    help="compare two dumped cell-key JSON files")
    args = ap.parse_args()

    if args.compare:
        with open(args.compare[0]) as fh:
            port_keys = [tuple(tuple(x) for x in c) for c in json.load(fh)]
        with open(args.compare[1]) as fh:
            ref_keys = [tuple(tuple(x) for x in c) for c in json.load(fh)]
        port_keys.sort()
        ref_keys.sort()
        print("PORT_NCELLS|%d" % len(port_keys))
        print("REF_NCELLS|%d" % len(ref_keys))
        match = port_keys == ref_keys
        print("CELLS_MATCH|%s" % match)
        if not match:
            sp, sr = set(port_keys), set(ref_keys)
            only_port = [c for c in port_keys if c not in sr]
            only_ref = [c for c in ref_keys if c not in sp]
            print("PORT_ONLY|%d REF_ONLY|%d" % (len(only_port), len(only_ref)))
            for c in only_port[:5]:
                print("  PORT-ONLY EQS=%s INEQS=%s" % c)
            for c in only_ref[:5]:
                print("  REF-ONLY  EQS=%s INEQS=%s" % c)
        return 0 if match else 1

    rk = build_ranking()
    canon, canon_str = canon_factory(rk)

    if args.dump_ref:
        print("running oracle reference decomposition (slow)...")
        sys.stdout.flush()
        ref_keys = run_reference(canon_str)
        with open(args.dump_ref, "w") as fh:
            json.dump([[list(e), list(i)] for (e, i) in ref_keys], fh)
        print("REF_NCELLS|%d written to %s" % (len(ref_keys), args.dump_ref))
        return 0

    print("PHASE6 hydrogen param_field=False: %d eqs (%d ansatz + %d const), "
          "%d ivar, %d dvar (%d jets + %d params)"
          % (len(EQS_MAPLE), len(ANSATZ), len(PCONST),
             len(IVAR), len(DVAR), len(JETS), len(PARS)))
    sys.stdout.flush()

    cells, wall = run_port(rk)
    print("PORT_NCELLS|%d" % len(cells))
    print("PORT_WALL|%.1f" % wall)
    sys.stdout.flush()

    port_keys = sorted(cell_key_port(c, canon) for c in cells)

    if args.dump_port:
        with open(args.dump_port, "w") as fh:
            json.dump([[list(e), list(i)] for (e, i) in port_keys], fh)
        print("PORT cell keys written to %s" % args.dump_port)

    if args.verify:
        print("running oracle reference decomposition (slow)...")
        sys.stdout.flush()
        ref_keys = run_reference(canon_str)
        print("REF_NCELLS|%d" % len(ref_keys))
        match = port_keys == ref_keys
        print("CELLS_MATCH|%s" % match)
        if not match:
            only_port = [c for c in port_keys if c not in set(ref_keys)]
            only_ref = [c for c in ref_keys if c not in set(port_keys)]
            print("PORT_ONLY|%d REF_ONLY|%d"
                  % (len(only_port), len(only_ref)))
            for c in only_port[:5]:
                print("  PORT-ONLY EQS=%s INEQS=%s" % c)
            for c in only_ref[:5]:
                print("  REF-ONLY  EQS=%s INEQS=%s" % c)
        return 0 if match else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
