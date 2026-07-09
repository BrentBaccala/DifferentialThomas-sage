#!/usr/bin/env sage-python
"""Per-cell PDE staging on the native DifferentialThomas-sage port.

Take the hydrogen ansatz decomposition (29 cells), pick one cell, adjoin the
Schrodinger PDE, and run a FURTHER differential-Thomas decomposition of
cell u {PDE}.  This is the operation open-maple could not finish on the generic
cell (exploded to 81 GB in Sage normal()); the BLAD-native port never calls
Sage normal(), so this tests whether it can.

Usage:
  sage -python cell_plus_pde.py --list                 # decompose ansatz, print cell table
  sage -python cell_plus_pde.py --cell N               # stage the PDE onto cell N and decompose
  sage -python cell_plus_pde.py --cell N --max-steps M # cap DoNextStep calls (probe)

Env: DT_BRANCH_TRACE=1 DT_BRANCH_TRACE_EVERY=200 for live branch/queue stats.
"""
import os
import sys
import time
import resource

sys.path.insert(0, os.path.expanduser('~/DifferentialThomas-sage'))
import differentialthomas as dt

# --- ring / ranking / PDE / ansatz  (identical to joca-thomas-native-dt.sage) --
JETS = ['DDPsi', 'DPsi', 'Psi', 'v', 'r']
ANSATZ_PARAMS = ['v1', 'v2', 'v3', 'v4', 'a0', 'a1', 'b0', 'b1', 'c0', 'c1']
DVAR = JETS + ['E'] + ANSATZ_PARAMS          # E ranked at the bottom of the jets
IVAR = ['x', 'y', 'z']
rk = dt.compute_ranking(IVAR, DVAR)
R = rk.ring

PDE = R('-(Psi[x,x] + Psi[y,y] + Psi[z,z])*r - 2*Psi - 2*E*r*Psi')

ansatz0 = [R('Psi[x] - DPsi*v[x]'),
           R('Psi[y] - DPsi*v[y]'),
           R('Psi[z] - DPsi*v[z]'),
           R('DPsi[x] - DDPsi*v[x]'),
           R('DPsi[y] - DDPsi*v[y]'),
           R('DPsi[z] - DDPsi*v[z]'),
           R('(a0 + a1*v)*DDPsi + (b0 + b1*v)*DPsi + (c0 + c1*v)*Psi'),
           R('v - (v1*x + v2*y + v3*z + v4*r)'),
           R('r^2 - x^2 - y^2 - z^2')]
pconst = [R('%s[%s]' % (p, iv)) for p in ANSATZ_PARAMS for iv in IVAR]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def argval(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    print("=== ansatz decomposition (29-cell target) ===", flush=True)
    t0 = time.time()
    cells = dt.differential_thomas_decomposition(ansatz0 + pconst, [], rk)
    print("ansatz: %d cells in %.0fs, peak RSS %.2f GB"
          % (len(cells), time.time() - t0, rss_gb()), flush=True)

    # characterise each cell
    print("\nidx  #eqs  #ineqs  ineqs(short)")
    table = []
    for i, c in enumerate(cells):
        eqs = dt.differential_system_equations(c)
        ineqs = dt.differential_system_inequations(c)
        table.append((eqs, ineqs))
        short = ", ".join(str(q)[:26] for q in ineqs[:3])
        print("%3d  %4d  %5d   %s" % (i, len(eqs), len(ineqs), short), flush=True)

    if '--list' in sys.argv or '--cell' not in sys.argv:
        return 0

    n = int(argval('--cell'))
    eqs, ineqs = table[n]
    print("\n=== staging PDE onto cell %d (%d eqs, %d ineqs) ==="
          % (n, len(eqs), len(ineqs)), flush=True)
    print("PDE:", PDE, flush=True)

    staged_eqs = list(eqs) + [PDE]
    t1 = time.time()
    all_erels = []
    try:
        subcells = dt.differential_thomas_decomposition(staged_eqs, list(ineqs), rk)
    except BaseException as exc:                       # MemoryError etc.
        dt_wall = time.time() - t1
        print("\n=== cell %d FAILED after %.0fs, peak RSS %.2f GB: %r ==="
              % (n, dt_wall, rss_gb(), exc), flush=True)
        print("SURVEYRESULT cell=%d status=ERROR subcells=0 wall=%.0f "
              "peakrss=%.2f err=%s" % (n, dt_wall, rss_gb(),
                                       type(exc).__name__), flush=True)
        return 1
    dt_wall = time.time() - t1
    print("\n=== RESULT: cell %d u {PDE} -> %d sub-cells in %.0fs, peak RSS %.2f GB ==="
          % (n, len(subcells), dt_wall, rss_gb()), flush=True)
    for j, sc in enumerate(subcells):
        seqs = dt.differential_system_equations(sc)
        sineqs = dt.differential_system_inequations(sc)
        # surface any energy relation E + c = 0 (pure E/coordinate relation)
        e_rels = [str(q) for q in seqs if 'E' in str(q) and 'Psi' not in str(q)
                  and 'DPsi' not in str(q) and 'v' not in str(q).replace('E', '')]
        all_erels += e_rels
        print("  sub-cell %d: %d eqs, %d ineqs%s"
              % (j, len(seqs), len(sineqs),
                 ("  E-rels: " + "; ".join(e_rels)) if e_rels else ""), flush=True)
    print("SURVEYRESULT cell=%d status=OK subcells=%d wall=%.0f peakrss=%.2f "
          "erels=%s" % (n, len(subcells), dt_wall, rss_gb(),
                        ("|".join(all_erels) if all_erels else "none")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
