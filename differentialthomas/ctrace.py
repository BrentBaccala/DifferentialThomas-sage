"""Env-gated coefficient-growth / integer-content instrumentation.

Diagnostic for the PDE-staging operand swell (survey cells 1, 2, 25, 26,
28 and cell 0): measures, inside the pseudo-division loop, whether the
swelling remainder carries removable integer content (over Z) or is
content-free -- i.e. whether an in-loop primitive-part pass could have
helped, or the swell is intrinsic to the reduction result.

Off by default; zero overhead beyond one module-flag check per
pseudo-division call.  Enable with::

    DT_CONTENT_TRACE=<file>    append trace lines to <file>
    DT_CONTENT_TRACE=1         write them to stderr

Tuning (all optional):

    DT_CONTENT_TRACE_MIN_NBMON   full coefficient stats every iteration once
                                 the remainder has >= this many monomials
                                 (default 300)
    DT_CONTENT_TRACE_SMALL_EVERY below the threshold, sample full stats every
                                 N-th loop iteration of a call (default 25) --
                                 catches coefficient-bit swell that happens
                                 at small term counts
    DT_CONTENT_TRACE_MAX_NBMON   above this many monomials, skip the term
                                 walk (log nbmon only, skipped=1) so the
                                 instrumentation itself cannot OOM the run
                                 (default 5000000)

Line formats (space-separated key=value; wall = seconds since import):

    PRCALL call=N uu_nbmon=.. vv_nbmon=.. lv_nbmon=.. lv_maxbits=.. wall=..
        -- emitted at pseudo_remainder entry when either operand is at or
           above MIN_NBMON
    PRSTEP call=N it=I dr=D r_nbmon=.. r_maxbits=.. r_contbits=..
           newf_nbmon=.. newf_maxbits=.. wall=..
        -- sampled loop iteration; r_contbits is the bit length of the gcd
           over Z of ALL coefficients of the remainder (1 = content-free,
           0 = zero polynomial); the gcd early-exits once it reaches 1
    SIMPLIFY p_nbmon=.. cont_nbmon=.. cont_maxbits=.. wall=..
        -- SimplifyPolynom removed a content != +-1 (what boundary content
           removal actually reclaims)
    PRHB calls=N wall=..
        -- heartbeat every 10000 pseudo_remainder calls
"""

import math
import os
import sys
import time

_dest = os.environ.get("DT_CONTENT_TRACE")
enabled = bool(_dest)

MIN_NBMON = int(os.environ.get("DT_CONTENT_TRACE_MIN_NBMON", "300"))
SMALL_EVERY = int(os.environ.get("DT_CONTENT_TRACE_SMALL_EVERY", "25"))
MAX_NBMON = int(os.environ.get("DT_CONTENT_TRACE_MAX_NBMON", "5000000"))
HEARTBEAT = 10000

_t0 = time.time()
_out = None
_ncalls = 0


def _write(line):
    global _out
    if _out is None:
        _out = sys.stderr if _dest == "1" else open(_dest, "a", buffering=1)
    _out.write(line + "\n")


def coeff_stats(elt):
    """``(nbmon, max_coeff_bits, content_bits)`` of a substrate element,
    coefficients over Z.  ``content_bits`` is ``gcd(all coeffs).bit_length()``
    (1 = primitive, 0 = zero polynomial); the gcd early-exits at 1, so on a
    content-free polynomial the scan cost is one pass of ``bit_length``."""
    from sage_differential_polynomial import _blad
    n = 0
    mx = 0
    g = 0
    for c, _term in _blad.read_terms(elt._h()):
        n += 1
        if c < 0:
            c = -c
        b = c.bit_length()
        if b > mx:
            mx = b
        if g != 1:
            g = math.gcd(g, c)
    return n, mx, g.bit_length()


def pr_call(uu, vv, lv):
    """Register a pseudo_remainder call; returns the call id (or None when
    the module is disabled -- callers gate on that)."""
    global _ncalls
    _ncalls += 1
    cid = _ncalls
    if cid % HEARTBEAT == 0:
        _write("PRHB calls=%d wall=%.1f" % (cid, time.time() - _t0))
    un = uu.number_of_terms()
    vn = vv.number_of_terms()
    if un >= MIN_NBMON or vn >= MIN_NBMON:
        _ln, lmx, _lc = coeff_stats(lv) if lv.number_of_terms() <= MAX_NBMON \
            else (lv.number_of_terms(), -1, -1)
        _write("PRCALL call=%d uu_nbmon=%d vv_nbmon=%d lv_nbmon=%d "
               "lv_maxbits=%d wall=%.1f"
               % (cid, un, vn, _ln, lmx, time.time() - _t0))
    return cid


def pr_step(cid, it, dr, r, newf):
    """Sampled loop-iteration record (see module docstring for gating)."""
    nb = r.number_of_terms()
    if nb < MIN_NBMON and it % SMALL_EVERY != 0:
        return
    if nb > MAX_NBMON:
        _write("PRSTEP call=%d it=%d dr=%s r_nbmon=%d skipped=1 wall=%.1f"
               % (cid, it, dr, nb, time.time() - _t0))
        return
    _n, mx, cb = coeff_stats(r)
    fn, fmx, _fc = coeff_stats(newf)
    _write("PRSTEP call=%d it=%d dr=%s r_nbmon=%d r_maxbits=%d r_contbits=%d "
           "newf_nbmon=%d newf_maxbits=%d wall=%.1f"
           % (cid, it, dr, nb, mx, cb, fn, fmx, time.time() - _t0))


def simplify_removed(p_before, c):
    """SimplifyPolynom removed a nontrivial content ``c`` from ``p_before``."""
    cn, cmx, _cc = coeff_stats(c) if c.number_of_terms() <= MAX_NBMON \
        else (c.number_of_terms(), -1, -1)
    _write("SIMPLIFY p_nbmon=%d cont_nbmon=%d cont_maxbits=%d wall=%.1f"
           % (p_before.number_of_terms(), cn, cmx, time.time() - _t0))
