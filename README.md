# DifferentialThomas-sage

A native Sage/Python port of **Markus Lange-Hegermann's DifferentialThomas**
Maple package — differential Thomas decomposition of polynomial differential
systems into disjoint simple systems.

- Reference implementation: the LGPL Maple source at
  `~/DifferentialThomas/src` (ported module by module).
- Algorithm: T. Bächler, V. Gerdt, M. Lange-Hegermann, D. Robertz,
  *Algorithmic Thomas decomposition of algebraic and differential systems*,
  J. Symbolic Computation 47 (2012) 1233–1266.
- Polynomial substrate: [`sage_differential_polynomial`]
  (`~/sage-differential-polynomial`) — BLAD-native `DifferentialPolynomial`
  elements (leader/initial/separant/prem/differential_prem/resultant/factor
  all delegate there). This package supplies the completion engine: object
  model, rankings, Janet trees, reduction driver, split operators, selection
  strategy, and the work-queue main loop.

## License

LGPL v3 (see `LICENSE`) — this is a derivative work of the LGPL
DifferentialThomas package. It depends only on LGPL components
(`sage_differential_polynomial` wrapping BLAD) and on SageMath. It must not
(and does not) depend on the GPL-v3 `regularchains-sage`.

## Environment

Runs under the conda `sage` environment (SageMath 10.7, Python 3.11):

```
~/miniforge3/envs/sage/bin/sage -python -c "import differentialthomas"
```

**One ring per process**: the substrate supports a single live
`DifferentialPolynomialRing` shape per process (BLAD has one global
differential ring). One decomposition = one ranking = one ring, which matches
the algorithm (branch systems share the ranking; the reference's `DeepCopy`
deliberately shares it via the `NoDeepCopy` flag). Tests that need different
rings run each example in its own subprocess.

## Ranking correspondence (established empirically, 2026-07-08)

The reference's default `DegRevLex` ranking — total order, then *reverse lex
on the differentiation exponents*, then dependent-variable position — is
exactly BLAD's **`degrevlexB`** subranking. BLAD's default `grlexA` does NOT
match (it ranks `u[y] > v[x]` where the reference has `v[x] > u[y]`). All
substrate rings constructed by `differentialthomas.ranking` install
`subranking='degrevlexB'`.

## Testing (Phase 1)

Requires the open-maple oracle (`~/open-maple/src/openmaple`, env
`OPENMAPLE_CAS=sage`) and the reference source (`$DT_SRC`, default
`~/DifferentialThomas/src`):

```
cd ~/DifferentialThomas-sage
~/miniforge3/envs/sage/bin/sage -python -m unittest discover -s tests -v
```

`tests/phase1_worker.py --example ex1|ex2|ex3` runs a single example's
accessor/ranking parity batch by hand.

## Status

Phase 1 (repo, object model, ranking adapter, oracle harness, parity tests).
Later phases: Janet trees (2), reduction driver (3), splits/factorization/
strategy (4), DifferentialSystem + main loop (5), hydrogen
`param_field=False` (6).
