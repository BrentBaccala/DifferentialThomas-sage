r"""
General utilities -- port of ``general`` from the DifferentialThomas Maple
package.

The load-bearing item is :func:`deep_copy` with the reference's ``NoDeepCopy``
semantics: when a branch-system is split, the system / its polynomial objects /
its Janet trees are deep-copied, but any object flagged ``no_deep_copy`` (the
Ranking -- reference ``rankingtable['NoDeepCopy'] := true``) is SHARED between
the copies.  Getting this wrong is either an aliased-mutation correctness bug
or (cloned rankings, cloned remember tables) a memory blowup.

Copy-vs-share rules of the port:

- objects with a true ``no_deep_copy`` attribute -> shared (returned as-is);
- lists / tuples -> element-wise :func:`deep_copy`;
- dicts -> key-preserving element-wise :func:`deep_copy`;
- objects exposing ``deep_copy()`` (PolynomialObject, later DifferentialSystem,
  JanetTree) -> their own ``deep_copy``;
- ``DifferentialPolynomial`` elements, numbers, strings, booleans, None ->
  shared (immutable values; Maple expressions have value semantics, and the
  substrate's elements are immutable).
"""


def lcm_list(l1, l2):
    """`DifferentialThomas/LCMList`: element-wise max of two exponent lists
    (the LCM of the corresponding derivative monomials)."""
    if len(l1) != len(l2):
        raise ValueError("LCMList arguments of different length")
    return [max(a, b) for a, b in zip(l1, l2)]


def list_sum(l):
    """`DifferentialThomas/ListSum`: the sum of a list."""
    return sum(l)


def deep_copy(t):
    """`DifferentialThomas/DeepCopy` with the share-ranking (``NoDeepCopy``)
    rule.  See the module docstring for the copy-vs-share table."""
    if isinstance(t, (list, tuple)):
        out = [deep_copy(x) for x in t]
        return out if isinstance(t, list) else tuple(out)
    if getattr(t, "no_deep_copy", False):
        return t
    if hasattr(t, "deep_copy"):
        return t.deep_copy()
    if isinstance(t, dict):
        return {k: deep_copy(v) for k, v in t.items()}
    # immutable / value-semantics leaves (DifferentialPolynomial, numbers,
    # strings, bools, None, JetVar)
    return t
