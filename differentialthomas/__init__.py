r"""
differentialthomas -- a native Sage/Python port of Markus Lange-Hegermann's
DifferentialThomas Maple package (differential Thomas decomposition).

Ported from the LGPL Maple source at ``~/DifferentialThomas/src`` (Bächler,
Gerdt, Lange-Hegermann, Robertz: "Algorithmic Thomas decomposition of algebraic
and differential systems", J. Symbolic Computation 47 (2012) 1233-1266).

The polynomial substrate is ``sage_differential_polynomial`` (BLAD-native
``DifferentialPolynomial``); this package supplies the completion engine's
object model, ranking, Janet trees, reduction, splitting and the work-queue
main loop.  Phase 1 provides the object model (PolynomialObject), the ranking
adapter, and the open-maple oracle harness.

License: LGPL v3 (derivative of the LGPL DifferentialThomas package).
"""

from .jetvar import (
    JetVar,
    differential_variable_function,
    differential_variable_derivation,
    differential_variable_order,
)
from .general import lcm_list, list_sum, deep_copy, set_max_order, reset_max_order
from .ranking import Ranking, compute_ranking, get_global_ranking, set_global_ranking
from .polyobj import PolynomialObject, create_polynomial_object, is_differential_field_element
from .derivation import (
    partial_derivative,
    partial_derivative_internal,
    multiple_partial_derivative,
)
from .janet import (
    INFINITY,
    JanetNode,
    JanetTreesObject,
    create_janet_trees_object,
    current_var,
    janet_divisor_in_trees,
    janet_divisor_in_tree,
    janet_divisor_in_tree_rek,
    janet_tree_leafs,
    janet_trees_leafs,
    insert_into_janet_trees,
    insert_into_janet_tree,
    remove_multiplicative_variable_in_subtree,
    remove_elements_in_subtree,
    complete_element_in_janet_tree,
    print_janet_tree,
    print_janet_trees,
)
from .reduction import (
    pseudo_remainder,
    differential_pseudo_reduction,
    reduce_wrt_janet_tree,
    reduce_wrt_janet_trees,
    reduce_nonlinear_tail_wrt_janet_trees,
    LinearCombinationStep,
    verify_linear_combination,
    maple_length,
)
from .polyobj import inconsistent_polynom
from .sorting import (
    compare_polynomials_by_equation_then_ranking,
    insert_into_qlist,
    sort_qlist,
)
from .strategy import (
    remove_leading_field_elements,
    fill_s_by_smallest_leader,
    strategy_smallest_element,
    strategy,
)
from .passivity import criteria
from .system import (
    DifferentialSystem,
    create_differential_system,
    differential_system_janet_trees,
    differential_system_equations,
    differential_system_inequations,
    differential_system_inequation_implied,
    differential_system_reduce_object,
    differential_system_reduce,
    differential_system_normal_form,
    differential_system_tail_reduce,
    differential_system_tail_reduction,
    reduce_inequations_in_differential_system,
    reduce_qlist_in_system,
)
from .factor import factorize
from .splitting import (
    ResultantData,
    initialize_resultant,
    sub_resultant,
    prs_gcd,
    co_factor,
    split_by_initial,
    split_by_squarefree,
    split_by_squarefree_old,
    divide_by_inequation,
    divide_by_inequation_old,
    inequation_lcm,
    reduce_with_side_effects,
    reduction,
)

__all__ = [
    "JetVar",
    "differential_variable_function",
    "differential_variable_derivation",
    "differential_variable_order",
    "lcm_list", "list_sum", "deep_copy",
    "Ranking", "compute_ranking", "get_global_ranking", "set_global_ranking",
    "PolynomialObject", "create_polynomial_object",
    "is_differential_field_element",
    "partial_derivative", "partial_derivative_internal",
    "multiple_partial_derivative",
    "INFINITY", "JanetNode", "JanetTreesObject",
    "create_janet_trees_object", "current_var",
    "janet_divisor_in_trees", "janet_divisor_in_tree",
    "janet_divisor_in_tree_rek", "janet_tree_leafs", "janet_trees_leafs",
    "insert_into_janet_trees", "insert_into_janet_tree",
    "remove_multiplicative_variable_in_subtree",
    "remove_elements_in_subtree", "complete_element_in_janet_tree",
    "print_janet_tree", "print_janet_trees",
    "set_max_order", "reset_max_order",
    "pseudo_remainder", "differential_pseudo_reduction",
    "reduce_wrt_janet_tree", "reduce_wrt_janet_trees",
    "reduce_nonlinear_tail_wrt_janet_trees",
    "LinearCombinationStep", "verify_linear_combination", "maple_length",
    "inconsistent_polynom",
    "compare_polynomials_by_equation_then_ranking", "insert_into_qlist",
    "sort_qlist",
    "remove_leading_field_elements", "fill_s_by_smallest_leader",
    "strategy_smallest_element", "strategy",
    "criteria",
    "DifferentialSystem", "create_differential_system",
    "differential_system_janet_trees", "differential_system_equations",
    "differential_system_inequations",
    "differential_system_inequation_implied",
    "differential_system_reduce_object", "reduce_qlist_in_system",
    "factorize",
    "ResultantData", "initialize_resultant", "sub_resultant", "prs_gcd",
    "co_factor", "split_by_initial", "split_by_squarefree",
    "split_by_squarefree_old", "divide_by_inequation",
    "divide_by_inequation_old", "inequation_lcm", "reduce_with_side_effects",
    "reduction",
    "differential_system_reduce", "differential_system_normal_form",
    "differential_system_tail_reduce", "differential_system_tail_reduction",
    "reduce_inequations_in_differential_system",
    "proc_input", "do_next_step", "differential_thomas_decomposition",
    "equations", "inequations",
]

from .main import (
    proc_input,
    do_next_step,
    differential_thomas_decomposition,
    equations,
    inequations,
)
