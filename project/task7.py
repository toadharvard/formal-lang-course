from itertools import product
from typing import Any, Set

import networkx as nx
from scipy.sparse import csc_array
from project.task6 import cfg_to_weak_normal_form
from pyformlang.cfg import Variable

from collections import defaultdict
from pyformlang.cfg import CFG, Terminal


def revert_cfg(cfg: CFG) -> tuple[dict, dict]:
    terminal_to_variable = defaultdict(set)
    variables_to_head = defaultdict(set)

    for production in cfg.productions:
        body = production.body
        head = production.head

        if len(body) == 2:
            variables_to_head[tuple(body)].add(head)

        elif len(body) == 1 and isinstance(body[0], Terminal):
            terminal_to_variable[body[0]].add(head)

    return terminal_to_variable, variables_to_head


def get_boolean_decompostion(
    wcnf_cfg: CFG,
    graph: nx.DiGraph,
    term_to_vars: dict[Terminal, set[Variable]],
    node_to_idx: dict[Any, int],
) -> dict[Variable, csc_array]:
    n = graph.number_of_nodes()
    boolean_decomposition = {
        var: csc_array((n, n), dtype=bool) for var in wcnf_cfg.variables
    }

    for v1, v2, lb in graph.edges.data("label"):
        term_vars = term_to_vars[Terminal(lb)]
        idx1, idx2 = node_to_idx[v1], node_to_idx[v2]
        for var in term_vars:
            boolean_decomposition[var][idx1, idx2] = True

    for v, var in product(graph.nodes, wcnf_cfg.get_nullable_symbols()):
        idx = node_to_idx[v]
        boolean_decomposition[var][idx, idx] = True

    return boolean_decomposition


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    wcnf_cfg = cfg_to_weak_normal_form(cfg)
    term_to_vars, vars_body_to_head = revert_cfg(wcnf_cfg)

    idx_to_node = {i: node for i, node in enumerate(graph.nodes)}
    node_to_idx = {node: i for i, node in idx_to_node.items()}
    boolean_decomposition = get_boolean_decompostion(
        wcnf_cfg, graph, term_to_vars, node_to_idx
    )

    recently_updated = set(wcnf_cfg.variables)
    while recently_updated:
        updated_var = recently_updated.pop()
        for body, heads in vars_body_to_head.items():
            if updated_var not in body:
                continue

            A, B = body
            new_matrix = boolean_decomposition[A] @ boolean_decomposition[B]
            for head in heads:
                old_matrix = boolean_decomposition[head]
                boolean_decomposition[head] += new_matrix
                if (old_matrix != boolean_decomposition[head]).count_nonzero():
                    recently_updated.add(head)

    start_var = wcnf_cfg.start_symbol
    if start_var not in boolean_decomposition:
        return set()

    nonzero_indices = boolean_decomposition[start_var].nonzero()
    return {
        (idx_to_node[idx1], idx_to_node[idx2])
        for idx1, idx2 in zip(*nonzero_indices)
        if idx_to_node[idx1] in start_nodes and idx_to_node[idx2] in final_nodes
    }
