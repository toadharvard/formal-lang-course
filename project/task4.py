from functools import reduce
from itertools import product
import networkx as nx
from pyformlang.finite_automaton import Symbol
import numpy as np
from numpy import bool_
from scipy.sparse import csr_array, csr_matrix

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA
from numpy.typing import NDArray


def create_array(n: int, m: int, i: int, j: int) -> NDArray[bool_]:
    a = np.zeros((n, m), dtype=bool_)
    a[i, j] = True
    return a


def create_initial_front(n: int, m: int, start_states: set[int]) -> csr_matrix:
    arrays = [[create_array(n, m, i, j)] for (i, j) in start_states]
    return csr_matrix(np.block(arrays))


def get_from_block(M: csr_matrix, n: int, m: int, k: int, i: int, j: int) -> bool_:
    return M[n * k + i, j]


def update_front(
    front: csr_matrix,
    dfa_boolean_decomposition: dict[Symbol, csr_array],
    nfa_boolean_decomposition: dict[Symbol, csr_array],
    k: int,
    n: int,
    symbols: set[Symbol],
) -> csr_matrix:
    decomposed_fronts = {}
    for sym in symbols:
        decomposed_fronts[sym] = front @ nfa_boolean_decomposition[sym]

        for i in range(k):
            decomposed_fronts[sym][n * i : n * (i + 1)] = (
                dfa_boolean_decomposition[sym].T
                @ decomposed_fronts[sym][n * i : n * (i + 1)]
            )
    return reduce(
        lambda x, y: x + y, decomposed_fronts.values(), csr_matrix(front.shape)
    )


def ms_bfs_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    mdfa = AdjacencyMatrixFA(regex_dfa)
    mnfa = AdjacencyMatrixFA(graph_nfa)
    n = len(mdfa.states)
    m = len(mnfa.states)

    nfa_index_to_state = {index: state for state, index in mnfa.states.items()}

    inter_start_states = product(mdfa.start_states, mnfa.start_states)

    k = len(mdfa.start_states) * len(mnfa.start_states)

    symbols = mdfa.boolean_decomposition.keys() & mnfa.boolean_decomposition.keys()

    front = create_initial_front(n, m, inter_start_states)
    visited = front
    while front.count_nonzero() > 0:
        new_front = update_front(
            front,
            mdfa.boolean_decomposition,
            mnfa.boolean_decomposition,
            k,
            n,
            symbols,
        )
        front = new_front > visited
        visited += front

    result = set()
    for dfa_final in mdfa.final_states:
        for i, start in enumerate(mnfa.start_states):
            block = visited[n * i : n * (i + 1)]
            for reached in block.getrow(dfa_final).indices:
                if reached in mnfa.final_states:
                    result.add(
                        (
                            nfa_index_to_state[start],
                            nfa_index_to_state[reached],
                        )
                    )
    return result
