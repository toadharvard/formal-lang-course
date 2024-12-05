import networkx as nx
import scipy as sp
from pyformlang.rsa import RecursiveAutomaton
from project.task3 import AdjacencyMatrixFA, intersect_automata
from project.task2 import graph_to_nfa
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
)


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return ebnf_to_rsm(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def rsm_to_nfa(automaton: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    result_nfa = NondeterministicFiniteAutomaton()

    for rule, container in automaton.boxes.items():
        dfa = container.dfa

        start_end_states = dfa.start_states.union(dfa.final_states)
        for state in start_end_states:
            combined_state = State((rule, state))
            if state in dfa.final_states:
                result_nfa.add_final_state(combined_state)
            if state in dfa.start_states:
                result_nfa.add_start_state(combined_state)

        transitions = dfa.to_networkx().edges(data="label")
        for origin, destination, transition_label in transitions:
            initial_state = State((rule, origin))
            target_state = State((rule, destination))
            result_nfa.add_transition(initial_state, transition_label, target_state)

    return result_nfa


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    graph = nx.MultiDiGraph(graph)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_adj_matrix = AdjacencyMatrixFA(graph_nfa)
    rsm_nfa = rsm_to_nfa(rsm)
    rsm_adj_matrix = AdjacencyMatrixFA(rsm_nfa)
    for nonterm in rsm.boxes:
        if graph_adj_matrix.boolean_decomposition.get(nonterm) is None:
            dim = (graph_adj_matrix.states_amount, graph_adj_matrix.states_amount)
            graph_adj_matrix.boolean_decomposition[nonterm] = sp.sparse.csc_matrix(
                dim, dtype=bool
            )
        if rsm_adj_matrix.boolean_decomposition.get(nonterm) is None:
            dim = (rsm_adj_matrix.states_amount, rsm_adj_matrix.states_amount)
            rsm_adj_matrix.boolean_decomposition[nonterm] = sp.sparse.csc_matrix(
                dim, dtype=bool
            )

    prev_nonzero_count = -1
    new_nonzero_count = 0
    while prev_nonzero_count != new_nonzero_count:
        prev_nonzero_count = new_nonzero_count
        automata_intersection = intersect_automata(rsm_adj_matrix, graph_adj_matrix)
        reachability_matrix = automata_intersection.transitive_closure()

        src_indices, dest_indices = reachability_matrix.nonzero()
        for idx in range(len(src_indices)):
            src_idx = src_indices[idx]
            dest_idx = dest_indices[idx]

            src_inner_rsm_state, src_graph_node = automata_intersection.indices_states[
                src_idx
            ].value
            src_symbol, src_rsm_node = src_inner_rsm_state.value
            dest_inner_rsm_state, dest_graph_node = (
                automata_intersection.indices_states[dest_idx].value
            )
            dest_symbol, dest_rsm_node = dest_inner_rsm_state.value

            if src_symbol != dest_symbol:
                continue

            src_rsm_states = rsm.boxes[src_symbol].dfa.start_states
            dest_rsm_states = rsm.boxes[src_symbol].dfa.final_states

            if src_rsm_node in src_rsm_states and dest_rsm_node in dest_rsm_states:
                graph_adj_matrix.boolean_decomposition[src_symbol][
                    graph_adj_matrix.states_indices[src_graph_node],
                    graph_adj_matrix.states_indices[dest_graph_node],
                ] = True

        new_nonzero_count = 0
        for _, adj_matrix in graph_adj_matrix.boolean_decomposition.items():
            nonzero_for_symbol = adj_matrix.count_nonzero()
            new_nonzero_count += nonzero_for_symbol

    cfpq_result = set()
    for start_state in graph_adj_matrix.start_states:
        for final_state in graph_adj_matrix.final_states:
            if graph_adj_matrix.boolean_decomposition[rsm.initial_label][
                graph_adj_matrix.states_indices[start_state],
                graph_adj_matrix.states_indices[final_state],
            ]:
                cfpq_result.add((start_state.value, final_state.value))
    return cfpq_result
